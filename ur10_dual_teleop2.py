"""
ur10_dual_teleop.py
-------------------
UR10 CB2 #1 (Left, Leader)    192.168.1.121
UR10 CB2 #2 (Right, Follower) 192.168.1.120

변경사항:
  - while 루프 → ROS2 타이머 콜백으로 변경 (지연 감소)
  - 키 입력은 별도 스레드 유지

실행 순서:
  터미널 1: ros2 run yur_ros2_driver yur_multi
  터미널 2: ros2 run Y2RobMotion singleArm_motion UR10_left
  터미널 3: ros2 run Y2RobMotion singleArm_motion UR10_right
  터미널 4: python3 ur10_dual_teleop.py

조작:
  f + Enter : Left Freedrive ON
  e + Enter : Left Freedrive OFF
  g + Enter : Left Hand-guiding 모드
  i + Enter : Left Idling 모드
  1 + Enter : 안전 모드 (0.05 rad/step 제한)
  2 + Enter : 일반 모드 (0.15 rad/step 제한)
  3 + Enter : 직접 모드 (제한 없음, 즉시 추종) ★
  q + Enter : 종료
"""

import threading
import socket
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray, String


# ================================================================== #
#  설정
# ================================================================== #
LEADER_NAME   = 'UR10_left'
FOLLOWER_NAME = 'UR10_right'
LEADER_IP     = '192.168.1.121'

HZ = 125
DT = 1.0 / HZ

JOINT_SIGN   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
JOINT_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Q_MIN = -2.0 * np.pi
Q_MAX =  2.0 * np.pi

INIT_INTERP_SEC = 3.0

MODE_SETTINGS = {
    '1': {'max_delta': 0.05,  'label': '안전 (0.05 rad/step)'},
    '2': {'max_delta': 0.15,  'label': '일반 (0.15 rad/step)'},
    '3': {'max_delta': None,  'label': '직접 (즉시 추종) ★'},
}


# ================================================================== #
#  URScript 전송
# ================================================================== #
def send_urscript(ip, script, port=30002):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((ip, port))
            s.sendall((script + '\n').encode())
    except Exception as e:
        print(f'  [URScript] 전송 실패: {e}')

def freedrive_on(ip):
    send_urscript(ip,
        'def fp():\n\tteach_mode()\n\tsleep(3600)\n\tend_teach_mode()\nend\n')

def freedrive_off(ip):
    send_urscript(ip, 'def s():\n\tend_teach_mode()\nend\n')


# ================================================================== #
#  ROS2 노드 — 타이머 기반 제어 루프 포함
# ================================================================== #
class DualUR10Teleop(Node):
    def __init__(self):
        super().__init__('dual_ur10_teleop')
        self._lock = threading.Lock()

        # 상태
        self._leader_q       = [0.0] * 6
        self._follower_q     = [0.0] * 6
        self._leader_ready   = False
        self._follower_ready = False

        # 제어 상태
        self._current_mode    = '1'
        self._freedrive_active = False
        self._running         = False
        self._step            = 0
        self._quit            = False

        # Subscribers
        self.create_subscription(
            Float64MultiArray,
            f'{LEADER_NAME}/currentJ',
            self._leader_cb, 1)
        self.create_subscription(
            Float64MultiArray,
            f'{FOLLOWER_NAME}/currentJ',
            self._follower_cb, 1)

        # Publishers
        self._leader_mode_pub = self.create_publisher(
            String, f'{LEADER_NAME}/cmdMode', 10)
        self._follower_mode_pub = self.create_publisher(
            String, f'{FOLLOWER_NAME}/cmdMode', 10)
        self._follower_joint_pub = self.create_publisher(
            Float64MultiArray, f'{FOLLOWER_NAME}/cmdJoint', 1)

        # 제어 타이머 (초기에는 비활성)
        self._ctrl_timer = None

    # ── Callbacks ─────────────────────────────────────────────── #
    def _leader_cb(self, msg):
        with self._lock:
            if len(msg.data) >= 6:
                self._leader_q     = list(msg.data[:6])
                self._leader_ready = True

    def _follower_cb(self, msg):
        with self._lock:
            if len(msg.data) >= 6:
                self._follower_q     = list(msg.data[:6])
                self._follower_ready = True

    # ── 제어 타이머 콜백 ──────────────────────────────────────── #
    def _control_loop(self):
        """ROS2 타이머 콜백 — spin 과 같은 스레드에서 실행 → 지연 최소"""
        if not self._running:
            return

        with self._lock:
            leader_q   = list(self._leader_q)
            follower_q = list(self._follower_q)
            mode       = self._current_mode

        # 매핑
        target_q = [
            float(np.clip(
                leader_q[i] * JOINT_SIGN[i] + JOINT_OFFSET[i],
                Q_MIN, Q_MAX
            ))
            for i in range(6)
        ]

        # 이동량 제한
        max_delta = MODE_SETTINGS[mode]['max_delta']
        if max_delta is not None:
            target_q = [
                float(np.clip(
                    target_q[i],
                    follower_q[i] - max_delta,
                    follower_q[i] + max_delta
                ))
                for i in range(6)
            ]

        # Follower 명령 전송
        msg = Float64MultiArray()
        msg.data = target_q
        self._follower_joint_pub.publish(msg)

        self._step += 1

        # 1초마다 Teleop 모드 재전송 + 상태 출력
        if self._step % HZ == 0:
            self.set_follower_mode('Teleop')
            err     = [abs(leader_q[i] - follower_q[i]) for i in range(6)]
            max_err = max(err)
            lq = ', '.join([f'{np.degrees(v):+6.1f}°' for v in leader_q])
            fq = ', '.join([f'{np.degrees(v):+6.1f}°' for v in follower_q])
            print(f'\n[{self._step//HZ:4d}s] 오차 max: {max_err:.4f} rad '
                  f'({np.degrees(max_err):.2f}°) | '
                  f'{MODE_SETTINGS[mode]["label"]}')
            print(f'  Leader  : [{lq}]')
            print(f'  Follower: [{fq}]')

    # ── 제어 시작/중지 ────────────────────────────────────────── #
    def start_control(self):
        self._running = True
        self._ctrl_timer = self.create_timer(DT, self._control_loop)
        self.get_logger().info(f'제어 루프 시작 ({HZ}Hz)')

    def stop_control(self):
        self._running = False
        if self._ctrl_timer:
            self._ctrl_timer.cancel()

    # ── Public API ────────────────────────────────────────────── #
    def is_ready(self):
        with self._lock:
            return self._leader_ready and self._follower_ready

    def get_leader_q(self):
        with self._lock:
            return list(self._leader_q)

    def get_follower_q(self):
        with self._lock:
            return list(self._follower_q)

    def set_leader_mode(self, mode):
        msg = String()
        msg.data = mode
        self._leader_mode_pub.publish(msg)

    def set_follower_mode(self, mode):
        msg = String()
        msg.data = mode
        self._follower_mode_pub.publish(msg)

    def send_follower_command(self, q):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q]
        self._follower_joint_pub.publish(msg)

    def set_mode(self, mode):
        with self._lock:
            self._current_mode = mode

    def set_quit(self):
        self._quit = True

    def should_quit(self):
        return self._quit


# ================================================================== #
#  키 입력 스레드
# ================================================================== #
class KeyInput:
    def __init__(self):
        self._cmd  = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                line = input()
                with self._lock:
                    self._cmd = line.strip().lower()
            except EOFError:
                break

    def get(self):
        with self._lock:
            cmd, self._cmd = self._cmd, None
        return cmd


# ================================================================== #
#  초기 보간 이동
# ================================================================== #
def interpolate_to_target(node, executor, target_q, duration):
    import time
    steps   = int(duration / DT)
    start_q = node.get_follower_q()
    print(f'  초기 위치 보간 ({duration:.1f}초)...')
    for step in range(steps + 1):
        t     = step / steps
        alpha = (1 - np.cos(t * np.pi)) / 2.0
        interp = [
            start_q[i] + alpha * (target_q[i] - start_q[i])
            for i in range(6)
        ]
        node.send_follower_command(interp)
        executor.spin_once(timeout_sec=DT)
    print('  보간 완료.')


# ================================================================== #
#  메인
# ================================================================== #
def main():
    import time

    rclpy.init()
    node = DualUR10Teleop()

    # SingleThreadedExecutor — 타이머와 콜백이 같은 스레드에서 실행
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # 연결 대기 (spin_once 루프)
    print('\n두 UR10 연결 대기 중...')
    t_start = time.time()
    while not node.is_ready():
        executor.spin_once(timeout_sec=0.1)
        if time.time() - t_start > 20.0:
            print('타임아웃.')
            rclpy.shutdown()
            return

    leader_q   = node.get_leader_q()
    follower_q = node.get_follower_q()

    print('\n연결 확인 완료.')
    print(f'  Leader  : {[f"{np.degrees(v):+.1f}°" for v in leader_q]}')
    print(f'  Follower: {[f"{np.degrees(v):+.1f}°" for v in follower_q]}')

    max_diff = max(abs(leader_q[i] - follower_q[i]) for i in range(6))
    print(f'  최대 joint 차이: {np.degrees(max_diff):.1f}°')

    if max_diff > 0.2:
        print(f'\n  ⚠️  초기 위치 차이 {np.degrees(max_diff):.1f}°')
        print('     계속: Enter  /  취소: q : ', end='', flush=True)
        if input().strip().lower() == 'q':
            rclpy.shutdown()
            return
        for _ in range(5):
            node.set_follower_mode('Teleop')
            time.sleep(0.05)
        interpolate_to_target(
            node, executor, [
                float(np.clip(
                    leader_q[i] * JOINT_SIGN[i] + JOINT_OFFSET[i],
                    Q_MIN, Q_MAX))
                for i in range(6)
            ], INIT_INTERP_SEC)
    else:
        for _ in range(5):
            node.set_follower_mode('Teleop')
            time.sleep(0.05)
        executor.spin_once(timeout_sec=0.3)

    print('\n' + '=' * 55)
    print('  UR10 Dual Teleop 시작 (타이머 기반)')
    print(f'  {HZ}Hz | Leader: {LEADER_NAME} | Follower: {FOLLOWER_NAME}')
    print('-' * 55)
    print('  f : Left Freedrive ON   e : Freedrive OFF')
    print('  g : Hand-guiding        i : Idling')
    print('  1 : 안전 (0.05 rad/step)')
    print('  2 : 일반 (0.15 rad/step)')
    print('  3 : 직접 (즉시 추종) ★')
    print('  q : 종료')
    print('=' * 55 + '\n')

    # 제어 타이머 시작
    node.start_control()

    key              = KeyInput()
    freedrive_active = False

    try:
        while rclpy.ok() and not node.should_quit():
            # ROS2 spin (타이머 콜백 실행)
            executor.spin_once(timeout_sec=0.05)

            # 키 입력 처리 (논블로킹)
            cmd = key.get()
            if cmd is None:
                continue

            if cmd == 'q':
                node.set_quit()

            elif cmd == 'f':
                freedrive_on(LEADER_IP)
                freedrive_active = True
                print('>>> Left Freedrive ON — 손으로 잡고 움직이세요')

            elif cmd == 'e':
                freedrive_off(LEADER_IP)
                freedrive_active = False
                print('>>> Left Freedrive OFF')

            elif cmd == 'g':
                node.set_leader_mode('Guiding')
                print('>>> Left Hand-guiding')

            elif cmd == 'i':
                node.set_leader_mode('Idling')
                if freedrive_active:
                    freedrive_off(LEADER_IP)
                    freedrive_active = False
                print('>>> Left Idling')

            elif cmd in MODE_SETTINGS:
                node.set_mode(cmd)
                print(f'>>> 모드 {cmd}: {MODE_SETTINGS[cmd]["label"]}')

    except KeyboardInterrupt:
        print('\nCtrl+C.')

    finally:
        print('정리 중...')
        node.stop_control()
        if freedrive_active:
            freedrive_off(LEADER_IP)
        node.set_follower_mode('Idling')
        node.set_leader_mode('Idling')
        executor.spin_once(timeout_sec=0.3)
        rclpy.shutdown()
        print('정상 종료.')


if __name__ == '__main__':
    main()
