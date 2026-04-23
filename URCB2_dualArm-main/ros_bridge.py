"""
ros_bridge.py
=============
Python 3.10 + ROS2 환경에서 실행.
카메라/joint state를 수신하여 act_server.py로 전송하고
받은 action을 UR10에 발행합니다.

실행:
  source /opt/ros/humble/setup.bash
  source /mnt/lerobot_data/lerobot_ros_env/bin/activate
  python3 ros_bridge.py

조작:
  Enter : 추론 시작/정지 토글
  q     : 종료
"""

import socket
import json
import struct
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# ============================================================
# 설정
# ============================================================
FOLLOWER_NAME = 'UR10_right'
IMAGE_HEIGHT  = 480
IMAGE_WIDTH   = 640
INFER_HZ      = 30
HOST          = '127.0.0.1'
PORT          = 55555


# ============================================================
# 소켓 유틸
# ============================================================
def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('소켓 연결 끊김')
        buf += chunk
    return buf

def send_msg(sock, data: bytes):
    sock.sendall(struct.pack('>I', len(data)) + data)

def recv_msg(sock) -> bytes:
    raw_len = recv_exact(sock, 4)
    msg_len = struct.unpack('>I', raw_len)[0]
    return recv_exact(sock, msg_len)


# ============================================================
# ROS2 노드
# ============================================================
class BridgeNode(Node):
    def __init__(self):
        super().__init__('ros_bridge')
        self.bridge = CvBridge()
        self._lock  = threading.Lock()

        self.latest = {
            'joints':    None,
            'cam_left':  None,
            'cam_right': None,
        }

        # Subscribers
        self.create_subscription(
            Float64MultiArray, f'{FOLLOWER_NAME}/currentJ',
            self._cb_joints, 1)
        self.create_subscription(
            Image, '/left/image_raw',
            lambda msg: self._cb_image(msg, 'cam_left'), 1)
        self.create_subscription(
            Image, '/right/image_raw',
            lambda msg: self._cb_image(msg, 'cam_right'), 1)

        # Publishers
        self._joint_pub = self.create_publisher(
            Float64MultiArray, f'{FOLLOWER_NAME}/cmdJoint', 1)
        self._mode_pub  = self.create_publisher(
            String, f'{FOLLOWER_NAME}/cmdMode', 10)

    def _cb_joints(self, msg):
        with self._lock:
            if len(msg.data) >= 6:
                self.latest['joints'] = list(msg.data[:6])

    def _cb_image(self, msg, key):
        try:
            enc = msg.encoding
            if enc in ('yuv422_yuy2', 'yuv422', 'yuyv', 'YUYV'):
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                img = cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUYV)
            elif enc in ('mono8', '8UC1'):
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            if img.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

            with self._lock:
                self.latest[key] = img.flatten().tolist()

        except Exception as e:
            self.get_logger().warn(f'이미지 오류 ({key}): {e}',
                                   throttle_duration_sec=3.0)

    def is_ready(self):
        with self._lock:
            return all(v is not None for v in self.latest.values())

    def get_obs(self):
        with self._lock:
            return {k: list(v) for k, v in self.latest.items()}

    def send_joint(self, angles):
        msg = Float64MultiArray()
        msg.data = [float(a) for a in angles]
        self._joint_pub.publish(msg)

    def set_mode(self, mode):
        msg = String()
        msg.data = mode
        self._mode_pub.publish(msg)


# ============================================================
# 키 입력 스레드
# ============================================================
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


# ============================================================
# 메인
# ============================================================
def main():
    rclpy.init()
    node = BridgeNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 센서 대기
    print('[대기] 카메라 및 joint state 수신 중...', end='', flush=True)
    while not node.is_ready():
        print('.', end='', flush=True)
        time.sleep(0.2)
    print(' 완료!\n')

    # act_server 연결
    print(f'[연결] act_server({HOST}:{PORT}) 연결 중...')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((HOST, PORT))
            break
        except ConnectionRefusedError:
            print('  act_server 대기 중...', end='\r')
            time.sleep(1.0)

    # ping 확인
    send_msg(sock, json.dumps({'cmd': 'ping'}).encode())
    resp = json.loads(recv_msg(sock).decode())
    print(f'[연결] act_server 응답: {resp["status"]}\n')

    print('='*55)
    print('  UR10 ACT 추론 (ROS Bridge)')
    print(f'  Follower : {FOLLOWER_NAME}')
    print(f'  추론 Hz  : {INFER_HZ}')
    print('-'*55)
    print('  Enter : 추론 시작 / 정지 토글')
    print('  q     : 종료')
    print('='*55 + '\n')

    running    = False
    period     = 1.0 / INFER_HZ
    step_count = 0
    key        = KeyInput()

    try:
        while rclpy.ok():
            cmd = key.get()
            if cmd is not None:
                if cmd == 'q':
                    break
                elif cmd == '':
                    running = not running
                    if running:
                        # 버퍼 리셋
                        send_msg(sock, json.dumps({'cmd': 'reset'}).encode())
                        recv_msg(sock)
                        node.set_mode('Teleop')
                        step_count = 0
                        print('[START ▶] 추론 시작!')
                    else:
                        node.set_mode('Idling')
                        print('[STOP  ■] 추론 정지')

            if not running:
                time.sleep(0.05)
                continue

            t_start = time.time()

            # 관측값 전송 및 action 수신
            obs = node.get_obs()
            req = {
                'cmd':       'infer',
                'joints':    obs['joints'],
                'cam_left':  obs['cam_left'],
                'cam_right': obs['cam_right'],
            }
            send_msg(sock, json.dumps(req).encode())
            resp = json.loads(recv_msg(sock).decode())

            if resp['status'] == 'ok':
                node.send_joint(resp['action'])
                step_count += 1

                if step_count % INFER_HZ == 0:
                    j = obs['joints']
                    print(f'  [{step_count//INFER_HZ:4d}s] '
                          f'buf={resp["buf_len"]:3d} | '
                          f'j=[{", ".join(f"{v:+.2f}" for v in j)}]')

            elapsed = time.time() - t_start
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\nCtrl+C')
    finally:
        print('정리 중...')
        node.set_mode('Idling')
        sock.close()
        executor.shutdown()
        rclpy.shutdown()
        print('종료')


if __name__ == '__main__':
    main()
