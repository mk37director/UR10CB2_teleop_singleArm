"""
act_inference.py
================
학습된 ACT 정책으로 UR10 Follower(Right)를 제어합니다.

실행 순서:
  터미널 1: ros2 run yur_ros2_driver yur_multi
  터미널 2: ros2 run Y2RobMotion singleArm_motion UR10_right  → '1' 입력
  터미널 3: ros2 run v4l2_camera v4l2_camera_node --ros-args \
              -p video_device:="/dev/video2" -r __ns:="/left" -r __node:="camera"
  터미널 4: ros2 run v4l2_camera v4l2_camera_node --ros-args \
              -p video_device:="/dev/video0" -r __ns:="/right" -r __node:="camera"
  터미널 5: python3 act_inference.py

조작:
  Enter  : 추론 시작/정지 토글
  q      : 종료
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ============================================================
# 설정  ← 필요시 수정
# ============================================================
CHECKPOINT_PATH = '/home/ur12e/outputs/train/act_ur10/checkpoints/last/pretrained_model'
FOLLOWER_NAME   = 'UR10_right'
IMAGE_HEIGHT    = 480
IMAGE_WIDTH     = 640
DEVICE          = 'cuda'
INFER_HZ        = 30      # 추론 주파수 (학습 Hz와 동일)
N_ACTION_STEPS  = 100     # ACT chunk_size (학습 설정과 동일)


# ============================================================
# ROS2 노드
# ============================================================
class InferenceNode(Node):
    def __init__(self):
        super().__init__('act_inference')
        self.bridge = CvBridge()
        self._lock  = threading.Lock()

        self.latest = {
            'follower_joints': None,
            'cam_left':        None,
            'cam_right':       None,
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

        self.get_logger().info('InferenceNode 초기화 완료')

    def _cb_joints(self, msg):
        with self._lock:
            if len(msg.data) >= 6:
                self.latest['follower_joints'] = np.array(msg.data[:6], dtype=np.float32)

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
                self.latest[key] = img

        except Exception as e:
            self.get_logger().warn(
                f'이미지 변환 오류 ({key}): {e}',
                throttle_duration_sec=3.0)

    def is_ready(self) -> bool:
        with self._lock:
            return all(v is not None for v in self.latest.values())

    def get_observation(self) -> dict:
        with self._lock:
            return {k: v.copy() for k, v in self.latest.items()}

    def send_joint_command(self, angles):
        msg = Float64MultiArray()
        msg.data = [float(a) for a in angles]
        self._joint_pub.publish(msg)

    def set_mode(self, mode: str):
        msg = String()
        msg.data = mode
        self._mode_pub.publish(msg)


# ============================================================
# ACT 정책 로드
# ============================================================
def load_policy(checkpoint_path: str, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    print(f'[INFO] ACT 정책 로드 완료')
    print(f'       경로: {checkpoint_path}')
    return policy


# ============================================================
# 관측값 → 텐서 변환
# ============================================================
def obs_to_tensor(obs: dict, device: str) -> dict:
    # joint state: (6,) → (1, 6)
    state = torch.from_numpy(obs['follower_joints']).unsqueeze(0).to(device)

    # 이미지: (H, W, 3) uint8 → (1, 3, H, W) float32 [0, 1]
    def img_to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)

    return {
        'observation.state':                 state,
        'observation.images.cam_left':       img_to_tensor(obs['cam_left']),
        'observation.images.cam_right':      img_to_tensor(obs['cam_right']),
    }


# ============================================================
# 키 입력 스레드 (non-blocking)
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
# 메인 추론 루프
# ============================================================
def main():
    rclpy.init()
    inf_node = InferenceNode()

    executor = SingleThreadedExecutor()
    executor.add_node(inf_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 센서 수신 대기
    print('\n[대기] 카메라 및 joint state 수신 중...', end='', flush=True)
    while not inf_node.is_ready():
        print('.', end='', flush=True)
        time.sleep(0.2)
    print(' 완료!\n')

    # 정책 로드
    policy = load_policy(CHECKPOINT_PATH, DEVICE)

    print('\n' + '='*55)
    print('  UR10 ACT 추론 준비 완료')
    print(f'  Follower : {FOLLOWER_NAME}')
    print(f'  추론 Hz  : {INFER_HZ}')
    print(f'  Chunk    : {N_ACTION_STEPS} steps')
    print('-'*55)
    print('  Enter : 추론 시작 / 정지 토글')
    print('  q     : 종료')
    print('='*55 + '\n')

    running    = False
    action_buf = []          # action chunk 버퍼
    period     = 1.0 / INFER_HZ
    step_count = 0

    key = KeyInput()

    try:
        while rclpy.ok():
            # 키 입력 처리
            cmd = key.get()
            if cmd is not None:
                if cmd == 'q':
                    break
                elif cmd == '':   # Enter
                    running = not running
                    if running:
                        action_buf.clear()
                        inf_node.set_mode('Teleop')
                        step_count = 0
                        print('[START ▶] 추론 시작 — 로봇이 움직입니다!')
                    else:
                        inf_node.set_mode('Idling')
                        action_buf.clear()
                        print('[STOP  ■] 추론 정지')

            if not running:
                time.sleep(0.05)
                continue

            t_start = time.time()

            # action chunk 소진 시 새로 예측
            if len(action_buf) == 0:
                obs        = inf_node.get_observation()
                obs_tensor = obs_to_tensor(obs, DEVICE)

                with torch.inference_mode():
                    action = policy.select_action(obs_tensor)
                    # shape: (1, N, 6) 또는 (N, 6)
                    if action.dim() == 3:
                        action = action.squeeze(0)
                    action_buf = action.cpu().numpy().tolist()

            # 버퍼에서 하나씩 꺼내어 발행
            if action_buf:
                joint_cmd = action_buf.pop(0)
                inf_node.send_joint_command(joint_cmd)
                step_count += 1

            # 1초마다 상태 출력
            if step_count % INFER_HZ == 0:
                obs = inf_node.get_observation()
                j = obs['follower_joints']
                print(f'  [{step_count//INFER_HZ:4d}s] '
                      f'buf={len(action_buf):3d} | '
                      f'j=[{", ".join(f"{v:+.2f}" for v in j)}]')

            # 주기 맞추기
            elapsed = time.time() - t_start
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\nCtrl+C 감지')

    finally:
        print('정리 중...')
        inf_node.set_mode('Idling')
        time.sleep(0.2)
        executor.shutdown()
        rclpy.shutdown()
        print('정상 종료')


if __name__ == '__main__':
    main()
