# ACT 학습 완료 후 UR10 추론(Inference) 실행 가이드

## 개요

LeRobot은 SO-100, ALOHA 등 자체 지원 하드웨어에 대해서는 `lerobot-record` 명령으로 바로 추론을 실행할 수 있습니다.
그러나 **UR10 CB2는 LeRobot 지원 하드웨어가 아니므로** 커스텀 추론 코드를 작성해야 합니다.

추론 흐름:
```
카메라(cam_left, cam_right) + Follower joint state
          ↓
    ACT 정책 (체크포인트 로드)
          ↓
    action (6 joint angles) 예측
          ↓
    ROS2 토픽 → UR10_right/cmdJoint 발행
          ↓
    ur10_motion.cpp control_teleop() 실행
```

---

## Step 1: 체크포인트 확인

```bash
# 학습 완료 후 체크포인트 목록
ls ~/outputs/train/act_ur10/checkpoints/
# 020000/
# 040000/
# 060000/
# 080000/
# 100000/
# last/    ← 최종 체크포인트 (심볼릭 링크)

# last 체크포인트 내용
ls ~/outputs/train/act_ur10/checkpoints/last/pretrained_model/
# config.json
# model.safetensors
# train_config.json
```

---

## Step 2: 학습 loss 확인

추론 전에 학습이 잘 됐는지 확인합니다.

```bash
# 학습 로그 확인
cat ~/outputs/train/act_ur10/logs/*.txt | tail -50

# 또는 tensorboard
pip install tensorboard
tensorboard --logdir ~/outputs/train/act_ur10/logs
# 브라우저에서 http://localhost:6006
```

**정상 학습 기준:**
- `loss`가 초기 2.0 이상 → 100k 스텝 후 0.1 이하로 감소
- `kl_loss`가 안정적으로 작게 유지 (0.01 이하)

---

## Step 3: 커스텀 추론 스크립트 — `act_inference.py`

UR10용 커스텀 추론 스크립트입니다.
학습 시 사용한 관측값(카메라 2대 + joint state)을 실시간으로 읽어 행동을 예측하고 ROS2로 발행합니다.

```python
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
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ============================================================
# 설정
# ============================================================
CHECKPOINT_PATH = '/home/ur12e/outputs/train/act_ur10/checkpoints/last/pretrained_model'
FOLLOWER_NAME   = 'UR10_right'
IMAGE_HEIGHT    = 480
IMAGE_WIDTH     = 640
DEVICE          = 'cuda'
INFER_HZ        = 30     # 추론 주파수 (학습 Hz와 동일)
N_ACTION_STEPS  = 100    # ACT chunk_size (학습 설정과 동일)


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
        self.running = False

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
            self.get_logger().warn(f'이미지 변환 오류 ({key}): {e}',
                                   throttle_duration_sec=3.0)

    def is_ready(self):
        with self._lock:
            return all(v is not None for v in self.latest.values())

    def get_observation(self):
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
    print(f'[INFO] ACT 정책 로드 완료: {checkpoint_path}')
    return policy


# ============================================================
# 관측값 → 텐서 변환
# ============================================================
def obs_to_tensor(obs: dict, device: str) -> dict:
    # joint state: (6,) → (1, 6)
    state = torch.from_numpy(obs['follower_joints']).unsqueeze(0).to(device)

    # 이미지: (H, W, 3) uint8 → (1, 3, H, W) float [0,1]
    def img_to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)

    return {
        'observation.state':                  state,
        'observation.images.cam_left':        img_to_tensor(obs['cam_left']),
        'observation.images.cam_right':       img_to_tensor(obs['cam_right']),
    }


# ============================================================
# 메인 추론 루프
# ============================================================
def main():
    rclpy.init()
    node = rclpy.create_node('act_inference_helper')
    inf_node = InferenceNode()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(inf_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print('\n[대기] 카메라 및 joint state 수신 중...')
    while not inf_node.is_ready():
        time.sleep(0.1)
    print('[완료] 모든 센서 수신 확인\n')

    print('[로드] ACT 정책 로드 중...')
    policy = load_policy(CHECKPOINT_PATH, DEVICE)

    print('\n' + '='*50)
    print('  UR10 ACT 추론 준비 완료')
    print(f'  체크포인트: {CHECKPOINT_PATH}')
    print(f'  추론 Hz: {INFER_HZ}')
    print('-'*50)
    print('  Enter : 추론 시작/정지')
    print('  q     : 종료')
    print('='*50 + '\n')

    running    = False
    action_buf = []   # action chunk 버퍼
    period     = 1.0 / INFER_HZ

    try:
        while True:
            cmd = input().strip().lower()

            if cmd == 'q':
                break

            elif cmd == '':
                running = not running
                if running:
                    inf_node.set_mode('Teleop')
                    print('[START] 추론 시작 — 로봇이 움직입니다!')
                else:
                    inf_node.set_mode('Idling')
                    action_buf.clear()
                    print('[STOP] 추론 정지')

            if running:
                # action chunk 소진 시 새로 예측
                if len(action_buf) == 0:
                    obs = inf_node.get_observation()
                    obs_tensor = obs_to_tensor(obs, DEVICE)

                    with torch.inference_mode():
                        # ACT: action chunk (N_ACTION_STEPS, 6) 예측
                        action = policy.select_action(obs_tensor)
                        # action shape: (1, N_ACTION_STEPS, 6) 또는 (N_ACTION_STEPS, 6)
                        if action.dim() == 3:
                            action = action.squeeze(0)
                        action_buf = action.cpu().numpy().tolist()

                # 버퍼에서 하나씩 꺼내어 발행
                if action_buf:
                    joint_cmd = action_buf.pop(0)
                    inf_node.send_joint_command(joint_cmd)

                time.sleep(period)

    except KeyboardInterrupt:
        print('\nCtrl+C')

    finally:
        inf_node.set_mode('Idling')
        executor.shutdown()
        rclpy.shutdown()
        print('종료')


if __name__ == '__main__':
    main()
```

---

## Step 4: 추론 실행 순서

### 사전 준비

```bash
# 터미널 1: ROS2 드라이버
ros2 run yur_ros2_driver yur_multi
# → Polyscope에서 Play(▶) 버튼 누르기

# 터미널 2: Follower (Right) 모션 노드만 실행
ros2 run Y2RobMotion singleArm_motion UR10_right → '1' 입력

# 터미널 3: 카메라 (좌)
ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video2" \
    -r __ns:="/left" -r __node:="camera"

# 터미널 4: 카메라 (우)
ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video0" \
    -r __ns:="/right" -r __node:="camera"
```

### 추론 실행

```bash
source ~/lerobot_env/bin/activate
python3 act_inference.py
```

### 체크포인트 선택

최종 체크포인트 대신 중간 체크포인트를 사용하려면 `act_inference.py` 내 `CHECKPOINT_PATH` 수정:

```python
# 최종
CHECKPOINT_PATH = '/home/ur12e/outputs/train/act_ur10/checkpoints/last/pretrained_model'

# 특정 스텝
CHECKPOINT_PATH = '/home/ur12e/outputs/train/act_ur10/checkpoints/080000/pretrained_model'
```

---

## Step 5: 성능 평가 기준

### 정상 동작 확인

| 항목 | 기준 |
|------|------|
| 추론 지연 | 30Hz 유지 |
| 초기 동작 | 작업 시작 위치로 이동 |
| 물체 접근 | 목표 물체 방향으로 일관되게 이동 |
| 성공률 | 10회 시도 중 6회 이상 성공 |

### 성능이 낮을 때

| 증상 | 원인 | 해결 |
|------|------|------|
| 로봇이 움직이지 않거나 랜덤하게 움직임 | 학습 부족 또는 데이터 불균형 | steps 증가 (200k), 데이터 추가 수집 |
| 물체 위치가 조금만 달라도 실패 | 데이터 다양성 부족 | 다양한 물체 위치에서 재수집 |
| 특정 자세에서 멈춤 | chunk_size 불일치 | 학습 chunk_size 확인 |
| 진동/떨림 | temporal ensemble 미사용 | `temporal_ensemble_coeff` 설정 |

### Temporal Ensemble 활성화 (떨림 감소)

```bash
lerobot-train \
    ... \
    --policy.temporal_ensemble_coeff=0.01
```

또는 추론 코드에서 직접 구현:

```python
# 여러 chunk의 예측을 지수 가중 평균으로 합산
# → 동작이 부드러워짐
```

---

## Step 6: 추가 데이터 수집 후 재학습

성능이 부족하면 데이터를 추가 수집하고 기존 데이터와 합쳐 재학습합니다.

```bash
# 새 세션 수집
python3 data_collector2A.py

# 기존 세션들 + 새 세션 병합 후 재변환
python3 merge_and_convert.py \
    ~/outputs/lerobot_data/session_기존_A \
    ~/outputs/lerobot_data/session_기존_B \
    ~/outputs/lerobot_data/session_새로운_C \
    --output /mnt/lerobot_data/lerobot_data/lerobot_v30_v2

# 후처리 (Step 4-1 ~ 4-5 반복)
# ...

# 재학습 (기존 체크포인트에서 이어서)
lerobot-train \
    --dataset.repo_id=local/ur10_cup_task \
    --dataset.root=/mnt/lerobot_data/lerobot_data/lerobot_v30_v2 \
    --dataset.revision=main \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=/home/ur12e/outputs/train/act_ur10_v2 \
    --job_name=act_ur10_cup_task_v2 \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --batch_size=8
```

---

## 체크포인트 구조 참고

```
~/outputs/train/act_ur10/
  checkpoints/
    020000/
      pretrained_model/
        config.json          ← 모델 구조 설정
        model.safetensors    ← 모델 가중치
        train_config.json    ← 학습 설정 전체
    040000/
    ...
    last/                    ← 최신 체크포인트 (심볼릭 링크)
  logs/
    train_*.log              ← 학습 로그 (loss 등)
```
