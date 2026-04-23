# UR10 Dual Arm - 카메라 연동 및 데이터 수집 가이드

## 시스템 구성

| 항목 | 내용 |
|------|------|
| 로봇 | UR10 CB2 × 2대 (Leader: UR10_left, Follower: UR10_right) |
| 카메라 | Logitech C922 Pro Stream × 2대 |
| OS | Ubuntu 22.04 |
| 미들웨어 | ROS2 |
| 학습 프레임워크 | LeRobot |

---

## 1. 카메라 디바이스 확인

터미널에서 연결된 카메라 장치를 확인합니다.

```bash
v4l2-ctl --list-devices
```

**예상 출력:**
```
C922 Pro Stream Webcam (usb-0000:07:00.1-3):
    /dev/video2
    /dev/video3
    /dev/media1

C922 Pro Stream Webcam (usb-0000:07:00.1-4):
    /dev/video0
    /dev/video1
    /dev/media0
```

**디바이스 매핑:**

| 역할 | 디바이스 | ROS2 네임스페이스 | 토픽 |
|------|----------|------------------|------|
| 왼쪽 카메라 | `/dev/video2` | `/left` | `/left/camera/image_raw` |
| 오른쪽 카메라 | `/dev/video0` | `/right` | `/right/camera/image_raw` |

> **주의:** `v4l2-ctl --list-devices` 출력에서 각 카메라의 첫 번째 디바이스 번호를 사용합니다.  
> (`/dev/video3`, `/dev/video1`은 메타데이터용이므로 사용하지 않습니다.)

---

## 2. 카메라 ROS2 노드 실행

카메라마다 별도 터미널을 열어 실행합니다.

### 오른쪽 카메라 (터미널 1)

```bash
taskset -c 7 ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:="/dev/video0" \
  -r __ns:="/right" \
  -r __node:="camera"
```

### 왼쪽 카메라 (터미널 2)

```bash
taskset -c 8 ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:="/dev/video2" \
  -r __ns:="/left" \
  -r __node:="camera"
```

### 카메라 수신 확인

두 카메라가 정상적으로 publish되는지 확인합니다.

```bash
# 토픽 목록 확인
ros2 topic list | grep camera

# 왼쪽 카메라 수신 확인
ros2 topic echo /left/camera/image_raw --once

# 오른쪽 카메라 수신 확인
ros2 topic echo /right/camera/image_raw --once

# 카메라 Hz 확인
ros2 topic hz /left/camera/image_raw
ros2 topic hz /right/camera/image_raw
```

### 카메라 영상 시각화 (선택)

```bash
ros2 run rqt_gui rqt_gui
```

rqt에서 `Plugins → Visualization → Image View` 선택 후 토픽을 `/left/camera/image_raw` 또는 `/right/camera/image_raw`로 설정합니다.

---

## 3. 전체 시스템 실행 순서

데이터 수집 전 아래 순서대로 모든 노드를 실행합니다.

```
터미널 1: 오른쪽 카메라 노드
터미널 2: 왼쪽 카메라 노드
터미널 3: dualArm_motion 노드  (기존 ROS2 시스템)
터미널 4: data_collector.py
```

### 터미널 3 - dualArm_motion 노드

```bash
ros2 run Y2RobMotion dualArm_motion
```

실행 후 양쪽 팔의 joint state가 수신되면 시작 옵션을 선택합니다.

```
1: Start both arms   ← 선택
2: Start left arm only
3: Start right arm only
0: Quit
```

---

## 4. 데이터 수집 실행

### 터미널 4 - 데이터 수집 노드
@ /ur10_ws/URCB2_dualArm-main$

```bash
taskset -c 9 python3 data_collector2A.py
```

### 시작 시 상태 확인

`[I]`를 입력하여 모든 데이터 소스가 수신되는지 먼저 확인합니다.

```
명령 (S/E/D/I/Q): I

--- 현재 상태 ---
  Leader mode  : Guiding
  Leader  J    : [ 10.2  -45.1  90.3 ... ] [deg]
  Follower J   : [ 10.1  -45.0  90.2 ... ] [deg]
  Cam Left     : OK (480, 640, 3)            ← 정상
  Cam Right    : OK (480, 640, 3)            ← 정상
  에피소드 수  : 0
-----------------
```

> **Cam Left / Cam Right 가 "미수신" 으로 나오면** 카메라 노드가 실행 중인지 확인하세요.

### 에피소드 수집 방법

| 명령 | 동작 |
|------|------|
| `S` | 에피소드 녹화 시작 |
| `E` | 에피소드 종료 + 성공으로 저장 |
| `D` | 에피소드 버림 (실패 / 오동작) |
| `I` | 현재 상태 확인 |
| `Q` | 전체 종료 및 pkl 파일 저장 |

### 수집 절차 (에피소드 1회)

```
1. UR10_left 를 Guiding 모드로 설정
2. [S] 입력 → 녹화 시작
3. 리더 팔(UR10_left)을 손으로 잡고 컵을 집어 파란 박스로 이동
4. 동작 완료 후 [E] 입력 → 저장
5. 다음 에피소드를 위해 초기 자세로 복귀
6. 1~5 반복 (목표: 50~100 에피소드)
```

> 오동작이 발생했을 때는 [E] 대신 **[D]** 를 입력하여 해당 에피소드를 버립니다.

### 수집 완료 후 저장

```
명령 (S/E/D/I/Q): Q
```

저장 경로 예시:
```
./collected_data/session_20240410_143022/
├── episode_000000.pkl
├── episode_000001.pkl
├── ...
└── summary.json
```

---

## 5. LeRobot 포맷 변환

수집이 완료되면 pkl 파일을 LeRobot 포맷으로 변환합니다.

```bash
python3 convert_to_lerobot.py ./collected_data/session_20240410_143022
```

### 출력 구조

```
session_20240410_143022/
└── lerobot_dataset/
    ├── meta/
    │   ├── info.json       ← 데이터셋 메타정보
    │   ├── stats.json      ← 정규화 통계 (mean, std, min, max)
    │   └── tasks.json      ← task 설명
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.cam_left_episode_000000.mp4
            ├── observation.images.cam_right_episode_000000.mp4
            └── ...
```

### 변환 결과 확인

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('./collected_data/session_20240410_143022/lerobot_dataset/data/chunk-000/episode_000000.parquet')
print(df.columns.tolist())
print(df.head(3))
print(df.dtypes)
"
```

**예상 컬럼:**
```
observation.state          # Follower joint angles (6,) rad
action                     # Leader  joint angles (6,) rad
observation.images.cam_left   # 비디오 내 프레임 인덱스
observation.images.cam_right  # 비디오 내 프레임 인덱스
timestamp
frame_index
episode_index
task_index
index
```

---

## 6. 토픽 전체 요약

```
[Joint - ur10_motion.cpp]
  UR10_left/currentJ    →  action            (Float64MultiArray, 6DoF, rad)
  UR10_right/currentJ   →  observation.state  (Float64MultiArray, 6DoF, rad)
  UR10_right/currentP   →  Cartesian pose    (optional, mm + rad)
  UR10_left/ctlMode     →  리더 제어모드 확인  (String)

[Camera - v4l2_camera]
  /left/camera/image_raw   →  왼쪽 카메라  (/dev/video2)
  /right/camera/image_raw  →  오른쪽 카메라 (/dev/video0)
```

---

## 7. 트러블슈팅

### 카메라 토픽이 수신되지 않을 때

```bash
# v4l2_camera 패키지 설치 확인
ros2 pkg list | grep v4l2

# 미설치 시
sudo apt install ros-humble-v4l2-camera

# 디바이스 권한 확인
ls -l /dev/video0 /dev/video2
# crw-rw----+ 1 root video 이어야 함

# 권한 부여 (필요 시)
sudo chmod 666 /dev/video0 /dev/video2
```

### 카메라가 뒤바뀐 경우

`v4l2-ctl --list-devices` 출력의 USB 포트 번호로 구분합니다.

```bash
# 각 카메라 영상 미리 확인
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite('test_video0.jpg', frame)
cap.release()
print('video0 캡처 완료')
"
```

저장된 이미지로 어느 쪽 카메라인지 확인 후 `/dev/video0`, `/dev/video2` 할당을 반대로 바꿉니다.

### 데이터 수집 중 이미지가 끊기는 경우

카메라 토픽 Hz를 확인합니다.

```bash
ros2 topic hz /left/camera/image_raw
ros2 topic hz /right/camera/image_raw
```

30Hz 미만이면 `data_collector.py` 상단의 `COLLECT_HZ` 값을 낮춥니다.

```python
COLLECT_HZ = 20   # 30 → 20으로 낮춤
```

### RAM 부족 경고

에피소드 1개당 메모리 사용량 추정:

```
640 × 480 × 3 bytes × 2 cameras × 30fps × 30초
= 약 1.6 GB / 에피소드
```

50 에피소드 × 1.6 GB = 약 80 GB → RAM 초과 가능

`data_collector.py`는 **에피소드 종료 시 즉시 pkl로 저장**하므로 실제로는 에피소드 1개 분량만 RAM에 유지됩니다. 문제가 발생하면 `stop_episode` 직후 `gc.collect()`를 호출하세요.

---

## 8. 파일 목록

| 파일 | 역할 |
|------|------|
| `data_collector.py` | ROS2 토픽 구독 및 에피소드 수집 |
| `convert_to_lerobot.py` | pkl → parquet + mp4 + meta JSON 변환 |
