# UR10 CB2 Dual Teleop — 작업 정리

## 시스템 개요

```
UR10 CB2 Left  (Leader,   192.168.1.121) — Freedrive 로 손으로 조작
UR10 CB2 Right (Follower, 192.168.1.120) — Left 를 실시간 추종
```

---

## 수정된 파일 목록 및 변경 내용

### 1. yur_ros2_driver 패키지

#### `yur_ros_wrapper.hpp / yur_ros_wrapper.cpp`
- `read_only` 파라미터 추가
- `read_only = true`  : Leader 모드 — uploadProg 없음, joint_states 만 publish
                        티치펜던트/Freedrive 자유롭게 사용 가능, Broken pipe 없음
- `read_only = false` : Follower 모드 — 기존 동작 (uploadProg + servoj)

#### `multiArm_bringup.cpp`
- Left  → `read_only = true`  (Leader)
- Right → `read_only = false` (Follower)

---

### 2. Y2RobMotion 패키지

#### `ur10_motion.hpp`
- `cmdJoint` subscriber 추가 (Float64MultiArray)
- `cmdJointCB()` 콜백 선언
- `control_teleop()` 함수 선언
- `teleop_target_angles` 버퍼, `teleop_cmd_received` 플래그 추가

#### `ur10_motion.cpp`
- `cmdJoint` subscriber 등록 (`robot_name + "/cmdJoint"`)
- `cmdJointCB()` 구현 — Teleop 모드일 때만 joint 버퍼 업데이트
- `control_teleop()` 구현 — IK 없이 joint angle 직접 추종 + joint limit 클리핑
- `main_control()` 에 "Teleop" 분기 추가

#### `singleArm_motion.cpp`
- `ROBOT_MODEL` 하드코딩 제거
- `argv[1]` 로 실행 인자 수신
- node 이름 중복 방지: `"urSingleArm_" + ROBOT_MODEL`

#### `singleArm_cmd.cpp`
- `ROBOT_MODEL` 하드코딩 제거
- `argv[1]` 로 실행 인자 수신
- node 이름 중복 방지: `"singleArm_cmd_" + ROBOT_MODEL`

---

### 3. Python

#### `ur10_dual_teleop.py`
- ROS2 타이머 기반 제어 루프 (지연 최소화)
- SingleThreadedExecutor 사용 (GIL 경합 제거)
- Left currentJ 구독 (yur_ros2_driver 불필요)
- 소프트웨어 Freedrive ON/OFF (30002 포트 URScript 전송)
- 3가지 추종 모드 (안전/일반/직접)
- Teleop 모드 매초 재전송 (타이밍 문제 방지)

---

## 파일 복사 위치

```
Y2RobMotion/
  src/ur10_motion.cpp       → .../Y2RobMotion/src/
  include/ur10_motion.hpp   → .../Y2RobMotion/include/Y2RobMotion/
  src/singleArm_motion.cpp  → .../Y2RobMotion/src/
  src/singleArm_cmd.cpp     → .../Y2RobMotion/src/

yur_ros2_driver/
  include/yur_ros_wrapper.hpp → .../yur_ros2_driver/include/yur_ros2_driver/
  src/yur_ros_wrapper.cpp     → .../yur_ros2_driver/src/
  src/multiArm_bringup.cpp    → .../yur_ros2_driver/src/

python/
  ur10_dual_teleop.py → 원하는 위치
```

---

## 빌드

```bash
cd /home/ur10cb2/urcb2_ws

# yur_ros2_driver 빌드
colcon build --packages-select yur_ros2_driver

# Y2RobMotion 빌드
colcon build --packages-select Y2RobMotion
```

---

## 실행 순서

```bash
# 터미널 1: 두 로봇 드라이버
ros2 run yur_ros2_driver yur_multi

# 터미널 2: Left 모션 노드 (Leader)
ros2 run Y2RobMotion singleArm_motion UR10_left

# 터미널 3: Right 모션 노드 (Follower)
ros2 run Y2RobMotion singleArm_motion UR10_right

# 터미널 4: Python Teleop
python3 ur10_dual_teleop2.py
```

---

## 조작 방법

| 키 | 동작 |
|----|------|
| `f + Enter` | Left Freedrive ON — 손으로 잡고 자유롭게 이동 |
| `e + Enter` | Left Freedrive OFF |
| `g + Enter` | Left Hand-guiding 모드 (F/T 센서 필요) |
| `i + Enter` | Left Idling 모드 |
| `1 + Enter` | 안전 모드 (0.05 rad/step 제한) |
| `2 + Enter` | 일반 모드 (0.15 rad/step 제한) |
| `3 + Enter` | 직접 모드 (제한 없음, 즉시 추종) ★ |
| `q + Enter` | 종료 |

---

## 토픽 흐름

```
yur_multi
  ├── UR10_left  (read_only=true)  → /UR10_left/joint_states  publish
  └── UR10_right (read_only=false) → /UR10_right/targetJ subscribe → servoj()

singleArm_motion UR10_left
  ├── /UR10_left/joint_states  subscribe
  └── /UR10_left/currentJ      publish (125Hz)

singleArm_motion UR10_right
  ├── /UR10_right/cmdMode   subscribe ← Python
  ├── /UR10_right/cmdJoint  subscribe ← Python
  └── /UR10_right/targetJ   publish → yur_multi → servoj()

ur10_dual_teleop.py
  ├── /UR10_left/currentJ   subscribe (Leader 위치 읽기)
  ├── /UR10_right/currentJ  subscribe (Follower 위치 읽기)
  ├── /UR10_right/cmdMode   publish ("Teleop", 매초 재전송)
  └── /UR10_right/cmdJoint  publish (target joints, 125Hz)
```

---

## 로봇 IP 및 포트

| 로봇 | IP | Reverse Port | 역할 |
|------|-----|-------------|------|
| UR10_right | 192.168.1.120 | 50001 | Follower |
| UR10_left  | 192.168.1.121 | 50002 | Leader |

---

## 다음 단계 (Phase 계획)

```
현재 (Phase 2):
  SO-ARM101 대기 중 → UR10 CB2 × 2 로 Teleop 구현

SO-ARM101 도착 후 (Phase 1 → 2):
  Phase 1: SO-ARM101 × 2 로 Lerobot 파이프라인 완성
  Phase 2: SO-ARM101 (Leader) + UR10 CB2 (Follower)

장기 (Phase 3~4):
  Phase 3: SO-ARM101 × 2 (Leader) + UR10e × 2 (Follower) Bimanual
  Phase 4: ACT / Diffusion Policy / smolVLA Autonomous
```
