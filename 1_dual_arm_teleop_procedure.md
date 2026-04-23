# UR10 CB2 Dual Arm Teleoperation 구동 절차

## 시스템 구성

| 항목 | 내용 |
|------|------|
| Leader (Left)   | UR10 CB2, IP: `192.168.1.121` |
| Follower (Right) | UR10 CB2, IP: `192.168.1.120` |
| PC              | IP: `192.168.1.130`, ROS2 Humble |
| 제어 주기       | 125 Hz |
| ROS2 드라이버   | `yur_ros2_driver` |
| 모션 노드       | `Y2RobMotion` (`singleArm_motion`) |

---

## 토픽 구조

```
yur_ros2_driver (yur_multi)
    │
    ├── UR10_left/joint_states   →  singleArm_motion (UR10_left)
    │                                   ├── UR10_left/currentJ  (publish)
    │                                   ├── UR10_left/currentP  (publish)
    │                                   ├── UR10_left/cmdMode   (subscribe)
    │                                   └── UR10_left/cmdJoint  (subscribe)
    │
    └── UR10_right/joint_states  →  singleArm_motion (UR10_right)
                                        ├── UR10_right/currentJ (publish)
                                        ├── UR10_right/currentP (publish)
                                        ├── UR10_right/cmdMode  (subscribe)
                                        └── UR10_right/cmdJoint (subscribe)

ur10_dual_teleop.py
    ├── subscribe: UR10_left/currentJ
    ├── subscribe: UR10_right/currentJ
    ├── publish:   UR10_left/cmdMode
    ├── publish:   UR10_right/cmdMode
    └── publish:   UR10_right/cmdJoint
```

---

## 사전 확인 사항

### 1. 네트워크 연결 확인

```bash
ping 192.168.1.121   # UR10_left (Leader)
ping 192.168.1.120   # UR10_right (Follower)
```

두 CB2 모두 응답이 와야 한다.

### 2. Polyscope Teach Pendant 상태 확인

**각 UR10의 Teach Pendant에서:**

- 로봇 전원 ON 및 브레이크 해제 확인
- Polyscope 메인 화면에서 `Program` 탭 진입
- `yur_ros2_driver` 용 External Control `.urp` 프로그램이 **Load** 되어 있는지 확인

> ⚠️ **중요**: `yur_multi` 실행 후 Polyscope에서 해당 프로그램을 **Play(▶)** 해야 CB2가 PC(`192.168.1.130:50001`)로 역접속을 시도한다. Play 상태가 아니면 드라이버가 `addCommandToQueue was executed` 이후 무한 대기 상태로 멈춘다.

### 3. 방화벽 확인

```bash
sudo ufw allow 50001/tcp
sudo ufw allow 50002/tcp
sudo ufw status
```

---

## 구동 절차

### Step 1. 이전 프로세스 정리

새로 시작하기 전에 잔여 프로세스와 ROS2 데몬을 초기화한다.

```bash
# 잔여 프로세스 확인
ps aux | grep -E 'teleop|yur_multi|singleArm'

# 잔여 프로세스 종료 (PID 확인 후)
kill -9 <PID>

# ROS2 데몬 재시작
ros2 daemon stop
ros2 daemon start

# 정리 확인
ros2 node list   # 아무것도 출력되지 않아야 정상
```

---

### Step 2. 터미널 1 — ROS2 드라이버 실행

```bash
taskset -c 3 ros2 run yur_ros2_driver yur_multi
```

**정상 출력 순서:**

```
DEBUG: Acquire firmware version: Connecting...
DEBUG: Acquire firmware version: Got connection
DEBUG: Firmware version detected: 1.8025319
DEBUG: Switching to secondary interface for masterboard data: Connecting...
DEBUG: Secondary interface: Got connection
DEBUG: Realtime port: Connecting...
DEBUG: Listening on 192.168.1.130:50001     ← 여기서 대기 시작
DEBUG: Realtime port: Got connection
addCommandToQueue was executed               ← CB2로 역접속 명령 전송
```

> ⚠️ `addCommandToQueue was executed` 이후 **즉시 Step 3을 진행**한다.  
> 이 시점에서 Polyscope Play가 완료되지 않으면 드라이버가 멈춘다.

---

### Step 3. Teach Pendant — 프로그램 Play

**UR10_left (Leader) Teach Pendant:**
1. Load된 External Control 프로그램 확인
2. **Play(▶) 버튼** 누르기
3. 드라이버 터미널에서 연결 완료 메시지 확인

**UR10_right (Follower) Teach Pendant:**
1. 동일하게 Play(▶) 버튼 누르기

**드라이버 터미널에서 연결 완료 확인:**

```
# 두 로봇 모두 연결되면 아래와 유사한 출력이 나타남
[INFO] Robot connected: UR10_left
[INFO] Robot connected: UR10_right
```

**토픽 발행 확인:**

```bash
ros2 topic list | grep joint_states
# UR10_left/joint_states
# UR10_right/joint_states
# 두 토픽이 모두 보여야 정상
```

---

### Step 4. 터미널 2 — UR10_left (Leader) 모션 노드 실행

```bash
taskset -c 4 ros2 run Y2RobMotion singleArm_motion UR10_left
```

**정상 출력:**

```
[INFO] ROBOT_MODEL: UR10_left
[INFO] Publisher was generated
[INFO] Subscription was generated
[INFO] Waiting for joint states...
[INFO] Joint states received!          ← joint_states 수신 확인
 [UR10_left] start? (1: start, 0: quit)
```

**`1` 입력 후 Enter** → 모션 제어 루프 시작

---

### Step 5. 터미널 3 — UR10_right (Follower) 모션 노드 실행

```bash
taskset -c 5 ros2 run Y2RobMotion singleArm_motion UR10_right
```

**정상 출력:**

```
[INFO] ROBOT_MODEL: UR10_right
[INFO] Joint states received!
 [UR10_right] start? (1: start, 0: quit)
```

**`1` 입력 후 Enter** → 모션 제어 루프 시작

---

### Step 6. 노드 상태 확인

```bash
ros2 node list
```

**정상 출력:**

```
/urSingleArm_UR10_left
/urSingleArm_UR10_right
```

**토픽 발행 확인:**

```bash
ros2 topic list | grep -E 'currentJ|cmdMode|cmdJoint'
# UR10_left/currentJ
# UR10_left/cmdMode
# UR10_left/cmdJoint
# UR10_right/currentJ
# UR10_right/cmdMode
# UR10_right/cmdJoint
```

---

### Step 7. 터미널 4 — Teleop 파이썬 노드 실행

```bash
taskset -c 6 python3 ur10_dual_teleop2.py
```

**정상 출력 (연결 대기):**

```
두 UR10 연결 대기 중...
연결 확인 완료.
  Leader  : ['+0.0°', '+0.0°', ...]
  Follower: ['+0.0°', '+0.0°', ...]
  최대 joint 차이: X.X°
```

**초기 위치 차이가 0.2 rad(≈11.5°) 이상인 경우:**

```
⚠️  초기 위치 차이 XX.X°
     계속: Enter  /  취소: q :
```

- `Enter` → Follower가 Leader 위치로 3초간 보간 이동 후 Teleop 시작
- `q` → 종료

**Teleop 시작 화면:**

```
=======================================================
  UR10 Dual Teleop 시작 (타이머 기반)
  125Hz | Leader: UR10_left | Follower: UR10_right
-------------------------------------------------------
  g : Hand-guiding        i : Idling
  1 : 안전 (0.05 rad/step)
  2 : 일반 (0.15 rad/step)
  3 : 직접 (즉시 추종) ★
  q : 종료
=======================================================
```

---

## 조작 명령

| 키 입력 | 동작 |
|---------|------|
| `g` + Enter | Leader → Hand-guiding 모드 (admittance 제어) |
| `i` + Enter | Leader → Idling 모드 |
| `1` + Enter | 안전 모드: Follower 이동량 ±0.05 rad/step 제한 |
| `2` + Enter | 일반 모드: Follower 이동량 ±0.15 rad/step 제한 |
| `3` + Enter | 직접 모드: 이동량 제한 없음 (즉시 추종) ★ |
| `q` + Enter | 종료 |

> **모드 선택 기준**  
> - 최초 구동 시: `1` (안전 모드) 로 시작 권장  
> - 동작 확인 후: `2` (일반 모드)  
> - 정밀 추종 필요 시: `3` (직접 모드, 주의 필요)

---

## 종료 절차

### 정상 종료

```
q + Enter   ← Teleop 터미널에서 입력
```

종료 시 자동으로:
1. 제어 타이머 중지
2. Follower → Idling 모드 전환
3. Leader → Idling 모드 전환
4. ROS2 shutdown

### 강제 종료 시 정리

```bash
# 각 터미널에서 Ctrl+C
# 이후 잔여 프로세스 확인 및 정리
ps aux | grep -E 'teleop|yur_multi|singleArm'
kill -9 <PID>

ros2 daemon stop
ros2 daemon start
```

---

## 문제 해결

### `addCommandToQueue was executed` 이후 멈춤

```
원인: Polyscope에서 프로그램이 Play 상태가 아님
해결: Teach Pendant에서 External Control 프로그램 Play(▶)
```

### `Joint states received!` 가 출력되지 않음

```
원인: yur_multi 가 완전히 연결되지 않았거나
      joint_states 토픽이 발행되지 않는 상태
해결:
  1. ros2 topic list | grep joint_states  로 토픽 확인
  2. yur_multi 터미널에서 연결 완료 메시지 재확인
  3. Polyscope Play 상태 재확인
```

### `두 UR10 연결 대기 중... 타임아웃.`

```
원인: currentJ 토픽이 20초 내 수신되지 않음
      → singleArm_motion 이 start 되지 않은 상태
해결: Step 4, 5에서 '1' 입력하여 start 완료 후 teleop 재실행
```

### `yur_multi <defunct>` 좀비 프로세스

```bash
# 부모 프로세스(ros2) PID 확인
ps aux | grep ros2

# 강제 종료
kill -9 <ros2 PID>

# ROS2 데몬 재시작
ros2 daemon stop
ros2 daemon start
```

---

## 전체 터미널 구성 요약

```
터미널 1: taskset -c 3 ros2 run yur_ros2_driver yur_multi
터미널 2: taskset -c 4 ros2 run Y2RobMotion singleArm_motion UR10_left   → '1' 입력
터미널 3: taskset -c 5 ros2 run Y2RobMotion singleArm_motion UR10_right  → '1' 입력
터미널 4: taskset -c 6 python3 ur10_dual_teleop2.py
```
Move to /ur10_ws/URCB2_dualArm-main before Terminal 4

> Step 순서 준수 필수: **터미널 1 실행 및 Polyscope Play 완료 후** 터미널 2, 3 실행
