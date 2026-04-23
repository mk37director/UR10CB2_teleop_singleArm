# UR10 Dual Arm - Imitation Learning 전체 파이프라인

## 시스템 구성

| 항목 | 내용 |
|------|------|
| Leader (Left)  | UR10 CB2, IP: `192.168.1.121` |
| Follower (Right) | UR10 CB2, IP: `192.168.1.120` |
| PC | Ubuntu, ROS2 Humble, RTX 5060 Ti 16GB |
| 수집 Hz | 30 Hz (카메라 기준) |
| 저장 형식 | pkl (streaming_v1) → LeRobot v3.0 |

---

## 파이프라인 개요

```
[1] 데이터 수집          data_collector2A.py
        ↓
[2] summary.json 생성    Makejson.py  (Q 키 없이 종료된 경우)
        ↓
[3] 병합 + 포맷 변환     merge_and_convert.py  (복수 세션 → LeRobot v3.0)
        ↓
[4] parquet 후처리       카메라 컬럼 제거 + episodes 컬럼 추가
        ↓
[5] 학습                 lerobot-train
```

---

## Step 1: 데이터 수집 — `data_collector2A.py`

### 사전 조건

```bash
# 터미널 1: ROS2 드라이버
ros2 run yur_ros2_driver yur_multi

# 터미널 2: Left arm 모션 노드
ros2 run Y2RobMotion singleArm_motion UR10_left  → '1' 입력

# 터미널 3: Right arm 모션 노드
ros2 run Y2RobMotion singleArm_motion UR10_right → '1' 입력

# 터미널 4: 카메라 노드 (좌)
ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video2" \
    -r __ns:="/left" -r __node:="camera"

# 터미널 5: 카메라 노드 (우)
ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video0" \
    -r __ns:="/right" -r __node:="camera"
```

### 수집 실행

```bash
python3 data_collector2A.py
```

### 조작 명령

| 키 | 동작 |
|----|------|
| `S` | 에피소드 시작 |
| `E` | 에피소드 저장 (성공) → 즉시 pkl 저장 후 RAM 해제 |
| `D` | 에피소드 버림 (실패) → RAM 즉시 해제 |
| `I` | 현재 상태 확인 (RAM 사용량 포함) |
| `Q` | 전체 종료 및 summary.json 저장 |

### 저장 경로

```
~/outputs/lerobot_data/session_YYYYMMDD_HHMMSS/
  episode_000000.pkl
  episode_000000_meta.json   ← 프레임 수 등 메타데이터
  episode_000001.pkl
  episode_000001_meta.json
  ...
  summary.json               ← Q 키 시 생성
```

### 주의사항

- **반드시 `Q` 키로 종료**해야 `summary.json`이 생성됩니다
- Ctrl+C로 강제 종료 시 `summary.json`이 없을 수 있음 → Step 2 필요
- RAM 부족 방지를 위해 에피소드는 즉시 디스크에 저장됨
- 에피소드당 약 50~100프레임당 자동 flush

---

## Step 2: summary.json 재생성 — `Makejson.py`

`Q` 키 없이 종료되어 `summary.json`이 없는 폴더에 사용합니다.

### 확인

```bash
# summary.json 누락 폴더 확인
for d in ~/outputs/lerobot_data/session_*/; do
    if [ -f "$d/summary.json" ]; then
        echo "✅ $d"
    else
        echo "❌ $d  ← summary.json 없음"
    fi
done
```

### 실행

```python
# Makejson.py 내 SESSION_PATH 수정
SESSION_PATH = '/home/ur12e/outputs/lerobot_data/session_20260413_160942'
```

```bash
python3 Makejson.py
```

폴더가 여러 개라면 `SESSION_PATH`만 바꿔서 반복 실행합니다.

---

## Step 3: 병합 + LeRobot v3.0 변환 — `merge_and_convert.py`

여러 세션 폴더를 하나의 LeRobot v3.0 데이터셋으로 변환합니다.

### 실행

```bash
python3 merge_and_convert.py \
    ~/outputs/lerobot_data/session_20260413_154550 \
    ~/outputs/lerobot_data/session_20260413_160942 \
    ~/outputs/lerobot_data/session_20260413_163152 \
    --output /mnt/lerobot_data/lerobot_data/lerobot_v30_new
```

### 출력 구조 (LeRobot v3.0)

```
lerobot_v30_new/
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/
      chunk-000/
        file-000.parquet
  data/
    chunk-000/
      file-000.parquet
  videos/
    observation.images.cam_left/
      chunk-000/
        file-000.mp4
        file-001.mp4
        ...
    observation.images.cam_right/
      chunk-000/
        file-000.mp4
        ...
```

### 손상된 pkl 파일 확인

변환 중 `[WARN] 프레임 없음` 이 출력되는 에피소드는 pkl이 손상된 것입니다.

```bash
python3 << 'EOF'
import pickle
from pathlib import Path

session = Path('/경로/session_손상된폴더')
for pkl_path in sorted(session.glob('episode_*.pkl')):
    try:
        with open(pkl_path, 'rb') as f:
            first = pickle.load(f)
            second = pickle.load(f)
            n = len(second) if isinstance(second, list) else 0
            print(f'  ✅ {pkl_path.name}: {n} frames')
    except Exception as e:
        print(f'  ❌ {pkl_path.name}: 손상 ({e})')
EOF
```

손상된 폴더는 변환에서 제외되며 해당 에피소드는 복구 불가합니다.

---

## Step 4: parquet 후처리

변환 후 반드시 아래 두 가지 후처리가 필요합니다.

### 4-1. data parquet 카메라 컬럼 제거

LeRobot v3.0은 data parquet에 카메라 컬럼이 없어야 합니다 (비디오는 info.json 경로 패턴으로만 참조).

```bash
python3 << 'EOF'
import pandas as pd
from pathlib import Path

data_dir = Path('/mnt/lerobot_data/lerobot_data/lerobot_v30_new/data')
for p in sorted(data_dir.glob('**/*.parquet')):
    df = pd.read_parquet(p)
    drop = [c for c in df.columns if 'images' in c or 'cam_' in c]
    if drop:
        df.drop(columns=drop).to_parquet(p, index=False)
        print(f'  {p.name}: {drop} 제거')
    else:
        print(f'  {p.name}: 이미 정상')
EOF
```

### 4-2. episodes parquet 필수 컬럼 추가

LeRobot이 episodes parquet에서 요구하는 모든 컬럼을 추가합니다.

```bash
python3 << 'EOF'
import pandas as pd
from pathlib import Path

root   = Path('/mnt/lerobot_data/lerobot_data/lerobot_v30_new')
ep_p   = root / 'meta/episodes/chunk-000/file-000.parquet'
data_p = root / 'data/chunk-000/file-000.parquet'

df_ep   = pd.read_parquet(ep_p)
df_data = pd.read_parquet(data_p)

# 1) dataset_from_index / dataset_to_index
ep_bounds = df_data.groupby('episode_index')['index'].agg(['min','max']).reset_index()
ep_bounds.columns = ['episode_index','dataset_from_index','dataset_to_index']
ep_bounds['dataset_to_index'] += 1   # exclusive end
df_ep = df_ep.merge(ep_bounds, on='episode_index', how='left')
df_ep['dataset_from_index'] = df_ep['dataset_from_index'].astype('int64')
df_ep['dataset_to_index']   = df_ep['dataset_to_index'].astype('int64')

# 2) timestamp 범위
ep_ts = df_data.groupby('episode_index')['timestamp'].agg(['min','max']).reset_index()
ep_ts.columns = ['episode_index','from_ts','to_ts']
df_ep = df_ep.merge(ep_ts, on='episode_index', how='left')

# 3) 카메라별 컬럼
for cam_key in ['observation.images.cam_left', 'observation.images.cam_right']:
    df_ep[f'videos/{cam_key}/chunk_index']      = df_ep['chunk_index'].astype('int64')
    df_ep[f'videos/{cam_key}/file_index']       = df_ep['file_index'].astype('int64')
    df_ep[f'videos/{cam_key}/from_timestamp']   = df_ep['from_ts'].astype('float32')
    df_ep[f'videos/{cam_key}/to_timestamp']     = df_ep['to_ts'].astype('float32')

# 4) data / meta/episodes 컬럼
df_ep['data/chunk_index']          = df_ep['chunk_index'].astype('int64')
df_ep['data/file_index']           = df_ep['file_index'].astype('int64')
df_ep['meta/episodes/chunk_index'] = df_ep['chunk_index'].astype('int64')
df_ep['meta/episodes/file_index']  = df_ep['file_index'].astype('int64')

df_ep = df_ep.drop(columns=['from_ts','to_ts'])
df_ep.to_parquet(ep_p, index=False)

print('episodes parquet 컬럼:')
for c in df_ep.columns:
    print(f'  {c}')
print(f'\n총 {len(df_ep)} 에피소드 완료')
EOF
```

### 4-3. stats.json 카메라 통계 추가

```bash
python3 << 'EOF'
import json
from pathlib import Path

p = Path('/mnt/lerobot_data/lerobot_data/lerobot_v30_new/meta/stats.json')
stats = json.load(open(p))
for cam in ['observation.images.cam_left', 'observation.images.cam_right']:
    if cam not in stats:
        stats[cam] = {
            'mean': [[[0.485, 0.456, 0.406]]],   # ImageNet mean
            'std':  [[[0.229, 0.224, 0.225]]],   # ImageNet std
            'min':  [[[0.0, 0.0, 0.0]]],
            'max':  [[[1.0, 1.0, 1.0]]],
        }
json.dump(stats, open(p,'w'), indent=2)
print('stats.json 키:', list(stats.keys()))
EOF
```

### 4-4. info.json 경로 패턴 및 features 확인

```bash
python3 << 'EOF'
import json
from pathlib import Path

info_p = Path('/mnt/lerobot_data/lerobot_data/lerobot_v30_new/meta/info.json')
with open(info_p) as f:
    info = json.load(f)

# video_path 패턴 수정 (LeRobot이 기대하는 형식)
info['video_path'] = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
info['data_path']  = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'

# 카메라 features names 수정 (None이면 오류 발생)
for cam_key in ['observation.images.cam_left', 'observation.images.cam_right']:
    info['features'][cam_key]['names'] = ['height', 'width', 'channel']

with open(info_p, 'w') as f:
    json.dump(info, f, indent=2)

print('video_path:', info['video_path'])
print('data_path: ', info['data_path'])
print('cam names: ', info['features']['observation.images.cam_left']['names'])
EOF
```

### 4-5. episode_index 연속성 확인 및 재번호 부여

episode_index가 `0, 1, 2, ...` 연속이어야 합니다. 손상된 에피소드가 제외되면 중간이 비어 오류가 발생합니다.

```bash
python3 << 'EOF'
import pandas as pd
from pathlib import Path

root   = Path('/mnt/lerobot_data/lerobot_data/lerobot_v30_new')
ep_p   = root / 'meta/episodes/chunk-000/file-000.parquet'
data_p = root / 'data/chunk-000/file-000.parquet'

df_ep   = pd.read_parquet(ep_p)
df_data = pd.read_parquet(data_p)

old_indices = sorted(df_ep['episode_index'].unique().tolist())
print('현재 episode_index:', old_indices)

# 연속이 아니면 재번호 부여
if old_indices != list(range(len(old_indices))):
    mapping = {old: new for new, old in enumerate(old_indices)}
    print('재번호 매핑:', mapping)

    df_data['episode_index'] = df_data['episode_index'].map(mapping)
    df_data['index'] = range(len(df_data))
    df_data.to_parquet(data_p, index=False)

    df_ep['episode_index'] = df_ep['episode_index'].map(mapping)
    df_ep = df_ep.sort_values('episode_index').reset_index(drop=True)
    df_ep.to_parquet(ep_p, index=False)

    import json
    info_p = root / 'meta/info.json'
    info = json.load(open(info_p))
    n = len(old_indices)
    info['total_episodes'] = n
    info['splits'] = {'train': f'0:{n}'}
    json.dump(info, open(info_p,'w'), indent=2)

    print('재번호 완료:', list(range(n)))
else:
    print('episode_index 정상 (연속)')
EOF
```

---

## Step 5: ACT 학습

### 가상환경 활성화

```bash
source ~/lerobot_env/bin/activate
```

### 학습 실행

```bash
lerobot-train \
    --dataset.repo_id=local/ur10_cup_task \
    --dataset.root=/mnt/lerobot_data/lerobot_data/lerobot_v30_new \
    --dataset.revision=main \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=/home/ur12e/outputs/train/act_ur10 \
    --job_name=act_ur10_cup_task \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --batch_size=8
```

### 주요 옵션 설명

| 옵션 | 설명 |
|------|------|
| `--dataset.revision=main` | HuggingFace 버전 확인 우회 (로컬 전용) |
| `--policy.push_to_hub=false` | HuggingFace 업로드 안 함 |
| `--steps=100000` | 총 학습 스텝 (약 2~3시간, RTX 5060 Ti) |
| `--batch_size=8` | 배치 크기 (16GB VRAM 기준 8~16 가능) |
| `--save_freq=20000` | 20000 스텝마다 체크포인트 저장 |
| `--log_freq=200` | 200 스텝마다 loss 출력 |

### 정상 학습 출력 예시

```
INFO dataset.num_frames=30955 (31K)
INFO dataset.num_episodes=16
INFO num_learnable_params=51597190 (52M)
INFO Start offline training on a fixed dataset
Training:   2%| 2000/100000 [05:23<4:24:12,  6.18step/s]
  loss: 1.234  kl_loss: 0.012  l1_loss: 1.222
```

### GPU 사용량 모니터링 (별도 터미널)

```bash
watch -n 2 nvidia-smi
```

정상 학습 중: `GPU-Util 80~99%`, `Memory 8000~12000MiB`

### 체크포인트 확인

```bash
ls ~/outputs/train/act_ur10/checkpoints/
# 020000/
# 040000/
# last/   ← 최신 체크포인트
```

### 학습 재개 (중단된 경우)

```bash
lerobot-train \
    --dataset.repo_id=local/ur10_cup_task \
    --dataset.root=/mnt/lerobot_data/lerobot_data/lerobot_v30_new \
    --dataset.revision=main \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=/home/ur12e/outputs/train/act_ur10 \
    --job_name=act_ur10_cup_task \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --batch_size=8 \
    --resume=true \
    --checkpoint_path=/home/ur12e/outputs/train/act_ur10/checkpoints/last
```

---

## 트러블슈팅

### HuggingFace 401 오류

```
RepositoryNotFoundError: 401 Client Error
```

→ `--dataset.revision=main` 옵션 추가 (로컬 데이터셋은 HF 인증 불필요)

### CastError: column names don't match

```
observation.images.cam_left: large_string
```

→ Step 4-1 실행 (data parquet 카메라 컬럼 제거)

### KeyError: 'dataset_from_index'

→ Step 4-2 실행 (episodes parquet 필수 컬럼 추가)

### KeyError: 'videos/.../from_timestamp'

→ Step 4-2 실행 (timestamp 컬럼 포함)

### KeyError: 'episode_chunk'

→ Step 4-4 실행 (info.json video_path 패턴 수정)

### TypeError: 'NoneType' object is not subscriptable

```
names[2] in ["channel", "channels"]
```

→ Step 4-4 실행 (카메라 features names 수정)

### IndexError: Invalid key X is out of bounds for size Y

→ Step 4-5 실행 (episode_index 연속성 확인 및 재번호)

### BackwardCompatibilityError: dataset is in 2.1 format

→ `info.json`의 `codebase_version`을 `"v3.0"`으로 수정

```bash
python3 -c "
import json; from pathlib import Path
p = Path('/mnt/.../lerobot_v30_new/meta/info.json')
info = json.load(open(p))
info['codebase_version'] = 'v3.0'
json.dump(info, open(p,'w'), indent=2)
print('완료')
"
```

### pkl 파일 손상 (Memo value not found)

```
❌ episode_000000.pkl: 손상: Memo value not found at index 23
```

→ 저장 중 비정상 종료로 인한 손상. 복구 불가. 해당 에피소드 제외 후 진행.

---

## 전체 명령 요약

```bash
# 1. 수집
python3 data_collector2A.py

# 2. summary.json 누락 시
python3 Makejson.py   # SESSION_PATH 수정 후

# 3. 변환
python3 merge_and_convert.py \
    ~/outputs/lerobot_data/session_A \
    ~/outputs/lerobot_data/session_B \
    ~/outputs/lerobot_data/session_C \
    --output /mnt/lerobot_data/lerobot_data/lerobot_v30_new

# 4. 후처리 (4개 스크립트 순서대로)
#    4-1. data parquet 카메라 컬럼 제거
#    4-2. episodes parquet 필수 컬럼 추가
#    4-3. stats.json 카메라 통계 추가
#    4-4. info.json 경로/features 수정
#    4-5. episode_index 연속성 확인

# 5. 학습
source ~/lerobot_env/bin/activate
lerobot-train \
    --dataset.repo_id=local/ur10_cup_task \
    --dataset.root=/mnt/lerobot_data/lerobot_data/lerobot_v30_new \
    --dataset.revision=main \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=/home/ur12e/outputs/train/act_ur10 \
    --job_name=act_ur10_cup_task \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000 \
    --batch_size=8
```
