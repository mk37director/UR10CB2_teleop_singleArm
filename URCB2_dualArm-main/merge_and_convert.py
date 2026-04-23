"""
merge_and_convert.py  (v3.0 완전 재작성)
=========================================
여러 세션 폴더의 pkl 데이터를 LeRobot v3.0 포맷으로 직접 변환합니다.

LeRobot v3.0 파일 구조 (utils.py 상수 기준):
  meta/
    info.json
    tasks.parquet
    stats.json
    episodes/chunk-000/file-000.parquet
  data/
    chunk-000/file-000.parquet
  videos/
    observation.images.cam_left/chunk-000/file-000.mp4
    observation.images.cam_right/chunk-000/file-000.mp4

사용법:
  python3 merge_and_convert.py \
      /mnt/lerobot_data/lerobot_data/session_A \
      /mnt/lerobot_data/lerobot_data/session_B \
      --output /mnt/lerobot_data/lerobot_data/lerobot_v30
"""

import argparse
import gc
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CODEBASE_VERSION = 'v3.0'
CHUNKS_SIZE      = 1000
VIDEO_CODEC      = 'mp4v'
TASK_DESCRIPTION = "Pick up the object and place it inside the blue square on the table."

CAM_KEYS = [
    'observation.images.cam_left',
    'observation.images.cam_right',
]

TASKS_PATH        = 'meta/tasks.parquet'
EPISODES_DIR      = 'meta/episodes'
INFO_PATH         = 'meta/info.json'
STATS_PATH        = 'meta/stats.json'
CHUNK_PAT         = 'chunk-{ci:03d}/file-{fi:03d}'
DATA_PAT          = 'data/' + CHUNK_PAT + '.parquet'
VIDEO_PAT         = 'videos/{vk}/' + CHUNK_PAT + '.mp4'
EPISODES_PAT      = EPISODES_DIR + '/' + CHUNK_PAT + '.parquet'


def load_frames(pkl_path: Path) -> list:
    frames = []
    with open(pkl_path, 'rb') as f:
        try:
            first = pickle.load(f)
        except EOFError:
            return []
        if isinstance(first, dict) and 'frames' in first:
            return first['frames']
        if isinstance(first, dict) and first.get('format') == 'chunked_v1':
            for _ in range(first.get('n_chunks', 9999)):
                try:
                    chunk = pickle.load(f)
                    if isinstance(chunk, list):
                        frames.extend(chunk)
                        del chunk
                except EOFError:
                    break
            return frames
        if isinstance(first, list):
            frames.extend(first)
            del first
            while True:
                try:
                    chunk = pickle.load(f)
                    if isinstance(chunk, list):
                        frames.extend(chunk)
                        del chunk
                except EOFError:
                    break
            return frames
    return frames


def collect_episodes(session_dirs):
    all_eps = []
    gidx = 0
    for sp in session_dirs:
        summary_path = sp / 'summary.json'
        if not summary_path.exists():
            print(f'  [WARN] summary.json 없음: {sp}')
            continue
        s = json.load(open(summary_path))
        print(f'  [{sp.name}] {s["n_episodes"]} ep, {s["total_frames"]} frames')
        for ep in s['episodes']:
            pkl_path = sp / ep['pkl_file']
            if not pkl_path.exists():
                print(f'  [WARN] 없음: {pkl_path.name}')
                continue
            all_eps.append({
                'gidx':     gidx,
                'pkl_path': pkl_path,
                'n_frames': ep['n_frames'],
                'source':   sp.name,
            })
            gidx += 1
    return all_eps


def save_video(frames, cam_key, path, fps, w, h):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for fr in frames:
        wr.write(cv2.cvtColor(fr[cam_key], cv2.COLOR_RGB2BGR))
    wr.release()
    return path.stat().st_size / 1e6


def merge_and_convert(session_dirs, output_dir):
    sessions = [Path(d).expanduser().resolve() for d in session_dirs]
    out      = Path(output_dir).expanduser().resolve()

    if out.exists():
        print(f'[WARN] 출력 폴더 존재: {out}')
        if input('  덮어쓰기? (y/N): ').strip().lower() != 'y':
            sys.exit(0)
        shutil.rmtree(out)

    s0  = json.load(open(sessions[0] / 'summary.json'))
    fps = s0['fps']
    h, w = s0['image_size'][0], s0['image_size'][1]
    print(f'\n[설정] fps={fps}, {w}x{h}')

    print('\n[에피소드 수집]')
    eps     = collect_episodes(sessions)
    n_total = len(eps)
    print(f'  총 {n_total} 에피소드\n')
    if n_total == 0:
        sys.exit(1)

    n_chunks = (n_total + CHUNKS_SIZE - 1) // CHUNKS_SIZE
    (out / 'meta' / 'episodes').mkdir(parents=True)

    data_rows = []
    ep_rows   = []
    all_obs   = []
    all_act   = []
    total_fr  = 0

    for ep in eps:
        gidx    = ep['gidx']
        ci      = gidx // CHUNKS_SIZE
        fi      = gidx % CHUNKS_SIZE

        print(f'[EP {gidx:03d}] {ep["source"]}/{ep["pkl_path"].name}')
        frames = load_frames(ep['pkl_path'])
        n = len(frames)
        if n == 0:
            print('  [WARN] 프레임 없음')
            continue
        print(f'  {n} frames ({n/fps:.1f}s)')

        # 비디오 저장
        for cam_key in CAM_KEYS:
            vp = out / VIDEO_PAT.format(vk=cam_key, ci=ci, fi=fi)
            mb = save_video(frames, cam_key, vp, fps, w, h)
            print(f'    {cam_key}/chunk-{ci:03d}/file-{fi:03d}.mp4 | {mb:.1f}MB')

        # data rows
        obs_arr = np.stack([fr['observation.state'].astype(np.float32) for fr in frames])
        act_arr = np.stack([fr['action'].astype(np.float32) for fr in frames])

        vpath_l = VIDEO_PAT.format(vk='observation.images.cam_left',  ci=ci, fi=fi)
        vpath_r = VIDEO_PAT.format(vk='observation.images.cam_right', ci=ci, fi=fi)

        for fi2 in range(n):
            data_rows.append({
                'observation.state':            obs_arr[fi2],
                'action':                       act_arr[fi2],
                'observation.images.cam_left':  vpath_l,
                'observation.images.cam_right': vpath_r,
                'timestamp':                    float(fi2) / fps,
                'frame_index':                  fi2,
                'episode_index':                gidx,
                'task_index':                   0,
                'index':                        total_fr + fi2,
            })

        ep_rows.append({'episode_index': gidx, 'task_index': 0,
                        'length': n, 'chunk_index': ci, 'file_index': fi})
        all_obs.extend(list(obs_arr))
        all_act.extend(list(act_arr))
        total_fr += n
        del frames, obs_arr, act_arr
        gc.collect()

    # data parquet
    print('\n[data parquet 저장]')
    df = pd.DataFrame(data_rows)
    for col in ['frame_index','episode_index','task_index','index']:
        df[col] = df[col].astype('int64')
    df['timestamp'] = df['timestamp'].astype('float32')
    del data_rows

    for ci in range(n_chunks):
        mask = (df['episode_index'] >= ci*CHUNKS_SIZE) & \
               (df['episode_index'] < (ci+1)*CHUNKS_SIZE)
        dp = out / DATA_PAT.format(ci=ci, fi=0)
        dp.parent.mkdir(parents=True, exist_ok=True)
        chunk_df = df[mask]
        chunk_df.to_parquet(dp, index=False)
        print(f'  data/chunk-{ci:03d}/file-000.parquet | {len(chunk_df)} rows | '
              f'{dp.stat().st_size/1e6:.1f}MB')
    del df
    gc.collect()

    # episodes parquet
    print('\n[meta/episodes parquet 저장]')
    df_ep = pd.DataFrame(ep_rows)
    for col in ['episode_index','task_index','length','chunk_index','file_index']:
        df_ep[col] = df_ep[col].astype('int64')
    for ci in range(n_chunks):
        mask  = (df_ep['episode_index'] >= ci*CHUNKS_SIZE) & \
                (df_ep['episode_index'] < (ci+1)*CHUNKS_SIZE)
        ep_p  = out / EPISODES_PAT.format(ci=ci, fi=0)
        ep_p.parent.mkdir(parents=True, exist_ok=True)
        df_ep[mask].to_parquet(ep_p, index=False)
        print(f'  episodes/chunk-{ci:03d}/file-000.parquet | {mask.sum()} rows')

    # tasks.parquet
    pd.DataFrame([{'task_index': 0, 'task': TASK_DESCRIPTION}])\
      .to_parquet(out / TASKS_PATH, index=False)
    print('\n  tasks.parquet 저장 완료')

    # 통계
    print('\n[통계 계산]')
    def _stat(arr):
        return {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(),
                'min': arr.min(0).tolist(),  'max': arr.max(0).tolist()}
    oa = np.stack(all_obs); aa = np.stack(all_act)
    stats = {'observation.state': _stat(oa), 'action': _stat(aa)}
    del all_obs, all_act, oa, aa; gc.collect()
    with open(out / STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print('  stats.json 저장 완료')

    # info.json
    actual_n = len(ep_rows)
    info = {
        'codebase_version': CODEBASE_VERSION,
        'robot_type':       'ur10',
        'fps':              fps,
        'total_episodes':   actual_n,
        'total_frames':     total_fr,
        'total_tasks':      1,
        'total_videos':     actual_n * len(CAM_KEYS),
        'total_chunks':     n_chunks,
        'chunks_size':      CHUNKS_SIZE,
        'splits':           {'train': f'0:{actual_n}'},
        'data_path':        DATA_PAT.replace('{ci:03d}','{episode_chunk:03d}')
                                   .replace('{fi:03d}', '{episode_index:06d}'),
        'video_path':       VIDEO_PAT.replace('{vk}',   '{video_key}')
                                     .replace('{ci:03d}','{episode_chunk:03d}')
                                     .replace('{fi:03d}', '{episode_index:06d}'),
        'features': {
            'observation.state':  {'dtype':'float32','shape':[6],
                'names':['j1','j2','j3','j4','j5','j6']},
            'action':             {'dtype':'float32','shape':[6],
                'names':['j1','j2','j3','j4','j5','j6']},
            'observation.images.cam_left':  {
                'dtype':'video','shape':[h,w,3],'names':None,
                'info':{'video.fps':fps,'video.codec':'mp4v',
                        'video.pix_fmt':'yuv420p','video.is_depth_map':False}},
            'observation.images.cam_right': {
                'dtype':'video','shape':[h,w,3],'names':None,
                'info':{'video.fps':fps,'video.codec':'mp4v',
                        'video.pix_fmt':'yuv420p','video.is_depth_map':False}},
            'timestamp':     {'dtype':'float32','shape':[1],'names':None},
            'frame_index':   {'dtype':'int64',  'shape':[1],'names':None},
            'episode_index': {'dtype':'int64',  'shape':[1],'names':None},
            'task_index':    {'dtype':'int64',  'shape':[1],'names':None},
            'index':         {'dtype':'int64',  'shape':[1],'names':None},
        },
    }
    with open(out / INFO_PATH, 'w') as f:
        json.dump(info, f, indent=2)

    print(f'\n{"="*60}')
    print(f'[DONE] LeRobot v3.0 변환 완료!')
    print(f'  경로: {out}')
    print(f'  에피소드: {actual_n}  |  프레임: {total_fr} ({total_fr/fps:.1f}s)')
    print(f'{"="*60}')
    print(f'\n학습:')
    print(f'  lerobot-train \\')
    print(f'      --dataset.repo_id=local/ur10_cup_task \\')
    print(f'      --dataset.root={out} \\')
    print(f'      --policy.type=act \\')
    print(f'      --policy.push_to_hub=false \\')
    print(f'      --output_dir=/home/ur12e/outputs/train/act_ur10 \\')
    print(f'      --job_name=act_ur10_cup_task \\')
    print(f'      --policy.device=cuda \\')
    print(f'      --wandb.enable=false \\')
    print(f'      --steps=100000 \\')
    print(f'      --batch_size=8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sessions', nargs='+')
    parser.add_argument('--output', '-o', required=True)
    args = parser.parse_args()
    merge_and_convert(args.sessions, args.output)
