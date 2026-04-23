"""
Makejson.py
===========
summary.json 이 없는 세션 폴더에 summary.json 을 재생성합니다.
pkl 파일들을 직접 읽어 에피소드 정보를 추출합니다.

사용법:
  1) 아래 SESSION_PATH 를 실제 폴더 경로로 수정
  2) python3 Makejson.py

summary.json 이 없는 폴더가 여러 개인 경우:
  SESSION_PATH 만 바꿔서 반복 실행
"""

import json
import pickle
from pathlib import Path

# ============================================================
# ★ 여기만 수정하세요 ★
SESSION_PATH = '/home/ur12e/outputs/lerobot_data/session_20260413/session_20260413_160942'
# ============================================================

FPS        = 30
IMAGE_SIZE = [480, 640, 3]
TASK       = "Pick up the object and place it inside the blue square on the table."


def count_frames(pkl_path: Path) -> int:
    """
    pkl 파일에서 프레임 수를 추출.
    모든 포맷 지원:
      - legacy      : dict with 'frames' key
      - chunked_v1  : dict with 'n_frames' key
      - streaming_v1: chunk list 반복
    """
    # _meta.json 이 있으면 거기서 바로 읽음 (가장 빠름)
    meta_path = pkl_path.parent / (pkl_path.stem + '_meta.json')
    if meta_path.exists():
        meta = json.load(open(meta_path))
        n = meta.get('n_frames', 0)
        print(f'    → _meta.json: {n} frames')
        return n

    # pkl 직접 파싱
    with open(pkl_path, 'rb') as f:
        try:
            first = pickle.load(f)
        except EOFError:
            return 0

        # legacy 포맷
        if isinstance(first, dict) and 'frames' in first:
            return len(first['frames'])

        # chunked_v1 포맷 (n_frames 키 포함)
        if isinstance(first, dict) and 'n_frames' in first:
            return first['n_frames']

        # streaming_v1 포맷 (chunk list 반복)
        if isinstance(first, list):
            n = len(first)
            del first
            while True:
                try:
                    chunk = pickle.load(f)
                    if isinstance(chunk, list):
                        n += len(chunk)
                        del chunk
                except EOFError:
                    break
            return n

    return 0


def make_summary(session_path: Path):
    pkl_files = sorted(session_path.glob('episode_*.pkl'))

    if not pkl_files:
        print(f'[ERROR] pkl 파일이 없습니다: {session_path}')
        return

    print(f'[폴더] {session_path}')
    print(f'[발견] pkl 파일 {len(pkl_files)}개\n')

    episodes     = []
    total_frames = 0

    for i, pkl_path in enumerate(pkl_files):
        print(f'  [{i:03d}] {pkl_path.name} ...', end='', flush=True)
        try:
            n = count_frames(pkl_path)
            print(f' {n} frames ({n / FPS:.1f}s)')

            episodes.append({
                'episode_index': i,
                'n_frames':      n,
                'duration_sec':  round(n / FPS, 2),
                'pkl_file':      pkl_path.name,
            })
            total_frames += n

        except Exception as e:
            print(f' [ERROR] {e}')

    summary = {
        'session_path':          str(session_path),
        'n_episodes':            len(episodes),
        'total_frames':          total_frames,
        'fps':                   FPS,
        'task':                  TASK,
        'image_size':            IMAGE_SIZE,
        'observation_state_dim': 6,
        'action_dim':            6,
        'topics': {
            'action':      'UR10_left/currentJ',
            'observation': 'UR10_right/currentJ',
            'cam_left':    '/left/image_raw',
            'cam_right':   '/right/image_raw',
        },
        'episodes': episodes,
    }

    out_path = session_path / 'summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n[완료] {out_path}')
    print(f'  에피소드 : {len(episodes)}개')
    print(f'  총 프레임: {total_frames} ({total_frames / FPS:.1f}s)')
    print(f'\n다음 단계:')
    print(f'  python3 merge_and_convert.py <session1> <session2> ... --output <출력경로>')


if __name__ == '__main__':
    make_summary(Path(SESSION_PATH))
