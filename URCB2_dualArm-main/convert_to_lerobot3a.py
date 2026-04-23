"""
convert_to_lerobot.py
=====================
data_collector2A.py 로 수집한 pkl 데이터를 LeRobot 포맷으로 변환합니다.
카메라 이미지는 mp4 비디오로 인코딩하여 저장합니다.

[지원 포맷]
  - legacy     : 구버전 (all_episodes 일괄 저장, frames 키 포함 단일 pickle)
  - chunked_v1 : 중간 버전 (meta + chunk 분할 pickle, _meta.json 없음)
  - streaming_v1: 최신 버전 (50프레임 단위 스트리밍, _meta.json 별도 존재)

사용법:
  python3 convert_to_lerobot.py <session_dir> [--repo-id yourname/dataset]

  예시:
    python3 convert_to_lerobot.py ~/outputs/lerobot_data/session_20260413_163152
    python3 convert_to_lerobot.py ~/outputs/lerobot_data/session_20260413_163152 \
        --repo-id jay/ur10_cup_task

출력 구조:
  <session_dir>/lerobot_dataset/
  ├── meta/
  │   ├── info.json
  │   ├── stats.json
  │   └── tasks.json
  ├── data/
  │   └── chunk-000/
  │       ├── episode_000000.parquet
  │       └── ...
  └── videos/
      └── chunk-000/
          ├── observation.images.cam_left_episode_000000.mp4
          ├── observation.images.cam_right_episode_000000.mp4
          └── ...
"""

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 설정
# ============================================================
DEFAULT_REPO_ID = "yourname/ur10_cup_to_blue_box"
VIDEO_CODEC     = "mp4v"   # H.264 가 필요하면 "avc1" (ffmpeg 필요)


# ============================================================
# pkl 로드 — 포맷 자동 판별
# ============================================================
def load_episode(pkl_path: Path, meta_hint: dict = None) -> dict:
    """
    pkl 파일을 포맷에 관계없이 로드하여
    {"episode_index", "n_frames", "fps", "frames": [...]} 딕셔너리를 반환.

    지원 포맷:
      legacy       : pickle.load(f) → dict with "frames" key (구버전)
      chunked_v1   : pickle.load(f) → meta_dict, 이후 chunk list 반복
      streaming_v1 : pickle.load(f) → chunk list 반복 (_meta.json 별도)
    """
    frames = []

    with open(pkl_path, "rb") as f:
        # 첫 번째 객체 로드
        try:
            first = pickle.load(f)
        except EOFError:
            print(f"  [WARN] 빈 파일: {pkl_path.name}")
            return {"episode_index": 0, "n_frames": 0, "fps": 30, "frames": []}

        # ── legacy 포맷 판별 ──────────────────────────────────
        # 첫 객체가 dict 이고 "frames" 키를 직접 포함
        if isinstance(first, dict) and "frames" in first:
            frames = first["frames"]
            result = {k: v for k, v in first.items() if k != "frames"}
            result["frames"] = frames
            return result

        # ── chunked_v1 판별 ───────────────────────────────────
        # 첫 객체가 dict 이고 "format": "chunked_v1"
        if isinstance(first, dict) and first.get("format") == "chunked_v1":
            meta = first
            n_chunks = meta.get("n_chunks", 9999)
            for _ in range(n_chunks):
                try:
                    chunk = pickle.load(f)
                    if isinstance(chunk, list):
                        frames.extend(chunk)
                        del chunk
                except EOFError:
                    break
            meta["frames"] = frames
            gc.collect()
            return meta

        # ── streaming_v1 판별 ─────────────────────────────────
        # 첫 객체가 list (chunk) — _meta.json 에서 메타 획득
        if isinstance(first, list):
            frames.extend(first)
            del first
            # 나머지 chunk 계속 로드
            while True:
                try:
                    chunk = pickle.load(f)
                    if isinstance(chunk, list):
                        frames.extend(chunk)
                        del chunk
                except EOFError:
                    break

            # _meta.json 로드
            meta_path = pkl_path.with_suffix("").parent / \
                        (pkl_path.stem + "_meta.json")
            if meta_path.exists():
                with open(meta_path) as mf:
                    meta = json.load(mf)
            elif meta_hint:
                meta = dict(meta_hint)
            else:
                meta = {
                    "episode_index": int(pkl_path.stem.split("_")[-1]),
                    "n_frames":      len(frames),
                    "fps":           30,
                    "task":          "",
                }
            meta["frames"] = frames
            gc.collect()
            return meta

        # ── 알 수 없는 포맷 ───────────────────────────────────
        print(f"  [WARN] 알 수 없는 pkl 포맷: {pkl_path.name} "
              f"(첫 객체 type={type(first).__name__})")
        return {"episode_index": 0, "n_frames": 0, "fps": 30, "frames": []}


# ============================================================
# 통계 계산
# ============================================================
def compute_stats(episodes_frames: list[list]) -> dict:
    """
    observation.state / action 정규화 통계 계산.
    episodes_frames: [[frame, ...], [frame, ...], ...]
    메모리 절약을 위해 에피소드 단위로 순회.
    """
    obs_list    = []
    action_list = []
    for frames in episodes_frames:
        for frame in frames:
            obs_list.append(frame["observation.state"])
            action_list.append(frame["action"])

    obs_arr    = np.stack(obs_list,    axis=0)
    action_arr = np.stack(action_list, axis=0)
    del obs_list, action_list
    gc.collect()

    def _stat(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
        }

    return {
        "observation.state": _stat(obs_arr),
        "action":            _stat(action_arr),
    }


# ============================================================
# 비디오 저장
# ============================================================
def save_episode_video(frames: list, cam_key: str,
                       out_path: Path, fps: int,
                       width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for frame in frames:
        img_rgb = frame[cam_key]                        # (H,W,3) RGB uint8
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    writer.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"    {out_path.name} | {len(frames)} frames | {size_mb:.1f} MB")


# ============================================================
# 메인 변환 함수
# ============================================================
def convert(session_dir: str, repo_id: str):
    session_path = Path(session_dir).expanduser().resolve()
    summary_path = session_path / "summary.json"

    if not summary_path.exists():
        print(f"[ERROR] summary.json 없음: {summary_path}")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    fps              = summary["fps"]
    task             = summary["task"]
    h, w             = summary["image_size"][0], summary["image_size"][1]
    n_episodes_total = summary["n_episodes"]

    print(f"\n[INFO] 세션 경로  : {session_path}")
    print(f"[INFO] 에피소드 수 : {n_episodes_total}")
    print(f"[INFO] 총 프레임   : {summary['total_frames']}")
    print(f"[INFO] FPS         : {fps}")

    # 출력 디렉토리 구성
    out_dir = session_path / "lerobot_dataset"
    (out_dir / "meta").mkdir(parents=True,            exist_ok=True)
    (out_dir / "data"   / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episode_meta_out = []
    all_frames_list  = []   # 통계 계산용 (frames 리스트의 리스트)
    total_frames     = 0

    # -------------------------------------------------------
    # 에피소드별 변환
    # -------------------------------------------------------
    for ep_info in summary["episodes"]:
        ep_idx   = ep_info["episode_index"]
        pkl_name = ep_info["pkl_file"]
        pkl_path = session_path / pkl_name

        if not pkl_path.exists():
            print(f"\n[WARN] 파일 없음, 건너뜀: {pkl_path.name}")
            continue

        print(f"\n[EP {ep_idx:03d}] 로드 중... {pkl_name}")
        ep     = load_episode(pkl_path, meta_hint=ep_info)
        frames = ep["frames"]
        n      = len(frames)

        if n == 0:
            print(f"  [WARN] 프레임 없음, 건너뜀")
            continue

        print(f"  프레임 수: {n} ({n / fps:.1f}s)")

        # 1) 비디오 저장
        print(f"  비디오 인코딩:")
        for cam_key in ("observation.images.cam_left",
                        "observation.images.cam_right"):
            video_name = f"{cam_key}_episode_{ep_idx:06d}.mp4"
            video_path = out_dir / "videos" / "chunk-000" / video_name
            save_episode_video(frames, cam_key, video_path, fps, w, h)

        # 2) Parquet 저장
        # observation.state / action → float32 numpy array 로 컬럼 생성
        #   pd.Series of numpy arrays → parquet 에 FixedSizeList<float32> 로 저장
        #   LeRobot 표준 dtype 일치
        n_frames_ep = len(frames)

        obs_arr    = np.stack([fr["observation.state"].astype(np.float32)
                               for fr in frames])          # (N, 6)
        action_arr = np.stack([fr["action"].astype(np.float32)
                               for fr in frames])          # (N, 6)

        df = pd.DataFrame({
            # [수정] float32 numpy array 행으로 저장 → dtype: object(array) = LeRobot 표준
            "observation.state":            list(obs_arr),
            "action":                       list(action_arr),

            # [수정] 비디오 파일 경로 문자열로 저장 (LeRobot VideoFrame 규약)
            "observation.images.cam_left": [
                f"videos/chunk-000/observation.images.cam_left_episode_{ep_idx:06d}.mp4"
            ] * n_frames_ep,
            "observation.images.cam_right": [
                f"videos/chunk-000/observation.images.cam_right_episode_{ep_idx:06d}.mp4"
            ] * n_frames_ep,

            "timestamp":     (np.arange(n_frames_ep, dtype=np.float32) / fps).tolist(),
            "frame_index":   list(range(n_frames_ep)),
            "episode_index": [ep_idx] * n_frames_ep,
            "task_index":    [0] * n_frames_ep,
            "index":         list(range(total_frames, total_frames + n_frames_ep)),
        })

        # dtype 명시적 지정
        df["timestamp"]     = df["timestamp"].astype(np.float32)
        df["frame_index"]   = df["frame_index"].astype(np.int64)
        df["episode_index"] = df["episode_index"].astype(np.int64)
        df["task_index"]    = df["task_index"].astype(np.int64)
        df["index"]         = df["index"].astype(np.int64)

        parquet_path = (out_dir / "data" / "chunk-000"
                        / f"episode_{ep_idx:06d}.parquet")
        df.to_parquet(parquet_path, index=False)

        # 저장 확인 출력
        size_kb = parquet_path.stat().st_size / 1e3
        print(f"  Parquet: {parquet_path.name} | {size_kb:.0f} KB | "
              f"obs.state dtype={df['observation.state'].iloc[0].dtype}")

        # 통계용 프레임 보관 (이미지 제외하여 메모리 절약)
        all_frames_list.append([
            {"observation.state": fr["observation.state"],
             "action":            fr["action"]}
            for fr in frames
        ])

        episode_meta_out.append({
            "episode_index": ep_idx,
            "n_frames":      n,
            "duration_sec":  round(n / fps, 2),
        })
        total_frames += n

        # 에피소드 데이터 즉시 해제
        del frames, ep, df, obs_arr, action_arr
        gc.collect()

    if total_frames == 0:
        print("\n[ERROR] 변환할 프레임이 없습니다.")
        sys.exit(1)

    # -------------------------------------------------------
    # meta/stats.json
    # -------------------------------------------------------
    print("\n[통계] 계산 중...")
    stats = compute_stats(all_frames_list)
    del all_frames_list
    gc.collect()

    with open(out_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  stats.json 저장 완료")

    # -------------------------------------------------------
    # meta/info.json
    # -------------------------------------------------------
    info = {
        "repo_id":    repo_id,
        "robot_type": "ur10",
        "fps":        fps,
        "n_episodes": len(episode_meta_out),
        "n_frames":   total_frames,
        "task":       task,

        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
                "unit":  "rad",
                "robot": "UR10_right (follower)",
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
                "unit":  "rad",
                "robot": "UR10_left (leader)",
            },
            "observation.images.cam_left": {
                "dtype":   "video",
                "shape":   [h, w, 3],
                "video_path": "videos/chunk-000/observation.images.cam_left_episode_{episode_index:06d}.mp4",
                "device":  "/dev/video2",
            },
            "observation.images.cam_right": {
                "dtype":   "video",
                "shape":   [h, w, 3],
                "video_path": "videos/chunk-000/observation.images.cam_right_episode_{episode_index:06d}.mp4",
                "device":  "/dev/video0",
            },
            "timestamp":     {"dtype": "float32", "shape": [1]},
            "frame_index":   {"dtype": "int64",   "shape": [1]},
            "episode_index": {"dtype": "int64",   "shape": [1]},
            "task_index":    {"dtype": "int64",   "shape": [1]},
            "index":         {"dtype": "int64",   "shape": [1]},
        },

        "source": {
            "collector": "data_collector2A.py",
            "topics": {
                "action":      "UR10_left/currentJ",
                "observation": "UR10_right/currentJ",
                "cam_left":    "/left/image_raw",
                "cam_right":   "/right/image_raw",
            },
        },
        "episodes": episode_meta_out,
    }

    with open(out_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("  info.json 저장 완료")

    # -------------------------------------------------------
    # meta/tasks.json
    # -------------------------------------------------------
    with open(out_dir / "meta" / "tasks.json", "w") as f:
        json.dump([{"task_index": 0, "task": task}], f, indent=2)
    print("  tasks.json 저장 완료")

    # -------------------------------------------------------
    # 완료
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"[DONE] LeRobot 포맷 변환 완료!")
    print(f"  출력 경로  : {out_dir}")
    print(f"  에피소드   : {len(episode_meta_out)}")
    print(f"  총 프레임  : {total_frames} ({total_frames / fps:.1f}s)")
    print(f"{'=' * 55}")
    print(f"\n데이터 확인:")
    print(f"  python3 -c \""
          f"import pandas as pd; "
          f"df=pd.read_parquet('{out_dir}/data/chunk-000/episode_000000.parquet'); "
          f"print(df.head(3)); print(df.dtypes)\"")
    print(f"\n학습 (ACT 예시):")
    print(f"  python3 train.py --dataset-path {out_dir} --policy act")


# ============================================================
# 엔트리 포인트
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="data_collector2A.py pkl → LeRobot 포맷 변환")
    parser.add_argument(
        "session_dir",
        help="수집 세션 경로 (summary.json 이 있는 디렉토리)")
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (기본값: {DEFAULT_REPO_ID})")
    args = parser.parse_args()

    convert(args.session_dir, args.repo_id)
