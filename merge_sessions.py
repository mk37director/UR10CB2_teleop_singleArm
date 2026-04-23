"""
merge_sessions.py
=================
data_collector2A.py 로 수집한 여러 세션 폴더를 하나로 병합합니다.
병합 결과는 새 폴더에 저장되며, 기존 세션 폴더는 변경하지 않습니다.
병합 후 convert_to_lerobot.py 로 변환하면 단일 데이터셋이 됩니다.

사용법:
  python3 merge_sessions.py <session1> <session2> <session3> ... \
      --output ~/outputs/lerobot_data/merged_session

예시:
  python3 merge_sessions.py \
      ~/outputs/lerobot_data/session_20260413_120000 \
      ~/outputs/lerobot_data/session_20260413_140000 \
      ~/outputs/lerobot_data/session_20260413_160000 \
      ~/outputs/lerobot_data/session_20260413_180000 \
      --output ~/outputs/lerobot_data/merged_session

동작 방식:
  - pkl 파일을 복사하지 않고 심볼릭 링크(symlink)로 연결 → 디스크 절약
  - episode_index 를 전체 기준으로 재번호 부여 (0, 1, 2, ...)
  - pkl 파일명도 새 인덱스로 재명명
  - 병합된 summary.json 생성
  - 이후 convert_to_lerobot.py 를 병합 폴더에 그대로 실행 가능
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def load_summary(session_path: Path) -> dict:
    summary_path = session_path / "summary.json"
    if not summary_path.exists():
        print(f"[ERROR] summary.json 없음: {session_path}")
        sys.exit(1)
    with open(summary_path) as f:
        return json.load(f)


def merge_sessions(session_dirs: list[str], output_dir: str):
    sessions = [Path(d).expanduser().resolve() for d in session_dirs]
    out_path = Path(output_dir).expanduser().resolve()

    # 출력 폴더 생성
    if out_path.exists():
        print(f"[WARN] 출력 폴더가 이미 존재합니다: {out_path}")
        ans = input("  덮어쓰겠습니까? (y/N): ").strip().lower()
        if ans != "y":
            print("취소합니다.")
            sys.exit(0)
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    print(f"\n[병합 시작]")
    print(f"  세션 수   : {len(sessions)}")
    print(f"  출력 경로 : {out_path}\n")

    # 세션별 summary 로드 및 검증
    summaries = []
    for s in sessions:
        summary = load_summary(s)
        summaries.append(summary)
        print(f"  [{s.name}] 에피소드: {summary['n_episodes']}, "
              f"프레임: {summary['total_frames']}")

    # fps, image_size 일치 여부 확인
    fps_set        = {s["fps"]        for s in summaries}
    image_size_set = {tuple(s["image_size"]) for s in summaries}
    if len(fps_set) > 1:
        print(f"\n[WARN] FPS가 세션마다 다릅니다: {fps_set}")
        print("  계속 진행하면 첫 번째 세션의 FPS를 사용합니다.")
    if len(image_size_set) > 1:
        print(f"\n[ERROR] image_size가 세션마다 다릅니다: {image_size_set}")
        print("  병합할 수 없습니다. 동일한 해상도로 수집된 세션만 병합 가능합니다.")
        sys.exit(1)

    fps        = summaries[0]["fps"]
    image_size = summaries[0]["image_size"]
    task       = summaries[0]["task"]

    # -------------------------------------------------------
    # pkl 파일 심볼릭 링크 + episode_index 재번호
    # -------------------------------------------------------
    merged_episodes = []
    new_ep_idx      = 0
    total_frames    = 0

    for sess_idx, (session_path, summary) in enumerate(zip(sessions, summaries)):
        print(f"\n  세션 [{sess_idx+1}/{len(sessions)}]: {session_path.name}")

        for ep_info in summary["episodes"]:
            old_pkl_name = ep_info["pkl_file"]
            old_pkl_path = session_path / old_pkl_name

            if not old_pkl_path.exists():
                print(f"    [WARN] 파일 없음, 건너뜀: {old_pkl_name}")
                continue

            # 새 episode_index 기준으로 파일명 생성
            new_pkl_name = f"episode_{new_ep_idx:06d}.pkl"
            new_pkl_path = out_path / new_pkl_name

            # 심볼릭 링크 생성 (복사 없이 원본 참조)
            new_pkl_path.symlink_to(old_pkl_path)

            # _meta.json 도 심볼릭 링크 (streaming_v1 포맷인 경우)
            old_meta_name = old_pkl_name.replace(".pkl", "_meta.json")
            old_meta_path = session_path / old_meta_name
            if old_meta_path.exists():
                new_meta_name = new_pkl_name.replace(".pkl", "_meta.json")
                new_meta_path = out_path / new_meta_name

                # meta.json은 episode_index가 바뀌므로 내용 수정 후 새 파일로 저장
                with open(old_meta_path) as f:
                    meta = json.load(f)
                meta["episode_index"] = new_ep_idx
                meta["pkl_file"]      = new_pkl_name
                with open(new_meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

            # 병합 에피소드 메타 구성
            new_ep_meta = dict(ep_info)
            new_ep_meta["episode_index"]   = new_ep_idx
            new_ep_meta["pkl_file"]        = new_pkl_name
            new_ep_meta["source_session"]  = session_path.name
            new_ep_meta["original_index"]  = ep_info["episode_index"]

            merged_episodes.append(new_ep_meta)
            total_frames += ep_info["n_frames"]

            print(f"    ep {ep_info['episode_index']:03d} → ep {new_ep_idx:03d} "
                  f"| {ep_info['n_frames']} frames "
                  f"| {old_pkl_name} → {new_pkl_name}")

            new_ep_idx += 1

    # -------------------------------------------------------
    # 병합 summary.json 저장
    # -------------------------------------------------------
    merged_summary = {
        "session_path":          str(out_path),
        "n_episodes":            new_ep_idx,
        "total_frames":          total_frames,
        "fps":                   fps,
        "task":                  task,
        "image_size":            image_size,
        "observation_state_dim": summaries[0].get("observation_state_dim", 6),
        "action_dim":            summaries[0].get("action_dim", 6),
        "topics":                summaries[0].get("topics", {}),
        "source_sessions": [
            {
                "name":       s.name,
                "path":       str(s),
                "n_episodes": summaries[i]["n_episodes"],
                "n_frames":   summaries[i]["total_frames"],
            }
            for i, s in enumerate(sessions)
        ],
        "episodes": merged_episodes,
    }

    with open(out_path / "summary.json", "w") as f:
        json.dump(merged_summary, f, indent=2)

    # -------------------------------------------------------
    # 완료 출력
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"[DONE] 병합 완료!")
    print(f"  출력 경로  : {out_path}")
    print(f"  총 에피소드: {new_ep_idx}")
    print(f"  총 프레임  : {total_frames} ({total_frames / fps:.1f}s)")
    print(f"\n세션별 구성:")
    for i, (s, summ) in enumerate(zip(sessions, summaries)):
        print(f"  [{i+1}] {s.name} "
              f"| {summ['n_episodes']} episodes "
              f"| {summ['total_frames']} frames")
    print(f"\n다음 단계:")
    print(f"  python3 convert_to_lerobot.py {out_path}")
    print(f"{'=' * 55}")


# ============================================================
# 엔트리 포인트
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="여러 수집 세션을 하나의 폴더로 병합")
    parser.add_argument(
        "sessions",
        nargs="+",
        help="병합할 세션 폴더 경로 (2개 이상)")
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="병합 결과를 저장할 폴더 경로")
    args = parser.parse_args()

    if len(args.sessions) < 2:
        print("[ERROR] 세션 폴더를 2개 이상 지정하세요.")
        sys.exit(1)

    merge_sessions(args.sessions, args.output)
