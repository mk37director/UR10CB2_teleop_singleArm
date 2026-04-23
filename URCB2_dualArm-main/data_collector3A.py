"""
UR10 Dual Arm - LeRobot Data Collector (카메라 포함)
=====================================================
[메모리 수정 사항 v4 - 스트리밍 저장]
  근본 원인: episode_buffer 가 recording 중 통째로 RAM 에 누적됨
             30Hz × 2cam × 1.76MB/frame → 60초 에피소드 = ~6GB RAM
  해결책:    recording 중 STREAM_CHUNK 프레임마다 즉시 디스크에 append 저장
             → episode_buffer 는 최대 STREAM_CHUNK(50) 프레임만 RAM 점유
             → 에피소드 길이와 무관하게 RAM 사용량 일정

토픽 구조:
  [Joint]
    UR10_left/currentJ    → action            (Float64MultiArray, 6DoF, rad)
    UR10_right/currentJ   → observation.state  (Float64MultiArray, 6DoF, rad)
    UR10_right/currentP   → Cartesian pose    (optional)
    UR10_left/ctlMode     → 리더 제어모드 확인

  [Camera - v4l2_camera 노드]
    /left/image_raw   → 왼쪽 웹캠  (/dev/video2, C922)
    /right/image_raw  → 오른쪽 웹캠 (/dev/video0, C922)

카메라 노드 실행 (별도 터미널):
  ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video0" -r __ns:="/right" -r __node:="camera"
  ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p video_device:="/dev/video2" -r __ns:="/left"  -r __node:="camera"

사용법:
  python3 data_collector2A.py
  저장 경로: ~/outputs/lerobot_data/session_YYYYMMDD_HHMMSS/
  [S] 에피소드 시작
  [E] 에피소드 종료 + 성공 저장 (즉시 디스크 저장 후 RAM 해제)
  [D] 에피소드 버림 (discard, RAM 즉시 해제)
  [I] 현재 상태 확인 (RAM 사용량 포함)
  [Q] 전체 종료 및 summary.json 저장
"""

import gc
import io
import json
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String


# ============================================================
# 설정값
# ============================================================
COLLECT_HZ       = 30
IMAGE_WIDTH      = 640
IMAGE_HEIGHT     = 480
STREAM_CHUNK     = 50    # 프레임 N개마다 즉시 디스크에 flush
OUTPUT_DIR       = str(Path.home() / "outputs" / "lerobot_data")
TASK_DESCRIPTION = "Pick up the object and place it inside the blue square on the table."

# Joint 토픽
TOPIC_LEADER_J    = "UR10_left/currentJ"
TOPIC_FOLLOWER_J  = "UR10_right/currentJ"
TOPIC_FOLLOWER_P  = "UR10_right/currentP"
TOPIC_LEADER_MODE = "UR10_left/ctlMode"

# 카메라 토픽
TOPIC_CAM_LEFT  = "/left/image_raw"
TOPIC_CAM_RIGHT = "/right/image_raw"


# ============================================================
# 데이터 수집 노드
# ============================================================
class DataCollectorNode(Node):
    def __init__(self, session_path: Path):
        super().__init__('lerobot_data_collector')

        self.bridge = CvBridge()

        # [수정] 세션 경로를 생성자에서 받아 즉시 사용
        self.session_path = session_path
        self.session_path.mkdir(parents=True, exist_ok=True)

        # 최신 데이터 버퍼 (latest 1프레임만 유지 — 누적 없음)
        self.latest = {
            "leader_joints":   None,
            "follower_joints": None,
            "follower_pose":   None,
            "cam_left":        None,
            "cam_right":       None,
            "leader_mode":     "Unknown",
        }
        self._lock = threading.Lock()

        # 카메라 수신 여부 로그용
        self.cam_left_received  = False
        self.cam_right_received = False

        # 수집 상태
        self.recording       = False
        self.episode_buffer  = []     # 임시 버퍼 (STREAM_CHUNK 크기만 유지)
        self.episode_count   = 0

        # 스트리밍 저장 상태 — recording 중 열려있는 pkl 파일 핸들
        self._stream_file    = None   # open file object
        self._stream_pickler = None   # Pickler instance
        self._stream_path    = None   # 현재 기록 중인 pkl 경로
        self._stream_frames  = 0      # 기록된 프레임 수

        # 메타데이터만 보관 (이미지 없음)
        self.episode_meta    = []

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        self.create_subscription(
            Float64MultiArray, TOPIC_LEADER_J,
            lambda msg: self._cb_joints(msg, "leader"), 1)

        self.create_subscription(
            Float64MultiArray, TOPIC_FOLLOWER_J,
            lambda msg: self._cb_joints(msg, "follower"), 1)

        self.create_subscription(
            Float64MultiArray, TOPIC_FOLLOWER_P,
            self._cb_follower_pose, 1)

        self.create_subscription(
            String, TOPIC_LEADER_MODE,
            self._cb_leader_mode, 10)

        self.create_subscription(
            Image, TOPIC_CAM_LEFT,
            lambda msg: self._cb_image(msg, "cam_left"), 1)

        self.create_subscription(
            Image, TOPIC_CAM_RIGHT,
            lambda msg: self._cb_image(msg, "cam_right"), 1)

        # 수집 타이머
        self.create_timer(1.0 / COLLECT_HZ, self._record_frame)

        self.get_logger().info("=== LeRobot Data Collector (메모리 최적화) ===")
        self.get_logger().info(f"  저장 경로   : {self.session_path}")
        self.get_logger().info(f"  Collect Hz  : {COLLECT_HZ}")

    # ----------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------
    def _cb_joints(self, msg, who):
        with self._lock:
            data = np.array(msg.data, dtype=np.float32)
            self.latest["leader_joints" if who == "leader" else "follower_joints"] = data

    def _cb_follower_pose(self, msg):
        with self._lock:
            self.latest["follower_pose"] = np.array(msg.data, dtype=np.float32)

    def _cb_leader_mode(self, msg):
        with self._lock:
            self.latest["leader_mode"] = msg.data

    def _cb_image(self, msg, cam_key):
        try:
            enc = msg.encoding

            if enc in ("yuv422_yuy2", "yuv422", "yuyv", "YUYV"):
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                img = cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUYV)
            elif enc in ("mono8", "8UC1"):
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            if img.shape[0] != IMAGE_HEIGHT or img.shape[1] != IMAGE_WIDTH:
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

            with self._lock:
                # [수정] 이전 latest 이미지를 명시적으로 del 후 교체
                #        → 참조 해제 즉시 GC 대상이 되도록
                old = self.latest.get(cam_key)
                if old is not None:
                    del old
                self.latest[cam_key] = img  # copy() 제거 — img 자체를 저장

            if cam_key == "cam_left" and not self.cam_left_received:
                self.cam_left_received = True
                self.get_logger().info(
                    f"[OK] 왼쪽 카메라 수신 | encoding: {enc} | "
                    f"size: {img.shape[1]}x{img.shape[0]}")
            elif cam_key == "cam_right" and not self.cam_right_received:
                self.cam_right_received = True
                self.get_logger().info(
                    f"[OK] 오른쪽 카메라 수신 | encoding: {enc} | "
                    f"size: {img.shape[1]}x{img.shape[0]}")

        except Exception as e:
            self.get_logger().warn(
                f"Image 변환 오류 ({cam_key}) encoding={msg.encoding}: {e}",
                throttle_duration_sec=3.0)

    # ----------------------------------------------------------
    # 프레임 수집 타이머 콜백
    # ----------------------------------------------------------
    def _record_frame(self):
        # recording 확인을 lock 안에서 수행 → stop_episode() 경쟁 조건 차단
        with self._lock:
            if not self.recording:
                return

            missing = [k for k, v in self.latest.items()
                       if v is None and k not in ("follower_pose", "leader_mode")]
            if missing:
                self.get_logger().warn(
                    f"데이터 대기 중: {missing}", throttle_duration_sec=2.0)
                return

            frame = {
                "observation.state":            self.latest["follower_joints"].copy(),
                "action":                       self.latest["leader_joints"].copy(),
                "observation.images.cam_left":  self.latest["cam_left"].copy(),
                "observation.images.cam_right": self.latest["cam_right"].copy(),
                "observation.state_cartesian": (
                    self.latest["follower_pose"].copy()
                    if self.latest["follower_pose"] is not None
                    else np.zeros(6, dtype=np.float32)
                ),
                "leader_mode": self.latest["leader_mode"],
                "timestamp":   time.time(),
            }

        self.episode_buffer.append(frame)
        self._stream_frames += 1

        # [v4] STREAM_CHUNK 프레임마다 즉시 디스크에 flush → RAM 해제
        if len(self.episode_buffer) >= STREAM_CHUNK:
            self._flush_buffer()

    # ----------------------------------------------------------
    # 에피소드 제어
    # ----------------------------------------------------------
    def start_episode(self) -> bool:
        with self._lock:
            missing = [k for k, v in self.latest.items()
                       if v is None and k not in ("follower_pose", "leader_mode")]
        if missing:
            print(f"\n[ERROR] 수신 안 된 데이터: {missing}")
            if "cam_left" in missing:
                print('  → ros2 run v4l2_camera v4l2_camera_node --ros-args '
                      '-p video_device:="/dev/video2" -r __ns:="/left" -r __node:="camera"')
            if "cam_right" in missing:
                print('  → ros2 run v4l2_camera v4l2_camera_node --ros-args '
                      '-p video_device:="/dev/video0" -r __ns:="/right" -r __node:="camera"')
            return False

        self.episode_buffer  = []
        self._stream_frames  = 0

        # [v4] recording 시작과 동시에 임시 pkl 파일 오픈 (스트리밍 저장 준비)
        tmp_name = f"episode_{self.episode_count:06d}.pkl"
        self._stream_path = self.session_path / tmp_name
        self._stream_file = open(self._stream_path, "wb")

        # 메타데이터는 나중에 덮어쓸 수 없으므로 placeholder 로 저장
        # stop_episode() 에서 실제 n_frames 확정 후 별도 meta 파일로 저장
        self._stream_pickler = pickle.Pickler(
            self._stream_file, protocol=pickle.HIGHEST_PROTOCOL)

        self.recording = True
        print(f"\n[REC ▶] Episode {self.episode_count} 시작 | "
              f"물건을 집어서 파란 박스로 이동 후 [E] 입력")
        return True

    def _flush_buffer(self):
        """
        [v4] episode_buffer 를 스트림 파일에 즉시 기록 후 RAM 에서 해제.
        Pickler memo 누수를 막기 위해 청크마다 새 Pickler 생성.
        """
        if not self.episode_buffer or self._stream_file is None:
            return

        chunk = self.episode_buffer          # 현재 버퍼 참조
        self.episode_buffer = []             # 즉시 새 빈 리스트로 교체 → RAM 해제

        # 새 Pickler 로 chunk 저장 → memo 캐시가 이 chunk 에만 국한됨
        pickler = pickle.Pickler(self._stream_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(chunk)
        self._stream_file.flush()            # OS 버퍼 → 디스크 즉시 기록

        del pickler                          # memo 해제
        del chunk
        gc.collect()

    def stop_episode(self, success: bool):
        # recording=False 와 버퍼 처리를 lock 안에서 원자적으로 수행
        with self._lock:
            self.recording = False
            remaining      = self.episode_buffer
            self.episode_buffer = []

        # 남은 버퍼 flush
        if remaining and self._stream_file is not None:
            pickler = pickle.Pickler(self._stream_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickler.dump(remaining)
            self._stream_file.flush()
            del pickler
        del remaining

        n = self._stream_frames

        # 스트림 파일 닫기
        if self._stream_file is not None:
            self._stream_file.close()
            self._stream_file    = None
            self._stream_pickler = None

        if n == 0:
            # 프레임 없음 → 임시 파일 삭제
            if self._stream_path and self._stream_path.exists():
                self._stream_path.unlink()
            self._stream_path = None
            print("[WARN] 프레임 없음. 버립니다.")
            return

        pkl_name = self._stream_path.name
        self._stream_path = None

        if success:
            # 메타데이터 별도 json 저장
            meta = {
                "episode_index": self.episode_count,
                "n_frames":      n,
                "duration_sec":  round(n / COLLECT_HZ, 2),
                "fps":           COLLECT_HZ,
                "task":          TASK_DESCRIPTION,
                "pkl_file":      pkl_name,
                "chunk_size":    STREAM_CHUNK,
                "format":        "streaming_v1",
                "success":       True,
            }
            meta_path = self.session_path / pkl_name.replace(".pkl", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            size_mb = (self.session_path / pkl_name).stat().st_size / 1e6
            print(f"  → 저장 완료: {pkl_name} | {size_mb:.1f} MB")

            self.episode_meta.append(meta)
            self.episode_count += 1
            print(f"[SAVE ✓] Episode {self.episode_count - 1} | "
                  f"{n} frames ({n / COLLECT_HZ:.1f}s) | {pkl_name}")
        else:
            # 실패 → 임시 파일 삭제
            tmp = self.session_path / pkl_name
            if tmp.exists():
                tmp.unlink()
            print(f"[DISC ✗] 버림 | {n} frames ({n / COLLECT_HZ:.1f}s)")

        gc.collect()
        gc.collect()
        self._print_ram_usage()

    # _save_episode_now → v4 에서 _flush_buffer + stop_episode 로 대체됨

    # ----------------------------------------------------------
    # 세션 종료 시 summary.json 저장
    # ----------------------------------------------------------
    def save_summary(self) -> str | None:
        if not self.episode_meta:
            print("[WARN] 저장된 에피소드 없음.")
            return None

        total_frames = sum(e["n_frames"] for e in self.episode_meta)
        summary = {
            "session_path":          str(self.session_path),
            "n_episodes":            len(self.episode_meta),
            "total_frames":          total_frames,
            "fps":                   COLLECT_HZ,
            "task":                  TASK_DESCRIPTION,
            "image_size":            [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "observation_state_dim": 6,
            "action_dim":            6,
            "topics": {
                "action":      TOPIC_LEADER_J,
                "observation": TOPIC_FOLLOWER_J,
                "cam_left":    TOPIC_CAM_LEFT,
                "cam_right":   TOPIC_CAM_RIGHT,
            },
            "episodes": self.episode_meta,
        }

        summary_path = self.session_path / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[DONE] 세션 종료")
        print(f"  경로       : {self.session_path}")
        print(f"  에피소드   : {len(self.episode_meta)}")
        print(f"  총 프레임  : {total_frames} ({total_frames / COLLECT_HZ:.1f}s)")
        print(f"  summary    : {summary_path}")
        print(f"\n다음 단계:")
        print(f"  python3 convert_to_lerobot.py {self.session_path}")
        return str(self.session_path)

    # ----------------------------------------------------------
    # 상태 출력 (RAM 사용량 포함)
    # ----------------------------------------------------------
    def print_status(self):
        with self._lock:
            lj   = self.latest["leader_joints"]
            fj   = self.latest["follower_joints"]
            mode = self.latest["leader_mode"]
            cl   = self.latest["cam_left"]
            cr   = self.latest["cam_right"]

        print("\n--- 현재 상태 ---")
        print(f"  Leader mode  : {mode}")
        print(f"  Leader  J    : " +
              (f"{np.round(np.degrees(lj), 1)} [deg]" if lj is not None else "수신 대기 중"))
        print(f"  Follower J   : " +
              (f"{np.round(np.degrees(fj), 1)} [deg]" if fj is not None else "수신 대기 중"))
        print(f"  Cam Left     : " +
              (f"OK {cl.shape}" if cl is not None else "미수신 → /dev/video2 확인"))
        print(f"  Cam Right    : " +
              (f"OK {cr.shape}" if cr is not None else "미수신 → /dev/video0 확인"))
        print(f"  저장 경로    : {self.session_path}")
        print(f"  에피소드 수  : {self.episode_count} (저장 완료)")
        if self.recording:
            n = len(self.episode_buffer)
            print(f"  [REC▶] {n} frames ({n / COLLECT_HZ:.1f}s)")
        self._print_ram_usage()
        print("-----------------\n")

    def _print_ram_usage(self):
        """현재 프로세스 RAM 사용량 출력 (psutil 없을 시 생략)."""
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            ram_mb = proc.memory_info().rss / 1e6
            print(f"  [RAM] 현재 사용량: {ram_mb:.0f} MB")
        except ImportError:
            pass   # psutil 미설치 시 무시


# ============================================================
# 메인
# ============================================================
def main():
    rclpy.init()

    # [수정] 세션 경로를 시작 시 1회 생성 → 에피소드별 즉시 저장에 사용
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = Path(OUTPUT_DIR) / f"session_{timestamp}"

    node = DataCollectorNode(session_path)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 카메라 첫 프레임 수신될 때까지 대기
    print("\n[대기] 카메라 수신 확인 중...", end="", flush=True)
    while rclpy.ok():
        with node._lock:
            cl = node.latest["cam_left"]
            cr = node.latest["cam_right"]
        if cl is not None and cr is not None:
            print(" 완료!")
            print(f"  Cam Left  : OK {cl.shape}")
            print(f"  Cam Right : OK {cr.shape}\n")
            break
        print(".", end="", flush=True)
        time.sleep(0.5)

    print("\n" + "=" * 58)
    print("  UR10 LeRobot Data Collector  (메모리 최적화)")
    print("=" * 58)
    print("  [사전 조건]")
    print("  1) dualArm_motion 또는 singleArm_motion 노드 실행 중")
    print("  2) 카메라 노드 실행 (위 참조)")
    print("-" * 58)
    print("  [S] 에피소드 시작")
    print("  [E] 저장(성공)  ← 즉시 디스크 저장 후 RAM 해제")
    print("  [D] 버림(실패)  ← 즉시 RAM 해제")
    print("  [I] 상태 확인 (RAM 사용량 포함)")
    print("  [Q] 종료 및 summary.json 저장")
    print("=" * 58 + "\n")

    try:
        while rclpy.ok():
            cmd = input("명령 (S/E/D/I/Q): ").strip().upper()

            if cmd == "S":
                node.start_episode()

            elif cmd == "E":
                if node.recording:
                    node.stop_episode(success=True)
                else:
                    print("[WARN] 기록 중 아님")

            elif cmd == "D":
                if node.recording:
                    node.stop_episode(success=False)
                else:
                    print("[WARN] 기록 중 아님")

            elif cmd == "I":
                node.print_status()

            elif cmd == "Q":
                if node.recording:
                    print("[WARN] 먼저 E 또는 D로 에피소드를 종료하세요.")
                    continue
                node.save_summary()
                break

            else:
                print("[WARN] 알 수 없는 명령입니다.")

    except KeyboardInterrupt:
        print("\n[INFO] 인터럽트. 종료합니다.")
        if node.recording:
            node.stop_episode(success=False)
        node.save_summary()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
