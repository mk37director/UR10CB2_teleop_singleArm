"""
act_server.py
=============
Python 3.12 + lerobot_env 에서 실행.
ACT 정책을 로드하고 TCP 소켓으로 추론 요청을 처리합니다.

실행:
  source ~/lerobot_env/bin/activate
  python3 act_server.py
"""

import socket
import json
import struct
import numpy as np
import torch
from pathlib import Path

# ============================================================
# 설정
# ============================================================
CHECKPOINT_PATH = '/home/ur12e/outputs/train/act_ur10/checkpoints/last/pretrained_model'
DEVICE          = 'cuda'
HOST            = '127.0.0.1'
PORT            = 55555
IMAGE_HEIGHT    = 480
IMAGE_WIDTH     = 640


# ============================================================
# 소켓 유틸
# ============================================================
def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('소켓 연결 끊김')
        buf += chunk
    return buf

def send_msg(sock, data: bytes):
    sock.sendall(struct.pack('>I', len(data)) + data)

def recv_msg(sock) -> bytes:
    raw_len = recv_exact(sock, 4)
    msg_len = struct.unpack('>I', raw_len)[0]
    return recv_exact(sock, msg_len)


# ============================================================
# ACT 추론
# ============================================================
def run_server():
    from lerobot.policies.act.modeling_act import ACTPolicy

    print(f'[INFO] ACT 정책 로드 중: {CHECKPOINT_PATH}')
    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()
    print(f'[INFO] ACT 정책 로드 완료')

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f'[INFO] 서버 대기 중: {HOST}:{PORT}')

    action_buf = []

    while True:
        conn, addr = server.accept()
        print(f'[INFO] 클라이언트 연결: {addr}')
        action_buf = []

        try:
            while True:
                raw = recv_msg(conn)
                req = json.loads(raw.decode())
                cmd = req.get('cmd')

                if cmd == 'reset':
                    action_buf = []
                    send_msg(conn, json.dumps({'status': 'ok'}).encode())

                elif cmd == 'infer':
                    # action chunk 소진 시 새로 예측
                    if len(action_buf) == 0:
                        joints    = np.array(req['joints'],   dtype=np.float32)
                        cam_left  = np.array(req['cam_left'], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
                        cam_right = np.array(req['cam_right'],dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)

                        def img_tensor(img):
                            t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
                            return t.unsqueeze(0).to(DEVICE)

                        obs = {
                            'observation.state':                torch.from_numpy(joints).unsqueeze(0).to(DEVICE),
                            'observation.images.cam_left':      img_tensor(cam_left),
                            'observation.images.cam_right':     img_tensor(cam_right),
                        }

                        with torch.inference_mode():
                            action = policy.select_action(obs)
                            if action.dim() == 3:
                                action = action.squeeze(0)
                            action_buf = action.cpu().numpy().tolist()

                    joint_cmd = action_buf.pop(0)
                    resp = {'status': 'ok', 'action': joint_cmd, 'buf_len': len(action_buf)}
                    send_msg(conn, json.dumps(resp).encode())

                elif cmd == 'ping':
                    send_msg(conn, json.dumps({'status': 'pong'}).encode())

        except (ConnectionError, ConnectionResetError):
            print('[INFO] 클라이언트 연결 종료')
        except Exception as e:
            print(f'[ERROR] {e}')
        finally:
            conn.close()


if __name__ == '__main__':
    run_server()
