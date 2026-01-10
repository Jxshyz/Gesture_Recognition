import threading

import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_finger_position(x, y):
    """
    x, y ∈ [0,1]  (normalisiert)
    """
    msg = f"{x} {y}"
    sock.sendto(msg.encode(), ("127.0.0.1", 5005))


_latest_frame = None
_lock = threading.Lock()


def set_latest_frame(jpeg_bytes: bytes):
    global _latest_frame
    with _lock:
        _latest_frame = jpeg_bytes


def get_latest_frame():
    with _lock:
        return _latest_frame
