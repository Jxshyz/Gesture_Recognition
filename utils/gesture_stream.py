import threading

_latest_frame = None
_lock = threading.Lock()


def set_latest_frame(jpeg_bytes: bytes):
    global _latest_frame
    with _lock:
        _latest_frame = jpeg_bytes


def get_latest_frame():
    with _lock:
        return _latest_frame
