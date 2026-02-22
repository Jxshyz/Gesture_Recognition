"""
Lightweight UDP and frame streaming utilities for gesture applications.

This module provides:

- UDP transmission of normalized finger coordinates
- A thread-safe in-memory buffer for the latest JPEG frame

Intended for simple real-time integrations (e.g., browser preview,
external visualizers, or local game control).
"""
# utils/gesture_stream.py
import threading
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_finger_position(x, y):
    """
    Send normalized finger coordinates via UDP.

    The coordinates must be in the range [0, 1] and are transmitted
    as a space-separated string to a local UDP endpoint.

    Parameters:
        x (float): Normalized horizontal position (0..1).
        y (float): Normalized vertical position (0..1).

    Returns:
        None
    """
    msg = f"{x} {y}"
    sock.sendto(msg.encode(), ("127.0.0.1", 5005))


_latest_frame = None
_lock = threading.Lock()


def set_latest_frame(jpeg_bytes: bytes):
    """
    Store the most recent JPEG-encoded frame in a thread-safe buffer.

    If None is passed, the function returns without updating the buffer.

    Parameters:
        jpeg_bytes (bytes): JPEG-encoded image data.

    Returns:
        None
    """
    global _latest_frame
    if jpeg_bytes is None:
        return
    with _lock:
        _latest_frame = bytes(jpeg_bytes)


def get_latest_frame():
    """
    Retrieve the most recently stored JPEG frame.

    Access is synchronized to ensure thread safety.

    Returns:
        Optional[bytes]: The latest JPEG frame, or None if not set.
    """
    with _lock:
        return _latest_frame
