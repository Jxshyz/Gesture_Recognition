# utils/overlay.py
import sys
import os
import json
import socket
import threading
import math

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QPoint, pyqtSignal


CALIB_FILE = os.path.join(os.getcwd(), "overlay_calibration.json")


# =========================
# Resize-Handle Widget
# =========================
class ResizeHandle(QWidget):
    def __init__(self, parent, direction):
        super().__init__(parent)
        self.direction = direction
        self.setMouseTracking(True)
        self.start_pos = None
        self.start_geom = None

        if direction in ("left", "right"):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.SizeVerCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos()
            self.start_geom = self.parent().geometry()

    def mouseMoveEvent(self, event):
        if self.start_pos is None:
            return

        delta = event.globalPos() - self.start_pos
        g = self.start_geom

        x, y, w, h = g.x(), g.y(), g.width(), g.height()

        if self.direction == "right":
            w += delta.x()
        elif self.direction == "left":
            x += delta.x()
            w -= delta.x()
        elif self.direction == "bottom":
            h += delta.y()
        elif self.direction == "top":
            y += delta.y()
            h -= delta.y()

        self.parent().setGeometry(x, y, max(150, w), max(150, h))

    def mouseReleaseEvent(self, event):
        self.start_pos = None
        self.start_geom = None


# =========================
# Overlay Window
# =========================
class Overlay(QWidget):
    HANDLE_SIZE = 40
    DRAW_BORDER = 4

    position_received = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        print("### OVERLAY STARTED ###")

        # window flags
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # click-through toggle
        self.click_through = False

        # default size
        self.setGeometry(200, 200, 360, 740)

        # try load calibration
        self.load_calibration()

        print("[OVERLAY] F9 = Save position/size | F7 = Toggle click-through")

        # dot
        self.dot_x = 200.0
        self.dot_y = 200.0
        self.dot_radius = 6  # keep as int

        # smoothing
        self.smooth_alpha = 0.275
        self.smooth_x = None
        self.smooth_y = None

        # Drag move
        self.drag_offset = QPoint()

        # Resize handles
        self.left_handle = ResizeHandle(self, "left")
        self.right_handle = ResizeHandle(self, "right")
        self.top_handle = ResizeHandle(self, "top")
        self.bottom_handle = ResizeHandle(self, "bottom")

        # Signal connect
        self.position_received.connect(self.on_position_received)

        # UDP
        self.start_udp_listener(5005)

    # -------------------------
    # Calibration save/load
    # -------------------------
    def save_calibration(self):
        g = self.geometry()
        data = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
        try:
            with open(CALIB_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(
                f"[OVERLAY] Saved calibration: {data['x']},{data['y']},{data['w']},{data['h']}"
            )
        except Exception as e:
            print("[OVERLAY] Save calibration failed:", e)

    def load_calibration(self):
        if not os.path.exists(CALIB_FILE):
            return
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            x = int(data.get("x", 200))
            y = int(data.get("y", 200))
            w = int(data.get("w", 360))
            h = int(data.get("h", 740))
            self.setGeometry(x, y, w, h)
            print(f"[OVERLAY] Loaded calibration: {x},{y},{w},{h}")
        except Exception as e:
            print("[OVERLAY] Load calibration failed:", e)

    # -------------------------
    # Click-through toggle
    # -------------------------
    def toggle_click_through(self):
        self.click_through = not self.click_through

        flags = self.windowFlags()
        # Qt.WindowTransparentForInput makes it click-through
        if self.click_through:
            self.setWindowFlags(flags | Qt.WindowTransparentForInput)
            print("[OVERLAY] Click-through: ON")
        else:
            self.setWindowFlags(flags & ~Qt.WindowTransparentForInput)
            print("[OVERLAY] Click-through: OFF")

        self.show()  # required after changing flags

    # =========================
    # UDP
    # =========================
    def start_udp_listener(self, port):
        def listen():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", port))
            print(f"[OVERLAY] Listening on UDP {port}")

            while True:
                data, _ = sock.recvfrom(1024)
                try:
                    x, y = map(float, data.decode().strip().split())
                    self.position_received.emit(x, y)
                except Exception as e:
                    print("[OVERLAY] UDP ERROR:", e)

        threading.Thread(target=listen, daemon=True).start()

    def on_position_received(self, x, y):
        # clamp input
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))

        target_x = x * self.width()
        target_y = y * self.height()

        # smoothing
        if self.smooth_x is None:
            self.smooth_x = target_x
            self.smooth_y = target_y
        else:
            a = self.smooth_alpha
            self.smooth_x = a * target_x + (1 - a) * self.smooth_x
            self.smooth_y = a * target_y + (1 - a) * self.smooth_y

        # guard NaN/inf
        if not (math.isfinite(self.smooth_x) and math.isfinite(self.smooth_y)):
            return

        self.dot_x = self.smooth_x
        self.dot_y = self.smooth_y
        self.update()

    # =========================
    # Paint
    # =========================
    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # border
            painter.setPen(QColor(0, 255, 0))
            painter.setBrush(Qt.NoBrush)
            b = self.DRAW_BORDER
            painter.drawRect(b, b, self.width() - 2 * b, self.height() - 2 * b)

            # dot (✅ cast to int so PyQt never crashes)
            painter.setBrush(QColor(255, 0, 0))
            painter.setPen(Qt.NoPen)

            r = int(self.dot_radius)
            x = int(round(self.dot_x - r))
            y = int(round(self.dot_y - r))
            d = int(2 * r)

            painter.drawEllipse(x, y, d, d)

        except Exception as e:
            # never crash overlay
            print("[OVERLAY] paintEvent ERROR:", e)

    # =========================
    # Resize-Handles Layout
    # =========================
    def resizeEvent(self, event):
        s = self.HANDLE_SIZE
        w, h = self.width(), self.height()

        self.left_handle.setGeometry(0, s, s, h - 2 * s)
        self.right_handle.setGeometry(w - s, s, s, h - 2 * s)
        self.top_handle.setGeometry(s, 0, w - 2 * s, s)
        self.bottom_handle.setGeometry(s, h - s, w - 2 * s, s)

        super().resizeEvent(event)

    # =========================
    # Move Window (only if not click-through)
    # =========================
    def mousePressEvent(self, event):
        if self.click_through:
            return
        if event.button() == Qt.LeftButton:
            self.drag_offset = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.click_through:
            return
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_offset)

    # =========================
    # Hotkeys
    # =========================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F9:
            self.save_calibration()
            return
        if event.key() == Qt.Key_F7:
            self.toggle_click_through()
            return
        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
