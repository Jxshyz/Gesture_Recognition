import sys
import socket
import threading

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal


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

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(360, 740)

        # Punkt
        self.dot_x = 200
        self.dot_y = 200
        self.dot_radius = 5

        # Drag
        self.drag_offset = QPoint()

        # Resize-Handles
        self.left_handle = ResizeHandle(self, "left")
        self.right_handle = ResizeHandle(self, "right")
        self.top_handle = ResizeHandle(self, "top")
        self.bottom_handle = ResizeHandle(self, "bottom")

        # Signal
        self.position_received.connect(self.on_position_received)

        # UDP
        self.start_udp_listener(5005)

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
        self.dot_x = x * self.width()
        self.dot_y = y * self.height()
        self.update()

    # =========================
    # Paint
    # =========================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Rahmen
        painter.setPen(QColor(0, 255, 0))
        painter.setBrush(Qt.NoBrush)
        b = self.DRAW_BORDER
        painter.drawRect(b, b, self.width() - 2*b, self.height() - 2*b)

        # Punkt
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            self.dot_x - self.dot_radius,
            self.dot_y - self.dot_radius,
            self.dot_radius * 2,
            self.dot_radius * 2
        )

    # =========================
    # Resize-Handles Layout
    # =========================
    def resizeEvent(self, event):
        s = self.HANDLE_SIZE
        w, h = self.width(), self.height()

        self.left_handle.setGeometry(0, s, s, h - 2*s)
        self.right_handle.setGeometry(w - s, s, s, h - 2*s)
        self.top_handle.setGeometry(s, 0, w - 2*s, s)
        self.bottom_handle.setGeometry(s, h - s, w - 2*s, s)

        super().resizeEvent(event)

    # =========================
    # Move Window
    # =========================
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_offset = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_offset)

    def mouseDoubleClickEvent(self, event):
        self.dot_x = event.x()
        self.dot_y = event.y()
        self.update()


# =========================
# Main
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
