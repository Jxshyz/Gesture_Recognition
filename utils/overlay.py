import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QPoint, QRect

import socket
import threading


class Overlay(QWidget):
    BORDER = 6

    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(400, 400)

        self.dot_x = 200
        self.dot_y = 200
        self.dot_radius = 5

        self.drag_offset = QPoint()
        self.resizing = False
        self.resize_dir = None
        self.start_geom = QRect()
        self.start_mouse = QPoint()
        
        self.start_udp_listener(port=5005)

        
    # ----------- LISTENER --------------
    """
    def start_udp_listener(self, port=5005):
        def listen():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", port))

            while True:
                try:
                    data, _ = sock.recvfrom(1024)
                    msg = data.decode().strip()
                    x, y = map(float, msg.split())

                    # 🔁 Koordinaten: 0–1 → Fenster
                    self.dot_x = x * self.width()
                    self.dot_y = y * self.height()
                    self.update()
                except Exception:
                    pass

        t = threading.Thread(target=listen, daemon=True)
        t.start()
    """
    
    def start_udp_listener(self, port=5005):
        def listen():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", port))
            print(f"[OVERLAY] Listening on UDP {port}")

            while True:
                data, addr = sock.recvfrom(1024)
                print("[OVERLAY] RAW:", data, "FROM", addr)

                try:
                    msg = data.decode().strip()
                    x, y = map(float, msg.split())
                    print("[OVERLAY] PARSED:", x, y)

                    self.dot_x = x * self.width()
                    self.dot_y = y * self.height()
                    self.update()
                except Exception as e:
                    print("[OVERLAY] ERROR:", e)

        threading.Thread(target=listen, daemon=True).start()


    # ---------------- DRAW ----------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Rahmen
        painter.setPen(QColor(0, 255, 0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(1, 1, self.width() - 2, self.height() - 2)

        # Punkt
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            self.dot_x - self.dot_radius,
            self.dot_y - self.dot_radius,
            self.dot_radius * 2,
            self.dot_radius * 2
        )

    # ---------------- HIT TEST ----------------
    def hit_test(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        b = self.BORDER

        left   = x < b
        right  = x > w - b
        top    = y < b
        bottom = y > h - b

        if top and left: return "tl"
        if top and right: return "tr"
        if bottom and left: return "bl"
        if bottom and right: return "br"
        if left: return "l"
        if right: return "r"
        if top: return "t"
        if bottom: return "b"
        return None

    # ---------------- MOUSE ----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.resize_dir = self.hit_test(event.pos())
            self.start_geom = self.geometry()
            self.start_mouse = event.globalPos()

            if self.resize_dir:
                self.resizing = True
            else:
                self.drag_offset = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.resizing:
            delta = event.globalPos() - self.start_mouse
            g = self.start_geom

            x, y, w, h = g.x(), g.y(), g.width(), g.height()

            if "r" in self.resize_dir:
                w += delta.x()
            if "l" in self.resize_dir:
                x += delta.x()
                w -= delta.x()
            if "b" in self.resize_dir:
                h += delta.y()
            if "t" in self.resize_dir:
                y += delta.y()
                h -= delta.y()

            self.setGeometry(x, y, max(100, w), max(100, h))

        elif event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_offset)

        # Cursor ändern
        dir = self.hit_test(event.pos())
        cursors = {
            "l": Qt.SizeHorCursor,
            "r": Qt.SizeHorCursor,
            "t": Qt.SizeVerCursor,
            "b": Qt.SizeVerCursor,
            "tl": Qt.SizeFDiagCursor,
            "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor,
            "bl": Qt.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(dir, Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.resize_dir = None

    # Punkt per Doppelklick verschieben
    def mouseDoubleClickEvent(self, event):
        self.dot_x = event.x()
        self.dot_y = event.y()
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())

