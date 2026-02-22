"""
Real-time index finger tracking with UDP streaming.

This script uses MediaPipe Hands to track the index fingertip
(landmark 8) from a webcam stream. The normalized (x, y)
coordinates are sent via UDP to a local receiver.

Intended use case:
    - Remote cursor control
    - Game interaction
    - External visualization systems

Press 'q' to exit.
"""
import cv2
import mediapipe as mp
import socket

# ----------------
# UDP
# ----------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(x, y):
    """
    Send normalized fingertip coordinates via UDP.

    The coordinates are transmitted as a space-separated string:
        "<x> <y>"

    Parameters:
        x (float): Normalized x-coordinate (0..1).
        y (float): Normalized y-coordinate (0..1).

    Returns:
        None
    """
    msg = f"{x} {y}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))


# ----------------
# MediaPipe
# ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)

print("[INFO] Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Index-Fingertip (Landmark 8)
        lm = hand.landmark[8]
        x_norm = lm.x  # 0..1
        y_norm = lm.y  # 0..1

        send(x_norm, y_norm)

        # Debug-Visualization (optional)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    cv2.imshow("Finger Tracking (Debug)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
