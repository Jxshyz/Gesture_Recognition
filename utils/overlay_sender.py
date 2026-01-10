import cv2
import mediapipe as mp
import socket

# ---------------- UDP ----------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def crop_center_9_16(frame):
    h, w, _ = frame.shape
    target_aspect = 9 / 16
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]

    return frame

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

print("[FINGER-SENDER] started")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = crop_center_9_16(frame)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark[8]  # Index-Finger-Spitze

        x, y = lm.x, lm.y
        sock.sendto(f"{x} {y}".encode(), (UDP_IP, UDP_PORT))

        # optional Debug
        cx, cy = int(x * w), int(y * h)
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    # Debug optional
    cv2.imshow("Finger Sender", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
