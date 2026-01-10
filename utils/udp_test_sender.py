import socket
import time
import math

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("[SENDER] UDP test sender started")
print(f"[SENDER] Sending to {UDP_IP}:{UDP_PORT}")

t = 0.0

while True:
    # Kreisbewegung im normierten Raum [0,1]
    x = 0.5 + 0.4 * math.cos(t)
    y = 0.5 + 0.4 * math.sin(t)

    msg = f"{x} {y}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

    print("[SENDER] TX:", msg)

    t += 0.08
    time.sleep(0.03)
