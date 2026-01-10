import socket
import time
import math

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("[SENDER] sending...")

t = 0.0
while True:
    x = 0.5 + 0.4 * math.cos(t)
    y = 0.5 + 0.4 * math.sin(t)
    msg = f"{x} {y}"
    sock.sendto(msg.encode(), ("127.0.0.1", 5005))
    print("[SENDER] TX:", msg)
    t += 0.1
    time.sleep(0.05)
