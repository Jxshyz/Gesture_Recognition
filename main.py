import sys

from utils.cam_test import run_test_cam
from utils.record_data import run_record


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>\nCommands: test_cam | record_data")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test_cam":
        cam_idx = 0
        if len(sys.argv) >= 3:
            try:
                cam_idx = int(sys.argv[2])
            except ValueError:
                pass
        run_test_cam(camera_index=cam_idx)

    elif cmd == "record_data":
        if len(sys.argv) < 4:
            print("Usage: python main.py record_data <l|r> <Name> [camera_index]")
            sys.exit(2)
        hand_arg = sys.argv[2]
        name = sys.argv[3]
        cam_idx = 0
        if len(sys.argv) >= 5:
            try:
                cam_idx = int(sys.argv[4])
            except ValueError:
                pass
        run_record(hand_arg=hand_arg, name=name, camera_index=cam_idx)

    else:
        print(f"Unbekannter Befehl: {cmd}\nVerfügbare Befehle: test_cam | record_data")
        sys.exit(2)


if __name__ == "__main__":
    main()
