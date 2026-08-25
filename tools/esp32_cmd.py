#!/usr/bin/env python3
"""
esp32_cmd.py - send commands to the ESP32 firmware and print what it says back.

A serial terminal you can drive from the command line, so wiring tests do not
need a GUI monitor open (which would hold the port and block scope_view.py).

Examples:
    python tools/esp32_cmd.py --port COM5 i
    python tools/esp32_cmd.py --port COM5 "m pwm" "s 3000"
    python tools/esp32_cmd.py --port COM5 "s 40000"
    python tools/esp32_cmd.py --port COM5 --listen 5
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--listen", type=float, default=1.5,
                    help="seconds to keep reading after the last command")
    ap.add_argument("--no-reset", action="store_true",
                    help="do not pulse the reset line on connect")
    ap.add_argument("commands", nargs="*", help="commands to send, in order")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("pyserial missing:  pip install -r tools/requirements.txt", file=sys.stderr)
        return 1

    with serial.Serial(args.port, args.baud, timeout=0.3) as ser:
        if not args.no_reset:
            ser.setDTR(False)
            ser.setRTS(True)
            time.sleep(0.1)
            ser.setRTS(False)
            time.sleep(0.4)
        ser.reset_input_buffer()

        for cmd in args.commands:
            print(f">>> {cmd}")
            ser.write((cmd + "\n").encode())
            time.sleep(0.4)
            while ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"    {line}")

        deadline = time.time() + args.listen
        while time.time() < deadline:
            line = ser.readline()
            if line:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    print(f"    {text}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
