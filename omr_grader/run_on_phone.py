"""Start the app so phones on the same Wi-Fi can open it.

Usage:  python run_on_phone.py
Then open the printed address in the phone's browser (Chrome or Safari).
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

PORT = 8501


def lan_addresses() -> list[str]:
    found: list[str] = []
    try:
        # No packet is sent; this only asks the OS which interface would reach the internet.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return found


def main() -> int:
    addresses = lan_addresses()
    print("\nOMR Grader is starting.")
    if addresses:
        print("On your phone (connected to the same Wi-Fi), open:")
        for ip in addresses:
            print(f"    http://{ip}:{PORT}")
    else:
        print("Could not detect this computer's Wi-Fi address; find it in your network settings and open http://<address>:8501")
    print("If the phone cannot connect, allow Python through the computer's firewall when asked.")
    print("Tip: on the phone, use Upload > Take Photo; the live camera tab needs an https address.\n")
    app = Path(__file__).resolve().parent / "app.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "0.0.0.0", "--server.port", str(PORT), "--browser.gatherUsageStats", "false"]
    )


if __name__ == "__main__":
    sys.exit(main())
