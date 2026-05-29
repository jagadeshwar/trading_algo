"""Launch the NSE Algo Trader dashboard from within the project folder.

Usage (with venv active):
    python run_dashboard.py
    python run_dashboard.py --port 8502
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8501, type=int)
    args = parser.parse_args()

    app = Path(__file__).parent / "production" / "monitoring" / "app.py"
    cmd = [
        "streamlit", "run", str(app),
        "--server.port", str(args.port),
        "--theme.base", "dark",
        "--theme.primaryColor", "#00b4d8",
        "--theme.backgroundColor", "#0e1117",
        "--theme.secondaryBackgroundColor", "#1e2130",
        "--theme.textColor", "#e8ecf4",
    ]
    print(f"\nDashboard → http://localhost:{args.port}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
