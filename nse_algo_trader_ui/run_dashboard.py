"""Launch the NSE Algo Trader dashboard.

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8502
    python run_dashboard.py --no-browser
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch NSE Algo Trader dashboard")
    parser.add_argument("--port",       default=8501, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app_path = Path(__file__).parent / "app.py"
    venv_streamlit = Path(__file__).parent.parent / "nse_algo_trader" / "algo_env" / "bin" / "streamlit"

    streamlit_cmd = str(venv_streamlit) if venv_streamlit.exists() else "streamlit"

    cmd = [
        streamlit_cmd, "run", str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#00b4d8",
        "--theme.backgroundColor", "#0e1117",
        "--theme.secondaryBackgroundColor", "#1e2130",
        "--theme.textColor", "#e8ecf4",
    ]

    print(f"\n{'='*55}")
    print("  NSE Algo Trader Dashboard")
    print(f"  URL: http://localhost:{args.port}")
    print(f"{'='*55}\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
