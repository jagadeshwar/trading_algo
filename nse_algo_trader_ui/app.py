"""NSE Algo Trader UI — standalone entry point.

This file lives in nse_algo_trader_ui/ and points back to the
monitoring module inside nse_algo_trader/. Run with:

    cd /Users/jthileti/Documents/Claude/Projects/Algo/nse_algo_trader_ui
    streamlit run app.py

Or use the launcher:
    python run_dashboard.py
"""

import sys
from pathlib import Path

# Add the main project to the Python path so all imports work
PROJECT_ROOT = Path(__file__).parent.parent / "nse_algo_trader"
sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to the project root so relative paths
# (configs/, data/, logs/) all resolve correctly
import os
os.chdir(str(PROJECT_ROOT))

# Hand off to the monitoring app
from production.monitoring.app import *  # noqa: F401, F403
