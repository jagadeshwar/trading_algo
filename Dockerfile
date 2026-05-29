# NSE Algo Trader — Production Docker Image
# Base: Python 3.11 slim on Linux (XGBoost works without libomp on Linux)
FROM python:3.11-slim

LABEL maintainer="FAJ79402"
LABEL description="NSE Algorithmic Trading System — Fyers API + ML + Streamlit"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY nse_algo_trader/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY nse_algo_trader/ ./nse_algo_trader/
COPY nse_algo_trader_ui/ ./nse_algo_trader_ui/

# ── Persistent data directories (mounted as volumes in production) ─────────────
RUN mkdir -p \
    nse_algo_trader/data/ohlcv \
    nse_algo_trader/data/features \
    nse_algo_trader/data/options \
    nse_algo_trader/models \
    nse_algo_trader/logs

# ── Default working directory for all commands ────────────────────────────────
WORKDIR /app/nse_algo_trader

# ── Ports: 8501 = Streamlit dashboard, 5000 = Fyers auth callback ─────────────
EXPOSE 8501 5000

# ── Default: start the dashboard ─────────────────────────────────────────────
CMD ["streamlit", "run", "production/monitoring/app.py", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--theme.base", "dark", \
     "--theme.primaryColor", "#00b4d8", \
     "--theme.backgroundColor", "#0e1117", \
     "--theme.secondaryBackgroundColor", "#1e2130", \
     "--theme.textColor", "#e8ecf4"]
