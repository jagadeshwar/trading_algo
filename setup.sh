#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# NSE Algo Trader — One-shot setup script
# Run this once on a new machine:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   NSE Algo Trader — Automated Setup          ║"
echo "║   Python + ML + Fyers API + TimescaleDB      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Detect setup mode ─────────────────────────────────────────────────────────
MODE=${1:-"local"}   # local | docker
info "Setup mode: $MODE"

if [ "$MODE" = "docker" ]; then
    section "Docker Setup"
    command -v docker >/dev/null 2>&1 || error "Docker not found. Install from https://docker.com"
    command -v docker-compose >/dev/null 2>&1 || error "docker-compose not found."

    info "Copying .env template..."
    [ -f nse_algo_trader/.env ] || cp nse_algo_trader/.env.example nse_algo_trader/.env
    warn "Edit nse_algo_trader/.env with your Fyers credentials before continuing."
    read -p "Press Enter once .env is filled in..."

    info "Building Docker images..."
    docker-compose build

    info "Starting database and Redis..."
    docker-compose up -d db redis
    sleep 8

    info "Database schema applied via docker-entrypoint-initdb.d ✓"

    info "Starting dashboard..."
    docker-compose up -d dashboard

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Docker setup complete!                      ║"
    echo "║                                              ║"
    echo "║  Dashboard: http://localhost:8501            ║"
    echo "║                                              ║"
    echo "║  Daily Fyers login (each morning):           ║"
    echo "║  docker-compose exec dashboard python auth.py║"
    echo "║                                              ║"
    echo "║  Start paper trading:                        ║"
    echo "║  docker-compose --profile trading up -d      ║"
    echo "╚══════════════════════════════════════════════╝"
    exit 0
fi

# ── Local setup ───────────────────────────────────────────────────────────────
section "Checking prerequisites"
command -v python3 >/dev/null 2>&1 || error "Python 3 not found. Install from python.org"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PYTHON_VERSION found"

command -v brew >/dev/null 2>&1 || error "Homebrew not found. Install from brew.sh (macOS only)"

section "Setting up virtual environment"
cd nse_algo_trader
if [ ! -d "algo_env" ]; then
    info "Creating virtual environment..."
    python3 -m venv algo_env
fi
source algo_env/bin/activate
info "Virtual environment: $(which python)"

section "Installing Python dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "All packages installed ✓"

section "Installing system dependencies (macOS)"
# libomp for XGBoost
brew list libomp >/dev/null 2>&1 || brew install libomp
info "libomp ✓"

section "Setting up PostgreSQL + TimescaleDB"
brew list postgresql@17 >/dev/null 2>&1 || brew install postgresql@17
brew list timescale/tap/timescaledb >/dev/null 2>&1 || {
    brew tap timescale/tap
    brew install timescale/tap/timescaledb
}
brew services start postgresql@17
sleep 3

# Enable TimescaleDB
PG17_CONF="/usr/local/var/postgresql@17/postgresql.conf"
grep -q "shared_preload_libraries = 'timescaledb'" "$PG17_CONF" || \
    sed -i '' "s/#shared_preload_libraries = ''/shared_preload_libraries = 'timescaledb'/" "$PG17_CONF"
timescaledb_move.sh 2>/dev/null || true
brew services restart postgresql@17
sleep 5
info "PostgreSQL 17 + TimescaleDB ✓"

section "Creating database and schema"
/usr/local/opt/postgresql@17/bin/psql -U "$USER" -d postgres -c "CREATE DATABASE nse_algo;" 2>/dev/null || \
    info "Database nse_algo already exists"
/usr/local/opt/postgresql@17/bin/psql -U "$USER" -d nse_algo -f production/db/schema.sql
info "Schema applied ✓"

section "Setting up credentials"
[ -f .env ] || cp .env.example .env
warn "Open nse_algo_trader/.env and add your:"
warn "  FYERS_APP_ID   — from myapi.fyers.in"
warn "  FYERS_SECRET   — from myapi.fyers.in"
echo ""
read -p "Press Enter once .env is filled in..."

section "Fyers authentication"
python auth.py
info "Token saved ✓"

section "Downloading historical data (this takes ~15 minutes)"
read -p "Download 2 years of OHLCV data now? [Y/n] " ans
if [[ "$ans" != "n" && "$ans" != "N" ]]; then
    python day1_run.py
    info "Data bootstrap complete ✓"
fi

section "Starting dashboard"
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Local setup complete!                       ║"
echo "║                                              ║"
echo "║  Start dashboard:                            ║"
echo "║  cd nse_algo_trader && source algo_env/bin/activate && python run_dashboard.py"
echo "║                                              ║"
echo "║  Dashboard: http://localhost:8501            ║"
echo "║                                              ║"
echo "║  Daily: run 'python auth.py' each morning    ║"
echo "║  before market opens (08:50 IST)             ║"
echo "╚══════════════════════════════════════════════╝"
