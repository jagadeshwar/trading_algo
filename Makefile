# NSE Algo Trader — Makefile shortcuts
# Usage: make <target>

.PHONY: setup start stop dashboard trade auth bootstrap logs clean

# ── Local setup ────────────────────────────────────────────────────────────────
setup:
	bash setup.sh local

# ── Docker ────────────────────────────────────────────────────────────────────
docker-setup:
	bash setup.sh docker

docker-up:
	docker-compose up -d db redis dashboard

docker-trade:
	docker-compose --profile trading up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-auth:
	docker-compose exec dashboard python auth.py

docker-bootstrap:
	docker-compose exec dashboard python day1_run.py

# ── Local shortcuts ────────────────────────────────────────────────────────────
auth:
	cd nse_algo_trader && source algo_env/bin/activate && python auth.py

dashboard:
	cd nse_algo_trader && source algo_env/bin/activate && python run_dashboard.py

trade:
	cd nse_algo_trader && source algo_env/bin/activate && \
	python run_paper_trading.py --symbols NSE:HDFCBANK-EQ NSE:ICICIBANK-EQ NSE:RELIANCE-EQ --interval 15 --capital 1000000

bootstrap:
	cd nse_algo_trader && source algo_env/bin/activate && python day1_run.py

backtest:
	cd nse_algo_trader && source algo_env/bin/activate && \
	python -m production.strategy.backtest --symbol NSE:NIFTYBANK-INDEX --interval 15

logs:
	tail -f nse_algo_trader/logs/paper_trading_$$(date +%Y-%m-%d).log

stop:
	cd nse_algo_trader && source algo_env/bin/activate && \
	python -c "from production.monitoring.pages.trading_controls import _stop; _stop(); print('Stopped')"
	lsof -ti:8501 | xargs kill -9 2>/dev/null || true

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
