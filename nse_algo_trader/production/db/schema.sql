-- NSE Algo Trader — PostgreSQL + TimescaleDB schema
-- Run once: psql -U jthileti -d nse_algo -f production/db/schema.sql

-- ── Extension ─────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── OHLCV hypertable ──────────────────────────────────────────────────────────
-- Time-series candle data for all symbols and timeframes.
CREATE TABLE IF NOT EXISTS ohlcv (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    interval_min    SMALLINT        NOT NULL,   -- 1, 5, 15, 1440
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    UNIQUE (time, symbol, interval_min)
);
SELECT create_hypertable('ohlcv', by_range('time'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval ON ohlcv (symbol, interval_min, time DESC);

-- ── Features hypertable ───────────────────────────────────────────────────────
-- Computed ML features per bar. Column list matches FeatureEngineer output.
CREATE TABLE IF NOT EXISTS features (
    time                TIMESTAMPTZ     NOT NULL,
    symbol              TEXT            NOT NULL,
    interval_min        SMALLINT        NOT NULL,
    -- Price features
    ret_1b              DOUBLE PRECISION,
    ret_5b              DOUBLE PRECISION,
    ret_10b             DOUBLE PRECISION,
    ret_20b             DOUBLE PRECISION,
    lag_1               DOUBLE PRECISION,
    lag_2               DOUBLE PRECISION,
    lag_3               DOUBLE PRECISION,
    lag_5               DOUBLE PRECISION,
    -- Bar features
    bar_range_pct       DOUBLE PRECISION,
    close_position      DOUBLE PRECISION,
    gap_pct             DOUBLE PRECISION,
    -- EMA features
    dist_ema_9          DOUBLE PRECISION,
    dist_ema_21         DOUBLE PRECISION,
    dist_ema_50         DOUBLE PRECISION,
    dist_ema_200        DOUBLE PRECISION,
    ema_9_21_cross      DOUBLE PRECISION,
    ema_21_50_cross     DOUBLE PRECISION,
    golden_cross        DOUBLE PRECISION,
    ema_21_slope        DOUBLE PRECISION,
    -- Momentum features
    rsi                 DOUBLE PRECISION,
    rsi_oversold        DOUBLE PRECISION,
    rsi_overbought      DOUBLE PRECISION,
    rsi_slope           DOUBLE PRECISION,
    macd_norm           DOUBLE PRECISION,
    macd_hist_norm      DOUBLE PRECISION,
    macd_cross          DOUBLE PRECISION,
    -- Volatility features
    atr_pct             DOUBLE PRECISION,
    atr_ratio           DOUBLE PRECISION,
    bb_width            DOUBLE PRECISION,
    bb_pct_b            DOUBLE PRECISION,
    bb_squeeze          DOUBLE PRECISION,
    -- Volume features
    vwap_dev            DOUBLE PRECISION,
    vol_ratio           DOUBLE PRECISION,
    obv_norm            DOUBLE PRECISION,
    vol_delta_ma        DOUBLE PRECISION,
    -- Regime features
    adx                 DOUBLE PRECISION,
    di_diff             DOUBLE PRECISION,
    regime_heuristic    SMALLINT,
    -- Target
    target              SMALLINT,
    UNIQUE (time, symbol, interval_min)
);
SELECT create_hypertable('features', by_range('time'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol, interval_min, time DESC);

-- ── Signals hypertable ────────────────────────────────────────────────────────
-- Every ML signal generated — immutable SEBI audit trail.
CREATE TABLE IF NOT EXISTS signals (
    time                TIMESTAMPTZ     NOT NULL,
    signal_id           UUID            NOT NULL DEFAULT gen_random_uuid(),
    symbol              TEXT            NOT NULL,
    interval_min        SMALLINT        NOT NULL,
    direction           SMALLINT        NOT NULL,  -- 1=Long, -1=Short, 0=Flat
    confidence          DOUBLE PRECISION NOT NULL,
    regime              TEXT,
    model_version       TEXT            NOT NULL,
    strategy_version    TEXT            NOT NULL,
    git_commit          TEXT,
    acted_on            BOOLEAN         NOT NULL DEFAULT FALSE
    -- Note: no UNIQUE on signal_id alone — hypertable unique constraints must include 'time'
    -- signal_id is UUID (gen_random_uuid) so collisions are astronomically unlikely
);
SELECT create_hypertable('signals', by_range('time'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, time DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_id ON signals (signal_id, time);

-- ── Orders table (regular PostgreSQL — append-only audit trail) ───────────────
CREATE TABLE IF NOT EXISTS orders (
    id                  BIGSERIAL       PRIMARY KEY,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    signal_id           UUID,
    broker              TEXT            NOT NULL,   -- 'fyers' | 'zerodha'
    broker_order_id     TEXT,
    symbol              TEXT            NOT NULL,
    side                TEXT            NOT NULL,   -- 'BUY' | 'SELL'
    order_type          TEXT            NOT NULL,   -- 'MARKET' | 'LIMIT' | 'SL'
    qty                 INTEGER         NOT NULL,
    price               DOUBLE PRECISION,
    trigger_price       DOUBLE PRECISION,
    status              TEXT            NOT NULL,   -- 'PENDING'|'OPEN'|'FILLED'|'REJECTED'|'CANCELLED'
    reject_reason       TEXT,
    strategy_version    TEXT            NOT NULL,
    model_version       TEXT            NOT NULL,
    git_commit          TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_broker_id ON orders (broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders (created_at DESC);

-- ── Fills table (regular PostgreSQL — append-only audit trail) ────────────────
CREATE TABLE IF NOT EXISTS fills (
    id                  BIGSERIAL       PRIMARY KEY,
    filled_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    order_id            BIGINT          REFERENCES orders(id),
    broker_order_id     TEXT            NOT NULL,
    symbol              TEXT            NOT NULL,
    side                TEXT            NOT NULL,
    qty_filled          INTEGER         NOT NULL,
    fill_price          DOUBLE PRECISION NOT NULL,
    expected_price      DOUBLE PRECISION,
    slippage_pct        DOUBLE PRECISION,
    brokerage           DOUBLE PRECISION,
    stt                 DOUBLE PRECISION,
    exchange_fee        DOUBLE PRECISION,
    gst                 DOUBLE PRECISION,
    stamp_duty          DOUBLE PRECISION,
    total_cost          DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_filled_at ON fills (filled_at DESC);

-- ── Model runs table (regular PostgreSQL — MLflow metadata mirror) ─────────────
CREATE TABLE IF NOT EXISTS model_runs (
    id                  BIGSERIAL       PRIMARY KEY,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    mlflow_run_id       TEXT            NOT NULL UNIQUE,
    model_type          TEXT            NOT NULL,   -- 'direction'|'regime'|'volatility'
    model_version       TEXT            NOT NULL,
    dataset_hash        TEXT,
    feature_version     TEXT,
    strategy_version    TEXT,
    training_from       DATE            NOT NULL,
    training_to         DATE            NOT NULL,
    n_obs               INTEGER,
    n_features          INTEGER,
    purged_cv_sharpe    DOUBLE PRECISION,
    dsr                 DOUBLE PRECISION,
    live_ready          BOOLEAN         NOT NULL DEFAULT FALSE,
    git_commit          TEXT,
    india_vix_range     TEXT,
    notes               TEXT
);

-- ── Risk events table (regular PostgreSQL — circuit break log) ────────────────
CREATE TABLE IF NOT EXISTS risk_events (
    id                  BIGSERIAL       PRIMARY KEY,
    occurred_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    event_type          TEXT            NOT NULL,   -- circuit break reason name
    detail              TEXT,
    daily_pnl_pct       DOUBLE PRECISION,
    daily_dd_pct        DOUBLE PRECISION,
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT
);

-- ── Retention policy: keep raw 1min data for 90 days, downsample then ─────────
-- (TimescaleDB compression — reduces storage ~95%)
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, interval_min'
);
SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, interval_min'
);
SELECT add_compression_policy('features', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('signals', INTERVAL '30 days', if_not_exists => TRUE);
