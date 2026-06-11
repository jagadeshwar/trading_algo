-- NSE Algo Trader — plain PostgreSQL schema (no TimescaleDB required)
-- Compatible with Neon, Supabase, Railway, Render, or any hosted PostgreSQL.
-- Run once:  psql "$DATABASE_URL" -f production/db/schema_plain.sql

CREATE TABLE IF NOT EXISTS ohlcv (
    time            TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    interval_min    SMALLINT         NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    UNIQUE (time, symbol, interval_min)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval ON ohlcv (symbol, interval_min, time DESC);

CREATE TABLE IF NOT EXISTS features (
    time                TIMESTAMPTZ      NOT NULL,
    symbol              TEXT             NOT NULL,
    interval_min        SMALLINT         NOT NULL,
    ret_1b              DOUBLE PRECISION,
    ret_5b              DOUBLE PRECISION,
    ret_10b             DOUBLE PRECISION,
    ret_20b             DOUBLE PRECISION,
    lag_1               DOUBLE PRECISION,
    lag_2               DOUBLE PRECISION,
    lag_3               DOUBLE PRECISION,
    lag_5               DOUBLE PRECISION,
    bar_range_pct       DOUBLE PRECISION,
    close_position      DOUBLE PRECISION,
    gap_pct             DOUBLE PRECISION,
    dist_ema_9          DOUBLE PRECISION,
    dist_ema_21         DOUBLE PRECISION,
    dist_ema_50         DOUBLE PRECISION,
    dist_ema_200        DOUBLE PRECISION,
    ema_9_21_cross      DOUBLE PRECISION,
    ema_21_50_cross     DOUBLE PRECISION,
    golden_cross        DOUBLE PRECISION,
    ema_21_slope        DOUBLE PRECISION,
    rsi                 DOUBLE PRECISION,
    rsi_oversold        DOUBLE PRECISION,
    rsi_overbought      DOUBLE PRECISION,
    rsi_slope           DOUBLE PRECISION,
    macd_norm           DOUBLE PRECISION,
    macd_hist_norm      DOUBLE PRECISION,
    macd_cross          DOUBLE PRECISION,
    atr_pct             DOUBLE PRECISION,
    atr_ratio           DOUBLE PRECISION,
    bb_width            DOUBLE PRECISION,
    bb_pct_b            DOUBLE PRECISION,
    bb_squeeze          DOUBLE PRECISION,
    vwap_dev            DOUBLE PRECISION,
    vol_ratio           DOUBLE PRECISION,
    obv_norm            DOUBLE PRECISION,
    vol_delta_ma        DOUBLE PRECISION,
    adx                 DOUBLE PRECISION,
    di_diff             DOUBLE PRECISION,
    regime_heuristic    SMALLINT,
    target              SMALLINT,
    UNIQUE (time, symbol, interval_min)
);
CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol, interval_min, time DESC);

CREATE TABLE IF NOT EXISTS signals (
    time                TIMESTAMPTZ      NOT NULL,
    signal_id           UUID             NOT NULL DEFAULT gen_random_uuid(),
    symbol              TEXT             NOT NULL,
    interval_min        SMALLINT         NOT NULL,
    direction           SMALLINT         NOT NULL,
    confidence          DOUBLE PRECISION NOT NULL,
    regime              TEXT,
    model_version       TEXT             NOT NULL,
    strategy_version    TEXT             NOT NULL,
    git_commit          TEXT,
    acted_on            BOOLEAN          NOT NULL DEFAULT FALSE,
    PRIMARY KEY (signal_id, time)
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, time DESC);

CREATE TABLE IF NOT EXISTS orders (
    id                  BIGSERIAL        PRIMARY KEY,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    signal_id           UUID,
    broker              TEXT             NOT NULL,
    broker_order_id     TEXT,
    symbol              TEXT             NOT NULL,
    side                TEXT             NOT NULL,
    order_type          TEXT             NOT NULL,
    qty                 INTEGER          NOT NULL,
    price               DOUBLE PRECISION,
    trigger_price       DOUBLE PRECISION,
    status              TEXT             NOT NULL,
    reject_reason       TEXT,
    strategy_version    TEXT             NOT NULL,
    model_version       TEXT             NOT NULL,
    git_commit          TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_broker_id  ON orders (broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders (created_at DESC);

CREATE TABLE IF NOT EXISTS fills (
    id                  BIGSERIAL        PRIMARY KEY,
    filled_at           TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    order_id            BIGINT           REFERENCES orders(id),
    broker_order_id     TEXT             NOT NULL,
    symbol              TEXT             NOT NULL,
    side                TEXT             NOT NULL,
    qty_filled          INTEGER          NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_fills_order_id  ON fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_filled_at ON fills (filled_at DESC);

CREATE TABLE IF NOT EXISTS model_runs (
    id                  BIGSERIAL        PRIMARY KEY,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    mlflow_run_id       TEXT             NOT NULL UNIQUE,
    model_type          TEXT             NOT NULL,
    model_version       TEXT             NOT NULL,
    dataset_hash        TEXT,
    feature_version     TEXT,
    strategy_version    TEXT,
    training_from       DATE             NOT NULL,
    training_to         DATE             NOT NULL,
    n_obs               INTEGER,
    n_features          INTEGER,
    purged_cv_sharpe    DOUBLE PRECISION,
    dsr                 DOUBLE PRECISION,
    live_ready          BOOLEAN          NOT NULL DEFAULT FALSE,
    git_commit          TEXT,
    india_vix_range     TEXT,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS risk_events (
    id                  BIGSERIAL        PRIMARY KEY,
    occurred_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    event_type          TEXT             NOT NULL,
    detail              TEXT,
    daily_pnl_pct       DOUBLE PRECISION,
    daily_dd_pct        DOUBLE PRECISION,
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT
);
