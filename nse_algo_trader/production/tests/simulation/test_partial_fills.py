"""Simulation tests — partial fill order management."""

import pytest


class TestPartialFills:
    def test_partial_fill_updates_position_correctly(self):
        # TODO: order for 10 lots filled for 8 → position = 8, pending = 2
        pass

    def test_remaining_quantity_tracked(self):
        # TODO: after partial fill, remaining qty should be tracked for re-submission
        pass

    def test_fill_at_different_prices_averaged(self):
        # TODO: 5 lots @ 100, 5 lots @ 102 → avg fill price = 101
        pass

    def test_stop_loss_placed_on_partial_fill(self):
        # TODO: stop should be placed immediately on partial fill, not waiting for full fill
        pass

    def test_cancel_remainder_on_signal_reversal(self):
        # TODO: if signal reverses before full fill, remaining order should be cancelled
        pass

    def test_80_pct_fill_rate_modeled_in_backtest(self):
        # TODO: limit orders in backtest should use 80% fill rate from broker.yaml
        pass
