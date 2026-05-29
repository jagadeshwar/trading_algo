"""Integration tests — broker rejection and retry logic."""

import pytest


class TestRetryHandling:
    def test_retry_on_transient_rejection():
        # TODO: mock broker returning REJECT → ACCEPT on second attempt
        # Verify order is retried up to N times with backoff
        pass

    def test_no_retry_on_margin_rejection():
        # TODO: MARGIN_INSUFFICIENT should not be retried
        pass

    def test_no_retry_on_invalid_symbol():
        # TODO: INVALID_SYMBOL should not be retried
        pass

    def test_retry_count_increments_rejection_tracker():
        # TODO: verify circuit breaker rejection counter increments correctly
        pass

    def test_circuit_break_after_max_rejections():
        # TODO: after 3 rejections in 10 min, circuit breaker should trip
        pass
