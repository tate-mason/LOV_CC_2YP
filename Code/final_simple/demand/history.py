"""Pure calendar-time habit-stock transition, shared by estimation and pricing."""
import numpy as np

DEFAULT_STOCK_RETENTION = 0.9


def advance_stock(stock, share, weeks=1, retention=DEFAULT_STOCK_RETENTION):
    """Pre-choice stock next time: rho**weeks * (current stock + current share).

    Shares arrive at the current observation; unobserved weeks add no purchases.
    Retention one reproduces the historical cumulative sum.
    """
    if not np.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError("Stock retention must lie in [0, 1].")
    if not np.isfinite(weeks) or weeks <= 0:
        raise ValueError("Stock transitions require a positive calendar-week gap.")
    return retention**weeks * (np.asarray(stock) + np.asarray(share))


DEFAULT_STOCK_MEASURE = "average"


def saved_stock_measure(settings):
    """Missing metadata means the original sum, or the later decay specification."""
    return settings.get("stock_measure", "decaying" if "stock_retention" in settings else "cumulative")


def update_history(stock, share, previous_occasions, weeks=1,
                   measure=DEFAULT_STOCK_MEASURE, retention=DEFAULT_STOCK_RETENTION):
    """Advance a pre-choice history; average counts observed purchase occasions only.

    Zero inside shares leave the average unchanged. Calendar gaps do not enter
    its denominator. Supports basket-by-flavor arrays in pricing.
    """
    if measure == "decaying":
        return advance_stock(stock, share, weeks, retention)
    if measure == "cumulative":
        return np.asarray(stock) + np.asarray(share)
    if measure != "average":
        raise ValueError("Unknown stock measure: " + str(measure))
    n = np.asarray(previous_occasions)
    if not np.isfinite(n).all() or (n < 0).any() or (n != np.floor(n)).any():
        raise ValueError("Previous purchase occasions must be nonnegative integers.")
    share = np.asarray(share)
    purchased = share.sum(axis=-1, keepdims=True) > 0
    denominator = n + purchased
    numerator = np.asarray(stock)*n + share
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator > 0)
