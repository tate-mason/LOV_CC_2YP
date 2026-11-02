"""Separate common novelty and own-stock coefficients; random alpha. See README.md."""
from est import main

if __name__ == "__main__":
    main("separate_history", default_no_price_control=False)
