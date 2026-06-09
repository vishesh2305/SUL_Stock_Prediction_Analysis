"""Pre-trains all 5 primary stocks for both 1-day and 5-day horizons.

Run once before deploying so the dashboard starts instantly.
    python -m dashboard.train_artifacts
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a plain script too: `python train_artifacts.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data_loader import PRIMARY_STOCKS
from dashboard.utils.models import train_for, save_bundle


def main() -> None:
    for symbol in PRIMARY_STOCKS:
        for horizon in (1, 5):
            start = time.perf_counter()
            print(f"Training {symbol} (horizon={horizon}d) ...", flush=True)
            bundle = train_for(symbol, horizon=horizon)
            save_bundle(bundle)
            print(
                f"  done in {time.perf_counter() - start:.1f}s  "
                f"(train={bundle.train_size}, test={bundle.test_size})",
                flush=True,
            )


if __name__ == "__main__":
    main()
