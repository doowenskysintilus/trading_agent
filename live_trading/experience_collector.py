"""
ExperienceCollector
====================
Compact, training-ready dataset of *placed orders and their realized
outcomes* — and nothing else.

Rationale
---------
RL (PPO-LSTM) and any parallel ML model only learn from decisions that were
actually taken and their results. Logging every cycle (including the ~99%
that produce HOLD / no trade) bloats the dataset with redundant rows and
slows training for no benefit. This collector therefore writes **one record
per executed trade**, capturing:

  * the observation (feature snapshot) at the moment the order was placed,
  * the action taken (direction / size),
  * the realized result once the position closes (pnl, outcome, duration).

That is exactly the (state, action, reward) tuple needed for offline RL /
supervised learning, with no useless data.

Storage: newline-delimited JSON (one trade per line), rotated daily, under
``data/storage/datasets/``. Easy to stream into pandas for training.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperienceCollector:
    """Append-only writer for closed-trade training experiences."""

    def __init__(self, out_dir: str = "data/storage/datasets") -> None:
        self._dir = Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _file_for(self, ts: datetime) -> Path:
        return self._dir / f"experiences_{ts.strftime('%Y%m%d')}.jsonl"

    def record(self, experience: dict[str, Any]) -> None:
        """Append one fully-resolved trade experience as a JSON line."""
        ts = datetime.now(timezone.utc)
        path = self._file_for(ts)
        line = json.dumps(experience, default=_json_default, separators=(",", ":"))
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:   # never let logging break the trading loop
            logger.warning("ExperienceCollector write failed: %s", exc)


def _json_default(obj: Any):
    """Serialize numpy / datetime values that json can't handle natively."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    # numpy scalars expose .item()
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def extract_observation(features) -> dict[str, float]:
    """Flatten the most recent feature row into a compact dict of floats.

    This is the model's observation at decision time — the single most
    informative, training-relevant snapshot for the trade about to be placed.
    """
    try:
        last = features.iloc[-1]
        out: dict[str, float] = {}
        for col, val in last.items():
            try:
                out[str(col)] = float(val)
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return {}
