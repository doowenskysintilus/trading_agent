"""
Training dataset loader
=======================
Reads the compact per-trade experiences written by
``live_trading.experience_collector.ExperienceCollector`` and turns them
into a flat, model-ready :class:`pandas.DataFrame`.

Each experience is one closed trade: the entry observation (features) plus
its realized outcome. This module expands the nested ``features`` dict into
columns and adds a binary ``label`` (1 = winning trade, 0 = losing/flat),
ready for the supervised win/loss classifier or for reward-shaping analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_experiences(dataset_dir: str = "data/storage/datasets") -> pd.DataFrame:
    """Load all ``experiences_*.jsonl`` files into a single DataFrame.

    Returns an empty DataFrame if no dataset files are present.
    """
    root = Path(dataset_dir)
    if not root.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for path in sorted(root.glob("experiences_*.jsonl")):
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception as exc:
            logger.warning("Failed reading %s: %s", path, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def build_supervised_dataset(
    dataset_dir: str = "data/storage/datasets",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y, feature_names) for the win/loss classifier.

    X : DataFrame of entry features (one row per trade).
    y : Series, 1 if the trade was a win (pnl > 0), else 0.
    feature_names : ordered list of feature columns used.
    """
    df = load_experiences(dataset_dir)
    if df.empty or "features" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype="int64"), []

    # Expand the nested feature dict into columns.
    feats = pd.json_normalize(df["features"]).fillna(0.0)
    feature_names = list(feats.columns)
    if not feature_names:
        return pd.DataFrame(), pd.Series(dtype="int64"), []

    # Label from realized outcome (prefer numeric pnl when available).
    if "pnl" in df.columns:
        y = (pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        y = (df.get("outcome", "") == "WIN").astype(int)

    return feats.reset_index(drop=True), y.reset_index(drop=True), feature_names
