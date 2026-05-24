"""
FeatureStore
============
Manages computation, caching (Parquet), versioning and retrieval of
feature matrices for multiple symbols and timeframes.

Storage layout on disk
----------------------
<root>/
  <version>/
    <symbol>_<timeframe>.parquet
    manifest.json              ← registry of all stored features

Usage
-----
    store = FeatureStore(root="data/storage/features")

    # Compute and persist
    store.build(df, symbol="EURUSD", timeframe="H1")

    # Load cached
    df_features = store.load(symbol="EURUSD", timeframe="H1")

    # List available
    store.list_features()
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research.feature_store.feature_engineer import FeatureConfig, FeatureEngineer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------

class FeatureManifest:
    """Tracks every stored feature set: symbol, timeframe, version, checksum."""

    def __init__(self, path: Path) -> None:
        self._path: Path = path
        self._entries: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        symbol: str,
        timeframe: str,
        version: str,
        n_rows: int,
        n_cols: int,
        checksum: str,
        config: dict,
    ) -> None:
        self._entries[key] = {
            "symbol":    symbol,
            "timeframe": timeframe,
            "version":   version,
            "n_rows":    n_rows,
            "n_cols":    n_cols,
            "checksum":  checksum,
            "config":    config,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def get(self, key: str) -> Optional[dict]:
        return self._entries.get(key)

    def all_entries(self) -> list[dict]:
        return [{"key": k, **v} for k, v in self._entries.items()]

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)
        self._save()

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                self._entries = json.load(fh)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2)


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Persistent, versioned feature store backed by Parquet files.

    Parameters
    ----------
    root : str | Path
        Root directory for all stored features.
    version : str
        Active version label (e.g. "v1", "v2"). All build / load calls
        use this version unless overridden.
    config : FeatureConfig | None
        Indicator configuration forwarded to FeatureEngineer.
    """

    def __init__(
        self,
        root: str | Path = "data/storage/features",
        version: str = "v1",
        config: FeatureConfig | None = None,
    ) -> None:
        self.root     = Path(root)
        self.version  = version
        self.engineer = FeatureEngineer(config or FeatureConfig())
        self.manifest = FeatureManifest(self.root / "manifest.json")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def build(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Compute features from raw OHLCV data and persist to Parquet.

        Parameters
        ----------
        data : pd.DataFrame
            Raw OHLCV data.
        symbol : str
            Asset identifier, e.g. "EURUSD".
        timeframe : str
            Bar interval, e.g. "M1", "H1", "D1".
        version : str | None
            Override active version for this call.
        overwrite : bool
            If False, returns cached version when it already exists.

        Returns
        -------
        pd.DataFrame  Feature matrix.
        """
        ver  = version or self.version
        key  = self._key(symbol, timeframe, ver)
        path = self._parquet_path(symbol, timeframe, ver)

        if not overwrite and path.exists():
            logger.info("Cache hit: %s — loading from disk.", key)
            return self.load(symbol, timeframe, ver)

        logger.info("Computing features for %s …", key)
        features = self.engineer.compute(data)
        checksum = self._checksum(features)

        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(path, index=True, engine="pyarrow", compression="snappy")

        self.manifest.register(
            key=key,
            symbol=symbol,
            timeframe=timeframe,
            version=ver,
            n_rows=len(features),
            n_cols=features.shape[1],
            checksum=checksum,
            config=self.engineer.config.__dict__,
        )
        logger.info("Stored %d rows × %d cols → %s", len(features), features.shape[1], path)
        return features

    def load(
        self,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None,
        columns: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load cached features from Parquet.

        Parameters
        ----------
        columns : list[str] | None
            Subset of columns to load (faster for large files).
        start / end : str | None
            ISO date strings for index filtering, e.g. "2023-01-01".
        """
        ver  = version or self.version
        path = self._parquet_path(symbol, timeframe, ver)

        if not path.exists():
            raise FileNotFoundError(
                f"No cached features for {symbol}/{timeframe}/{ver}. "
                "Run build() first."
            )

        df = pd.read_parquet(path, columns=columns, engine="pyarrow")

        if start or end:
            df = df.loc[
                (df.index >= start if start else True) &
                (df.index <= end   if end   else True)
            ]

        return df

    def delete(self, symbol: str, timeframe: str, version: Optional[str] = None) -> None:
        """Remove a cached feature set from disk and manifest."""
        ver  = version or self.version
        key  = self._key(symbol, timeframe, ver)
        path = self._parquet_path(symbol, timeframe, ver)

        if path.exists():
            path.unlink()
            logger.info("Deleted %s", path)

        self.manifest.remove(key)

    # ------------------------------------------------------------------
    # Discovery & inspection
    # ------------------------------------------------------------------

    def list_features(self) -> pd.DataFrame:
        """Return a summary DataFrame of all stored feature sets."""
        entries = self.manifest.all_entries()
        if not entries:
            return pd.DataFrame(columns=["key", "symbol", "timeframe", "version",
                                         "n_rows", "n_cols", "created_at"])
        return pd.DataFrame(entries).set_index("key")

    def exists(self, symbol: str, timeframe: str, version: Optional[str] = None) -> bool:
        path = self._parquet_path(symbol, timeframe, version or self.version)
        return path.exists()

    def get_feature_names(self, symbol: str, timeframe: str, version: Optional[str] = None) -> list[str]:
        """Return column names without loading the full file."""
        path = self._parquet_path(symbol, timeframe, version or self.version)
        if not path.exists():
            raise FileNotFoundError(f"No cache found for {symbol}/{timeframe}.")
        return pd.read_parquet(path, engine="pyarrow").columns.tolist()

    def get_matrix(
        self,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None,
        exclude: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Load features and return a raw float32 numpy array (ML-ready).

        OHLCV columns are excluded by default unless explicitly included.
        """
        exclude = exclude or ["open", "high", "low", "close", "volume"]
        df = self.load(symbol, timeframe, version)
        df = df.drop(columns=[c for c in exclude if c in df.columns])
        return df.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Multi-symbol batch
    # ------------------------------------------------------------------

    def build_batch(
        self,
        datasets: dict[tuple[str, str], pd.DataFrame],
        version: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """
        Build features for multiple (symbol, timeframe) pairs at once.

        Parameters
        ----------
        datasets : dict
            {(symbol, timeframe): ohlcv_df}

        Returns
        -------
        dict  {(symbol, timeframe): feature_df}
        """
        results: dict[tuple[str, str], pd.DataFrame] = {}
        for (symbol, tf), df in datasets.items():
            results[(symbol, tf)] = self.build(df, symbol, tf, version, overwrite)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parquet_path(self, symbol: str, timeframe: str, version: str) -> Path:
        return self.root / version / f"{symbol}_{timeframe}.parquet"

    @staticmethod
    def _key(symbol: str, timeframe: str, version: str) -> str:
        return f"{symbol}/{timeframe}/{version}"

    @staticmethod
    def _checksum(df: pd.DataFrame) -> str:
        """MD5 of the raw values for integrity checking."""
        raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.md5(raw).hexdigest()  # noqa: S324 — non-cryptographic use
