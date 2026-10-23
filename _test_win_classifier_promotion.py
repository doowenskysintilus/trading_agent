#!/usr/bin/env python3
"""
Test script for the WinClassifier promotion gate (Phase C of the
"make quant-fund-ai more reliable" work).

Before this fix, RetrainService._retrain_ml() -> WinClassifier.train()
always overwrote the deployed model file unconditionally, even when the
freshly fitted candidate was worse than what was already live. This
verifies:
  1. First-ever training promotes (nothing deployed yet to beat).
  2. A clearly WORSE candidate (near-random labels) is rejected — the
     deployed (better) model stays on disk and the report says so.
  3. A clearly BETTER candidate is promoted over what's deployed.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from strategies.ml_agent.win_classifier import WinClassifier

PASS = "✓"
FAIL = "✗"

rng = np.random.default_rng(42)


def make_separable_dataset(n=400, n_features=6, signal_strength=3.0):
    """Labels strongly correlated with feature 0 -> high AUC when fit."""
    X = rng.normal(size=(n, n_features))
    logits = signal_strength * X[:, 0] + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    names = [f"f{i}" for i in range(n_features)]
    return X, y, names


def make_corrupted_train_dataset(n=400, n_features=6, signal_strength=3.0, corrupt_frac=0.5):
    """Same true signal as make_separable_dataset, but ~50% of the labels
    in the chronological TRAINING portion (first 75%, matching
    WinClassifier.train()'s internal split) are flipped — simulating a bad
    batch of mislabeled trade outcomes. The held-out TEST portion (last
    25%) keeps the true relationship intact, so it's a fair, uncorrupted
    yardstick: a model fit on the corrupted training slice should
    generalize worse to it than a model that learned the clean signal.
    """
    X = rng.normal(size=(n, n_features))
    logits = signal_strength * X[:, 0] + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    split = int(n * 0.75)
    n_corrupt = int(split * corrupt_frac)
    corrupt_idx = rng.choice(split, size=n_corrupt, replace=False)
    y[corrupt_idx] = 1 - y[corrupt_idx]
    names = [f"f{i}" for i in range(n_features)]
    return X, y, names


tmpdir = Path(tempfile.mkdtemp(prefix="wc_promotion_test_"))
model_path = tmpdir / "win_classifier.joblib"
all_ok = True

try:
    print("=" * 70)
    print("TEST 1: first-ever training promotes (nothing deployed yet)")
    print("=" * 70)
    clf = WinClassifier(model_path=str(model_path))
    X1, y1, names1 = make_separable_dataset()
    report1 = clf.train(X1, y1, names1)
    ok = report1.trained and report1.promoted and model_path.exists()
    print(f"  {PASS if ok else FAIL} trained={report1.trained} promoted={report1.promoted} "
          f"auc={report1.auc:.3f} file_exists={model_path.exists()}")
    all_ok &= ok
    baseline_auc = report1.auc

    print("\n" + "=" * 70)
    print("TEST 2: worse candidate (corrupted training labels) is REJECTED")
    print("=" * 70)
    clf2 = WinClassifier(model_path=str(model_path))
    X2, y2, names2 = make_corrupted_train_dataset()
    report2 = clf2.train(X2, y2, names2)
    ok = report2.trained and not report2.promoted and report2.deployed_auc is not None
    print(f"  {PASS if ok else FAIL} trained={report2.trained} promoted={report2.promoted} "
          f"candidate_auc={report2.auc:.3f} deployed_auc={report2.deployed_auc}")
    print(f"    message: {report2.message}")
    all_ok &= ok

    # The deployed file must be untouched — reload it and confirm it still
    # performs like the original (separable) model, not the noisy one.
    verify = WinClassifier(model_path=str(model_path))
    verify.load()
    proba = verify._model.predict_proba(X1)[:, 1]
    from sklearn.metrics import roc_auc_score
    reloaded_auc_on_original_data = roc_auc_score(y1, proba)
    ok2 = reloaded_auc_on_original_data > 0.8  # still the good model
    print(f"  {PASS if ok2 else FAIL} deployed model on disk is still the ORIGINAL "
          f"(good) one — auc on original data = {reloaded_auc_on_original_data:.3f}")
    all_ok &= ok2

    print("\n" + "=" * 70)
    print("TEST 3: clearly better candidate IS promoted")
    print("=" * 70)
    clf3 = WinClassifier(model_path=str(model_path))
    X3, y3, names3 = make_separable_dataset(n=800, signal_strength=6.0)  # cleaner signal
    report3 = clf3.train(X3, y3, names3)
    ok = report3.trained and report3.promoted
    print(f"  {PASS if ok else FAIL} trained={report3.trained} promoted={report3.promoted} "
          f"candidate_auc={report3.auc:.3f} deployed_auc={report3.deployed_auc}")
    all_ok &= ok

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All promotion-gate checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
