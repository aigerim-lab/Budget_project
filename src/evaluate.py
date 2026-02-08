from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from .utils import MODELS_DIR, FIG_DIR, ensure_dirs

def print_metrics() -> None:
    ensure_dirs()
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json not found. Train first: python3 main.py train")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rep = data["classification_report"]

    # show main numbers
    acc = rep.get("accuracy", None)
    f1_risk = rep.get("1", {}).get("f1-score", None)
    f1_ok = rep.get("0", {}).get("f1-score", None)

    print("Accuracy:", acc)
    print("F1 (RISK=1):", f1_risk)
    print("F1 (OK=0):", f1_ok)

    cm_path = FIG_DIR / "confusion_matrix.npy"
    if cm_path.exists():
        cm = np.load(cm_path)
        print("\nConfusion Matrix:\n", cm)
