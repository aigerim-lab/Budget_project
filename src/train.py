from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from .model import build_risk_model
from .utils import MODELS_DIR, FIG_DIR, ensure_dirs

FEATURES = ["total_spent", "txn_count", "avg_txn", "max_txn", "spending_velocity"]

def train_nn(df: pd.DataFrame, epochs: int = 10, batch_size: int = 512, seed: int = 42):
    ensure_dirs()

    X = df[FEATURES].fillna(0.0).astype(float).values
    y = df["risk"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_risk_model(input_dim=X_train_s.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    probs = model.predict(X_test_s).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    # Save artifacts
    model_path = MODELS_DIR / "risk_model.keras"
    scaler_path = MODELS_DIR / "scaler.pkl"
    schema_path = MODELS_DIR / "feature_schema.json"
    report_path = MODELS_DIR / "metrics.json"
    cm_path = FIG_DIR / "confusion_matrix.npy"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES}, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"classification_report": report}, f, indent=2)

    np.save(cm_path, cm)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "schema_path": str(schema_path),
        "report_path": str(report_path),
        "cm_path": str(cm_path),
        "history": history.history,
    }
