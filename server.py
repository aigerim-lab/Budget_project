import json
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model(MODELS / "risk_model.keras", compile=False)
scaler = joblib.load(MODELS / "scaler.pkl")
schema = json.loads((MODELS / "feature_schema.json").read_text(encoding="utf-8"))
FEATURES = schema["features"]

DEFAULT_SHARES = {
    "Rent": 0.35, "Food": 0.20, "Transport": 0.10, "Education": 0.10,
    "Health": 0.05, "Clothes": 0.05, "Entertainment": 0.10, "Other": 0.05
}

def recommend_allocation(monthly_budget, shares, top_k=8):
    items = sorted(shares.items(), key=lambda x: x[1], reverse=True)[:top_k]
    ssum = sum(v for _, v in items) or 1.0
    rows = []
    for k, v in items:
        rows.append({"category": k, "recommended_budget": round(monthly_budget * (v/ssum), 2)})
    return rows

@app.post("/predict")
def predict():
    data = request.get_json(force=True)

    monthly_budget = float(data.get("monthly_budget", 0))
    txn_count = float(data.get("txn_count", 60))
    max_txn = float(data.get("max_txn", 0))
    category_spend = data.get("category_spend", {})

    total_spent = float(sum(float(v) for v in category_spend.values()))
    avg_txn = total_spent / max(txn_count, 1.0)
    spending_velocity = 0.0

    feature_map = {
        "total_spent": total_spent,
        "txn_count": txn_count,
        "avg_txn": avg_txn,
        "max_txn": max_txn,
        "spending_velocity": spending_velocity
    }

    X = np.array([[feature_map[f] for f in FEATURES]], dtype=float)
    Xs = scaler.transform(X)
    prob = float(model.predict(Xs, verbose=0).reshape(-1)[0])
    prediction = "RISK" if prob >= 0.5 else "OK"

    budget_ratio = (total_spent / monthly_budget) if monthly_budget > 0 else 0.0

    # allocation shares from input if provided, else defaults
    if total_spent > 0:
        shares = {k: float(v) / total_spent for k, v in category_spend.items()}
    else:
        shares = DEFAULT_SHARES

    allocation = recommend_allocation(monthly_budget, shares, top_k=8)

    return jsonify({
        "prediction": prediction,
        "risk_prob": prob,
        "total_spent": total_spent,
        "budget_ratio": budget_ratio,
        "allocation": allocation
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
