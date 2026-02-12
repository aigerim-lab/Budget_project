import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.allocation import recommend_allocation

st.set_page_config(page_title="My Money Mentor", layout="wide")
st.title("My Money Mentor — Budget Risk Alerts + Budget Allocation")

MODELS = ROOT / "models"

# --- Load model artifacts (safe mode: compile=False) ---
model_path = MODELS / "risk_model.keras"
scaler_path = MODELS / "scaler.pkl"
schema_path = MODELS / "feature_schema.json"

if not model_path.exists():
    st.error("Model not found. Train first: python3 main.py train")
    st.stop()

# IMPORTANT: compile=False reduces TF issues when loading
model = tf.keras.models.load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
FEATURES = schema["features"]

# ---------------- Sidebar: Budget input ----------------
st.sidebar.header("Inputs")
monthly_budget = st.sidebar.number_input("Monthly budget", min_value=0.0, value=120000.0, step=1000.0)

st.subheader("1) Budget Risk Alert (manual input)")
st.write("Enter your estimated monthly spending by category. The app converts it into behavioral features and predicts **RISK / OK**.")

# Default categories for demo (normal & meaningful)
categories = ["Food", "Transport", "Rent", "Clothes", "Education", "Entertainment", "Health", "Other"]

cols = st.columns(2)
spend = {}

for i, cat in enumerate(categories):
    with cols[i % 2]:
        spend[cat] = st.number_input(f"{cat} spending", min_value=0.0, value=0.0, step=100.0)

total_spent = float(sum(spend.values()))
txn_count = st.number_input("Estimated number of transactions (per month)", min_value=1, value=60, step=1)

# derived features
avg_txn = total_spent / float(txn_count)
max_txn = st.number_input("Estimated maximum single transaction", min_value=0.0, value=min(20000.0, total_spent), step=100.0)

# For spending_velocity in our training, we can set it to 0 (unknown for one input)
spending_velocity = 0.0

# Build feature vector in correct order
feature_map = {
    "total_spent": total_spent,
    "txn_count": float(txn_count),
    "avg_txn": float(avg_txn),
    "max_txn": float(max_txn),
    "spending_velocity": float(spending_velocity),
}

X = np.array([[feature_map[f] for f in FEATURES]], dtype=float)
Xs = scaler.transform(X)

if st.button("Predict Risk Alert"):
    prob = float(model.predict(Xs, verbose=0).reshape(-1)[0])
    label = "RISK" if prob >= 0.5 else "OK"

    st.metric("Prediction", label)
    st.metric("Risk probability", f"{prob:.3f}")

    # simple budget rule message
    ratio = total_spent / monthly_budget if monthly_budget > 0 else 0
    if ratio >= 1.0:
        st.error("You are above your monthly budget. High risk of overspending.")
    elif ratio >= 0.8:
        st.warning("You are close to your budget limit (≥80%). Consider reducing optional spending.")
    else:
        st.success("Your spending is within a safe range based on the provided budget.")

st.divider()

# ---------------- Allocation plan ----------------
st.subheader("2) Budget Allocation Plan (recommended split)")
st.write("Based on your entered spending distribution, the app generates a recommended budget allocation for your monthly budget.")

# Convert manual input to shares
spend_series = pd.Series(spend)
if spend_series.sum() > 0:
    shares = (spend_series / spend_series.sum()).sort_values(ascending=False)
else:
    # If user entered nothing, use a reasonable default split
    shares = pd.Series({
        "Rent": 0.35,
        "Food": 0.20,
        "Transport": 0.10,
        "Education": 0.10,
        "Health": 0.05,
        "Clothes": 0.05,
        "Entertainment": 0.10,
        "Other": 0.05,
    }).sort_values(ascending=False)

top_k = st.slider("Top categories to show", min_value=3, max_value=8, value=8)

alloc = recommend_allocation(monthly_budget, shares, top_k=top_k)
st.dataframe(alloc, use_container_width=True)
st.bar_chart(alloc.set_index("category")["recommended_budget"])

# ---------------- Pie chart for budget allocation ----------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(alloc["recommended_budget"], labels=alloc["category"], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
st.pyplot(fig)