from __future__ import annotations
import pandas as pd
import numpy as np

def compute_category_shares(personal_df: pd.DataFrame) -> pd.Series:
    """
    Expects columns after normalization:
      - amount
      - category
    Your file shows: Amount, Category -> we lowercase them in loader.
    """
    df = personal_df.copy()
    if "amount" not in df.columns or "category" not in df.columns:
        raise ValueError("Personal expenses CSV must have 'Amount' and 'Category' columns.")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount", "category"])
    s = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    shares = s / s.sum()
    return shares

def recommend_allocation(monthly_budget: float, shares: pd.Series, top_k: int = 8) -> pd.DataFrame:
    s = shares.head(top_k)
    s = s / s.sum()
    rec = (s * float(monthly_budget)).round(2)
    out = pd.DataFrame({"category": rec.index, "recommended_budget": rec.values})
    return out
