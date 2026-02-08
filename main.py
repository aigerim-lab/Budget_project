from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.utils import DATA_DIR, PROCESSED_DIR, ensure_dirs
from src.data_loader import load_paysim_csv, load_creditcard_csv, load_personal_expenses_csv
from src.preprocessing import paysim_to_windows, make_risk_label
from src.train import train_nn
from src.evaluate import print_metrics
from src.allocation import compute_category_shares, recommend_allocation

def find_file_by_prefix(folder: Path, prefix: str) -> Path:
    files = list(folder.glob(f"{prefix}*"))
    if not files:
        raise FileNotFoundError(f"No file starting with '{prefix}' in {folder}")
    return files[0]

def cmd_preprocess(args):
    ensure_dirs()

    if args.source == "paysim":
        # your file starts with PS_
        paysim_path = find_file_by_prefix(DATA_DIR, "PS_")
        df = load_paysim_csv(paysim_path)
        agg = paysim_to_windows(df, window=args.window)
        labeled = make_risk_label(agg, percentile=args.percentile)
        out = PROCESSED_DIR / "risk_features.csv"
        labeled.to_csv(out, index=False)
        print("Saved:", out)

    elif args.source == "creditcard":
        # your file: credit_card_transactions.csv
        cc_path = find_file_by_prefix(DATA_DIR, "credit_card_transactions")
        df = load_creditcard_csv(cc_path)
        # creditcard doesn't have nameOrig/step like PaySim.
        # For this project we keep it as a secondary dataset only for reporting,
        # so preprocessing step is skipped here.
        print("CreditCard dataset loaded:", df.shape)
        print("Note: Used as supporting heavy dataset (optional validation), not primary training source.")
    else:
        raise ValueError("source must be paysim or creditcard")

def cmd_train(args):
    ensure_dirs()
    features_path = PROCESSED_DIR / "risk_features.csv"
    if not features_path.exists():
        raise FileNotFoundError("Run preprocess first: python3 main.py preprocess --source paysim")

    df = pd.read_csv(features_path)
    info = train_nn(df, epochs=args.epochs, batch_size=args.batch_size)
    print("Training done. Saved artifacts:")
    for k, v in info.items():
        if k != "history":
            print("-", k, ":", v)

def cmd_evaluate(args):
    print_metrics()

def cmd_allocate(args):
    ensure_dirs()
    # your file: myExpenses1 2.csv
    exp_path = find_file_by_prefix(DATA_DIR, "myExpenses")
    exp = load_personal_expenses_csv(exp_path)
    shares = compute_category_shares(exp)
    alloc = recommend_allocation(args.budget, shares, top_k=args.top_k)
    print(alloc.to_string(index=False))

def build_parser():
    p = argparse.ArgumentParser(description="My Money Mentor - Budget Risk + Allocation")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("preprocess", help="Aggregate transactions and build supervised labels")
    p1.add_argument("--source", choices=["paysim", "creditcard"], default="paysim")
    p1.add_argument("--window", choices=["day", "week"], default="day")
    p1.add_argument("--percentile", type=float, default=90.0)
    p1.set_defaults(func=cmd_preprocess)

    p2 = sub.add_parser("train", help="Train TensorFlow NN risk model")
    p2.add_argument("--epochs", type=int, default=10)
    p2.add_argument("--batch-size", type=int, default=512)
    p2.set_defaults(func=cmd_train)

    p3 = sub.add_parser("evaluate", help="Print saved metrics and confusion matrix")
    p3.set_defaults(func=cmd_evaluate)

    p4 = sub.add_parser("allocate", help="Print budget allocation table from personal expenses")
    p4.add_argument("--budget", type=float, required=True)
    p4.add_argument("--top-k", type=int, default=8)
    p4.set_defaults(func=cmd_allocate)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
