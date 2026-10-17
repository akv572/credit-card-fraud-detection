"""End to end fraud detection pipeline.

Generates (or loads) transaction data, trains logistic regression, random
forest and KNN classifiers, then prints and saves a metric comparison.

Usage:
    python main.py [--samples N] [--csv PATH] [--seed S] [--out-dir DIR]
"""

from __future__ import annotations

import argparse

from src.data import load_transactions
from src.evaluate import print_comparison, save_reports, score_model
from src.features import split_and_scale
from src.models import train_all, tune_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit card fraud detection")
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a real transaction CSV (same schema)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="reports")
    args = parser.parse_args()

    print("Loading data...")
    data = load_transactions(args.csv, n_samples=args.samples, seed=args.seed)
    frauds = int(data["Class"].sum())
    print(f"  {len(data):,} transactions, {frauds:,} fraudulent "
          f"({frauds / len(data):.3%})")

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test, _ = split_and_scale(data, seed=args.seed)
    print(f"  train={X_train.shape[0]:,}, test={X_test.shape[0]:,}")

    print("Training models...")
    models = train_all(X_train, y_train, seed=args.seed)

    print("Tuning decision thresholds (3-fold CV on train)...")
    thresholds = {name: tune_threshold(m, X_train, y_train, seed=args.seed)
                  for name, m in models.items()}
    for name, t in thresholds.items():
        print(f"  {name:<22}threshold={t:.2f}")

    print("Evaluating on the held out test set...")
    results = {name: score_model(m, X_test, y_test, threshold=thresholds[name])
               for name, m in models.items()}
    for name, t in thresholds.items():
        results[name]["threshold"] = t
    print()
    print_comparison(results)

    save_reports(models, results, X_test, y_test, args.out_dir)
    print(f"\nReports written to {args.out_dir}/ (metrics.json, pr_curves.png)")


if __name__ == "__main__":
    main()
