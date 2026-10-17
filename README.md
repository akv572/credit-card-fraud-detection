# Credit Card Fraud Detection

Machine learning pipeline for detecting fraudulent card transactions in highly
imbalanced data. Trains and compares logistic regression, random forest and
k-nearest neighbours classifiers, with class imbalance handled explicitly and
decision thresholds tuned by cross validation.

## Background

This repository is a clean room reimplementation of my graduate project
**Securing Transactions**, which processed 280,000+ real transactions with
logistic regression, random forest and KNN, reaching 92% precision through
class weight balancing on the imbalanced data. The code here rebuilds that
approach from scratch on a synthetic dataset of the same shape, so the full
pipeline runs without shipping a large data file. Swap in a real CSV with
`--csv` to run it against production shaped data.

## Approach

* **Imbalance handling.** Fraud is ~0.17% of transactions. Logistic regression
  and random forest use balanced class weights; KNN trains on a downsampled
  set (every fraud case plus a 10:1 majority sample) because neighbourhood
  methods collapse at 1:600 imbalance. The test set is never resampled.
* **Threshold tuning.** At this imbalance the default 0.5 decision threshold
  is meaningless, so each model's threshold is tuned to maximise F1 via 3-fold
  cross validation on the training data only.
* **No leakage.** Scaling is fit on the training fold; the test set is held
  out until final evaluation.

## Results

Synthetic data, 40,000 transactions, seed 42. Metrics on the held out test set:

| Model | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| Random forest | 0.733 | 0.786 | 0.759 | 0.961 |
| Logistic regression | 0.647 | 0.786 | 0.710 | 0.982 |
| KNN | 0.117 | 0.857 | 0.205 | 0.998 |

Random forest gives the best precision/recall trade off. KNN ranks well
(ROC AUC 0.998) but its precision collapses, a useful reminder that ranking
metrics alone mislead on imbalanced problems.

## Project structure

```
├── main.py            # end to end pipeline (CLI)
├── src/
│   ├── data.py        # synthetic data generator + CSV loader
│   ├── features.py    # stratified split, scaling without leakage
│   ├── models.py      # the three classifiers, KNN resampling, threshold tuning
│   └── evaluate.py    # metrics, comparison table, PR curves
├── reports/           # generated: metrics.json, pr_curves.png
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py                        # 40k synthetic transactions
python main.py --samples 100000       # larger run
python main.py --csv data/transactions.csv   # your own data (same schema)
```

## Requirements

Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
