#!/usr/bin/env python3
"""
gen_big_datasets.py — datasets for the giant-population campaign (§15).

  california  : sklearn fetch_california_housing (Pace & Barry 1997; 20,640 x 8).
                80/20 split (fixed seed), features AND target z-standardized on
                TRAIN statistics only. 8 features -> needs NUM_REGISTERS_OVERRIDE=16.
                This mirrors the EvoGP paper's benchmark protocol in spirit; their
                published curves were measured on different hardware, so nothing
                is overlaid -- both systems are re-measured here.
  feynman_i_9_18 : F = G*m1*m2 / ((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2), 9 variables
                (AI-Feynman I.9.18). Sampled U[1,2] for G,m1,m2,x1,y1,z1 and
                U[3,4] for x2,y2,z2 (denominator bounded away from 0). Exactly
                solvable -> solve-rate-vs-P story. Kept RAW (no standardization:
                the exact law must stay expressible; ranges are already O(1)).

Deterministic (fixed seeds). Run from repo root: python tools/gen_big_datasets.py   (needs scikit-learn)
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets", "big")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260709)


def write_csv(path, X, y, feat_names):
    with open(path, "w") as f:
        f.write(",".join(feat_names) + ",target\n")
        for row, t in zip(X, y):
            f.write(",".join(f"{v:.7g}" for v in row) + f",{t:.7g}\n")
    print(f"  {os.path.relpath(path, ROOT)}: {X.shape[0]} rows x {X.shape[1]} features")


def standardize(train, test):
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    X, y = d.data.astype(np.float64), d.target.astype(np.float64)
    idx = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, Xte = standardize(X[tr], X[te])
    ytr, yte = standardize(y[tr].reshape(-1, 1), y[te].reshape(-1, 1))
    write_csv(os.path.join(OUT, "california_train.csv"), Xtr, ytr.ravel(), d.feature_names)
    write_csv(os.path.join(OUT, "california_test.csv"), Xte, yte.ravel(), d.feature_names)


def feynman():
    names = ["G", "m1", "m2", "x1", "x2", "y1", "y2", "z1", "z2"]
    def sample(n):
        cols = {}
        for v in ["G", "m1", "m2", "x1", "y1", "z1"]:
            cols[v] = rng.uniform(1, 2, n)
        for v in ["x2", "y2", "z2"]:
            cols[v] = rng.uniform(3, 4, n)
        X = np.column_stack([cols[v] for v in names])
        F = (cols["G"] * cols["m1"] * cols["m2"] /
             ((cols["x2"] - cols["x1"])**2 + (cols["y2"] - cols["y1"])**2 + (cols["z2"] - cols["z1"])**2))
        return X, F
    Xtr, ytr = sample(1024)
    Xte, yte = sample(2048)
    write_csv(os.path.join(OUT, "feynman_i_9_18_train.csv"), Xtr, ytr, names)
    write_csv(os.path.join(OUT, "feynman_i_9_18_test.csv"), Xte, yte, names)


if __name__ == "__main__":
    print("generating datasets/big/ ...")
    california()
    feynman()
    print("done")
