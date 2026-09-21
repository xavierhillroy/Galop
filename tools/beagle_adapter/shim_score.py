#!/usr/bin/env python3
"""Production scorer for Beagle quality V3: test MSE of one genome computed
from BEAGLE'S OWN VM (BatchExecuteGenome shim -> stock BeagleLib CodeMachine,
float32 predictions; MSE aggregated in double). Replaces gcl_eval in the
quality pipeline so the scorer's semantic fidelity is by construction, not
by validation. Conventions match gcl_eval.mse: any non-finite prediction on
the test set makes the MSE inf (disclosed); a genome loading beyond the
dataset arity scores nan (should not occur from real evolution -- loud).
Usage: shim_score.py "<gcl>" test.csv BatchExecuteGenome.dll   -> prints MSE
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_gcl_eval import SkipArity, their_predictions


def main():
    gcl, test_csv, eg_dll = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [l.strip().split(",") for l in open(test_csv)][1:]
    X = [[float(v) for v in r[:-1]] for r in rows]
    y = [float(r[-1]) for r in rows]
    try:
        _, preds = their_predictions(gcl, X, eg_dll, len(X[0]), bulk=True)
    except SkipArity as e:
        print("nan")
        print(f"SKIP: genome loads {e.n} inputs, arity {len(X[0])}", file=sys.stderr)
        return
    if any(not math.isfinite(p) for p in preds):
        print("inf")
        return
    print(f"{sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y):.6e}")


if __name__ == "__main__":
    main()
