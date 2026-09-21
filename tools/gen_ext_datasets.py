#!/usr/bin/env python3
"""Nguyen 9/10/12 datasets — the multi-variable Nguyen problems expressible in
the five-system shared function set {+,-,*,/,sin,cos,<,>}. (N7, N8, N11 need
ln, sqrt, pow — absent from the parity set — and stay excluded by rule.)
Standard domains per Uy et al.: x1,x2 ~ U[0,1]. train=100 / test=200, the same
sizes as the N1-6 sweep sets. Version-stable CPython MT, seeds 1234 (train) /
5678 (test); md5s printed (reference values in REPRODUCE.md).
Descriptive tier only — not part of the inferential families. Idempotent: existing files are left untouched (md5 still printed)."""
import random, os, math, hashlib

FN = {
    "nguyen9":  lambda a, b: math.sin(a) + math.sin(b * b),
    "nguyen10": lambda a, b: 2.0 * math.sin(a) * math.cos(b),
    "nguyen12": lambda a, b: a**4 - a**3 + (b * b) / 2.0 - b,
}
os.makedirs("datasets/ext", exist_ok=True)
for name, f in FN.items():
    for split, n, seed in (("train", 100, 1234), ("test", 200, 5678)):
        p = f"datasets/ext/{name}_{split}.csv"
        if not os.path.exists(p):
            rng = random.Random(seed)
            rows = ["x1,x2,y"]
            for _ in range(n):
                # U(0,1) per the benchmark definition (Uy et al. 2011): the
                # bivariate Nguyen problems are published on [0,1]^2, unlike
                # the univariate 1-6 on [-1,1]; domains are per-problem by suite design
                a, b = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
                rows.append(f"{a!r},{b!r},{f(a, b)!r}")
            open(p, "w").write("\n".join(rows) + "\n")
        print(name, split, hashlib.md5(open(p, "rb").read()).hexdigest())
