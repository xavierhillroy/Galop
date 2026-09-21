#!/usr/bin/env bash
# =============================================================================
# EvoGP quality references (fig5, Sec. 5.2): the same cells and seeds as
# tools/quality_galop.sh under the parity configuration (101-node cap, the 8
# shared functions, fixed constant pool, tournament 3, survivor_rate 1.0 —
# the parity value: wall-neutral, modest quality effect, no verdict flips
# against 0.1/0.5 — elite 0.1, mutation 0.8, execute_mode auto). Multi-input
# problems use the 16-constant pool. EvoGP run values are not seed-stable
# across driver/torch versions, so references are generated in the same
# environment as the GALOP runs. Best-of-run train/test MSE on the shared files.
#   bash tools/quality_evogp.sh          (PYTHON=, QUALITY_NSEEDS= optional)
# -> results/quality_evogp_rtx.csv   problem,P,seed,train_mse,test_mse
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
python3 tools/gen_ext_datasets.py >/dev/null
QUALITY_NSEEDS=${QUALITY_NSEEDS:-100} ${PYTHON:-.venv/bin/python} - <<'PYEOF'
import torch, sys, csv, os
sys.path.insert(0, "harness")
from evogp_run import make_descriptor, make_algorithm, make_problem, load_csv, mse
from evogp.pipeline import StandardPipeline
PATH = "results/quality_evogp_rtx.csv"; N = int(os.environ["QUALITY_NSEEDS"])
os.makedirs("results", exist_ok=True)
done = set()
if os.path.exists(PATH):
    for r in csv.DictReader(open(PATH)): done.add((r["problem"], r["P"], r["seed"]))
new = not os.path.exists(PATH)
out = open(PATH, "a", newline=""); w = csv.writer(out)
if new: w.writerow(["problem", "P", "seed", "train_mse", "test_mse"])
FUNCS = ["+", "-", "*", "/", "sin", "cos", "<", ">"]
NG = {"max_tree_len": 101, "features": 1, "using_funcs": FUNCS,
      "const_samples": [0., 1., -1., 3.14159, 0.5, 2., 10., 100.],
      "tournament": 3, "survivor_rate": 1.0, "elite_rate": 0.1, "mutation_rate": 0.8, "execute_mode": "auto"}
BIG = dict(NG, const_samples=[0., 1., -1., 3.14159, 0.5, 2., 10., 100., 0.1, -0.5, 0.25, 5., 20., -2., 0.01, 1000.])
jobs  = [(t, NG, f"datasets/sweep/{t}", (1000, 10000, 100000)) for t in ("nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6")]
jobs += [(p, dict(BIG, features=nf), f"datasets/big/{p}", (10000, 100000)) for p, nf in (("feynman_i_9_18", 9), ("california", 8))]
jobs += [(t, dict(BIG, features=2), f"datasets/ext/{t}", (1000, 10000, 100000)) for t in ("nguyen9", "nguyen10", "nguyen12")]
for prob, cfg, base, Ps in jobs:
    desc = make_descriptor(cfg)
    Xtr, ytr = load_csv(base + "_train.csv"); Xte, yte = load_csv(base + "_test.csv")
    problem = make_problem(cfg, Xtr, ytr)
    for P in Ps:
        for s in range(N):
            if (prob, str(P), str(s)) in done: continue
            torch.manual_seed(s)
            best = StandardPipeline(make_algorithm(cfg, desc, P), problem, generation_limit=100, is_show_details=False).run()
            torch.cuda.synchronize()
            w.writerow([prob, P, s, f"{mse(best, Xtr, ytr):.6e}", f"{mse(best, Xte, yte):.6e}"]); out.flush()
        print(f"  evogp {prob} P={P} done", flush=True)
print("EVOGP QUALITY COMPLETE -> " + PATH, flush=True)
PYEOF
