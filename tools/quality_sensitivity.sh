#!/usr/bin/env bash
# =============================================================================
# Sensitivity arm (Sec. 5.2): EvoGP under its own published example configuration
# — max_tree_len 512, generation depth 9, layer_leaf_prob 0.3, tournament 20,
# survivor_rate 0.5, mutation 0.1 with depth-4 subtrees — with primitives,
# constants and datasets kept matched. 100 seeded runs at P=1e4 on all eight
# primary cells (Nguyen 1-6, California, Feynman); its own Holm family.
#   bash tools/quality_sensitivity.sh    (PYTHON=, QUALITY_NSEEDS= optional)
# -> results/quality_sensitivity_rtx.csv   problem,P,seed,train_mse,test_mse
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
QUALITY_NSEEDS=${QUALITY_NSEEDS:-100} ${PYTHON:-.venv/bin/python} - <<'PYEOF'
import torch, sys, csv, os
sys.path.insert(0, "harness")
from evogp_run import load_csv, mse
from evogp.tree import Forest, GenerateDescriptor
from evogp.problem import SymbolicRegression
from evogp.algorithm import GeneticProgramming, DefaultCrossover, DefaultMutation, TournamentSelection
from evogp.pipeline import StandardPipeline
PATH = "results/quality_sensitivity_rtx.csv"; N = int(os.environ["QUALITY_NSEEDS"]); P = 10000
os.makedirs("results", exist_ok=True)
done = set()
if os.path.exists(PATH):
    for r in csv.DictReader(open(PATH)): done.add((r["problem"], r["seed"]))
new = not os.path.exists(PATH)
out = open(PATH, "a", newline=""); w = csv.writer(out)
if new: w.writerow(["problem", "P", "seed", "train_mse", "test_mse"])
FUNCS = ["+", "-", "*", "/", "sin", "cos", "<", ">"]
NGC = [0., 1., -1., 3.14159, 0.5, 2., 10., 100.]
BIGC = NGC + [0.1, -0.5, 0.25, 5., 20., -2., 0.01, 1000.]
jobs  = [(t, 1, NGC, f"datasets/sweep/{t}") for t in ("nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6")]
jobs += [(p, nf, BIGC, f"datasets/big/{p}") for p, nf in (("feynman_i_9_18", 9), ("california", 8))]
for prob, nf, consts, base in jobs:
    desc = GenerateDescriptor(max_tree_len=512, input_len=nf, output_len=1, using_funcs=FUNCS,
                              max_layer_cnt=9, layer_leaf_prob=0.3, const_samples=consts)
    Xtr, ytr = load_csv(base + "_train.csv"); Xte, yte = load_csv(base + "_test.csv")
    problem = SymbolicRegression(datapoints=Xtr, labels=ytr, execute_mode="auto")
    for s in range(N):
        if (prob, str(s)) in done: continue
        torch.manual_seed(s)
        alg = GeneticProgramming(initial_forest=Forest.random_generate(pop_size=P, descriptor=desc),
                                 crossover=DefaultCrossover(),
                                 mutation=DefaultMutation(mutation_rate=0.1, descriptor=desc.update(max_layer_cnt=4)),
                                 selection=TournamentSelection(tournament_size=20, best_probability=1.0, survivor_rate=0.5, elite_rate=0.1))
        best = StandardPipeline(alg, problem, generation_limit=100, is_show_details=False).run()
        torch.cuda.synchronize()
        w.writerow([prob, P, s, f"{mse(best, Xtr, ytr):.6e}", f"{mse(best, Xte, yte):.6e}"]); out.flush()
    print(f"  sensitivity {prob} done", flush=True)
print("SENSITIVITY ARM COMPLETE -> " + PATH, flush=True)
PYEOF
