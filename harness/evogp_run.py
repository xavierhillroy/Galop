#!/usr/bin/env python3
"""
evogp_run.py — run EvoGP under the SAME config/schema as GALOP, so its rows drop
straight into results/*/summary.csv (system=evogp) for the head-to-head.

    python3 harness/evogp_run.py harness/configs/evogp_eval_crossover.toml

Modes mirror run.py:
  eval       -> time problem.evaluate(forest) only (the eval-kernel comparison)
  whole_run  -> time StandardPipeline.run() (full loop, all on GPU)
  quality    -> full run on the 6 Nguyen, train+test MSE + per-gen fitness

Metric (matched to GALOP): GPops/s = G * P * S_bar * D / T  (raw ops/s),
with S_bar = mean tree node count (measured), D = data points.

NOTE: EvoGP API calls that need confirming against the installed version are
were checked against the installed EvoGP (commit 5538d75): function names, operator signatures, forest attributes.
"""
import sys, os, csv, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import SUMMARY_COLS, provenance, run_id, base_row, ROOT   # reuse GALOP's schema

import tomllib
import torch
from evogp.tree import Forest, GenerateDescriptor
from evogp.problem import SymbolicRegression
from evogp.algorithm import GeneticProgramming, DefaultCrossover, DefaultMutation, TournamentSelection
from evogp.pipeline import StandardPipeline

DEV = "cuda"

# ---- data: load the SAME generated CSVs GALOP uses (identical points) --------
def load_csv(path):
    xs, ys = [], []
    with open(os.path.join(ROOT, path)) as f:
        next(f)  # header
        for line in f:
            cols = [float(v) for v in line.strip().split(",")]
            xs.append(cols[:-1]); ys.append(cols[-1])
    X = torch.tensor(xs, device=DEV, dtype=torch.float)                  # (D, n_features)
    y = torch.tensor(ys, device=DEV, dtype=torch.float).reshape(-1, 1)   # (D, output_len)
    return X, y

# ---- EvoGP building blocks ---------------------------------------------------
def make_descriptor(cfg):
    return GenerateDescriptor(
        max_tree_len=cfg.get("max_tree_len", 64),
        input_len=cfg.get("features", 1),
        output_len=1,
        using_funcs=cfg["using_funcs"],           # the shared 8 primitives: +,-,*,/,sin,cos,<,>
        max_layer_cnt=cfg.get("max_layer_cnt", 6),
        const_samples=cfg.get("const_samples", [-1.0, 0.0, 1.0]),
    )

def make_algorithm(cfg, desc, P):
    return GeneticProgramming(
        initial_forest=Forest.random_generate(pop_size=P, descriptor=desc),
        crossover=DefaultCrossover(),                                   # EvoGP's subtree crossover; it takes no rate (always applied = rate 1.0)
        mutation=DefaultMutation(mutation_rate=cfg.get("mutation_rate", 0.8), descriptor=desc),
        selection=TournamentSelection(                                  # matches GALOP:
            tournament_size=cfg.get("tournament", 3),                  #   TOURNAMENT_SIZE
            best_probability=1.0,                                       #   deterministic "pick max"
            survivor_rate=cfg.get("survivor_rate", 0.1),                #   NOTE: despite upstream docstring ("retain in
                                                                        #   next generation"), step() uses these as the
                                                                        #   MATING POOL: all offspring parents drawn from
                                                                        #   top survivor_rate*P tournament winners. 0.1 was
                                                                        #   set from the docstring reading (elites-only
                                                                        #   persistence, GALOP parity); upstream examples
                                                                        #   use 0.3-0.5.
            elite_rate=cfg.get("elite_rate", 0.1)),                     #   ELITES
    )

def make_problem(cfg, X, y):
    return SymbolicRegression(datapoints=X, labels=y,
                              execute_mode=cfg.get("execute_mode", "auto"))

def get_data(cfg, D):
    """CSV if generated; otherwise synthesize on-device (large D). A 1e8-row CSV
    is ~2GB and minutes of Python parsing; kernel timing is value-independent,
    so uniform[-1,1] + the nguyen1 target at the same distribution is fair."""
    path = f"datasets/sweep/{cfg['dataset']}_D{D}_train.csv"
    if os.path.exists(os.path.join(ROOT, path)):
        return load_csv(path)
    assert cfg["dataset"] == "nguyen1", "on-device synth only implemented for nguyen1"
    torch.manual_seed(cfg.get("seed", 1234))
    X = torch.rand((D, 1), device=DEV) * 2 - 1
    y = X[:, 0]**3 + X[:, 0]**2 + X[:, 0]
    return X, y.reshape(-1, 1)

def mean_tree_size(forest):
    # S_bar: mean node count across the forest (batch_subtree_size[:, 0] = root subtree size)
    for attr in ("batch_subtree_size", "batch_node_cnt", "node_cnts", "sizes"):
        if hasattr(forest, attr):
            v = getattr(forest, attr)
            v = v[:, 0] if v.dim() > 1 else v
            return float(v.float().mean().item())
    return float(len(forest))  # fallback (wrong scale) -- fix once API confirmed

def mse(tree, X, y):
    pred = tree.forward(X).reshape(-1)
    return float(((pred - y.reshape(-1)) ** 2).mean().item())

# ---- modes -------------------------------------------------------------------
def run_eval(cfg, prov, writer, out_dir):
    desc = make_descriptor(cfg)
    max_work = cfg.get("max_work")          # optional P*D cap for grid sweeps
    for P in cfg["pops"]:
        try:
            forest = Forest.random_generate(pop_size=P, descriptor=desc)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"   P={P}: OOM generating forest -- skipping ({str(e)[:60]})")
            torch.cuda.empty_cache(); continue
        sbar = mean_tree_size(forest)
        for D in cfg["contexts"]:
            if max_work and float(P) * D > max_work:
                continue                     # cell too expensive; grid stays bounded
            X, y = get_data(cfg, D)
            problem = make_problem(cfg, X, y)
            for _ in range(cfg.get("warmup", 3)):
                problem.evaluate(forest)                                 # EvoGP's fitness = -MSE on the problem's data
            torch.cuda.synchronize()
            ts = []
            for _ in range(cfg.get("reps", 12)):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                problem.evaluate(forest)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            T = statistics.median(ts)
            writer.writerow(base_row(cfg, prov) | {
                "run_id": run_id(cfg, P=P, D=D), "mode": "eval", "strategy": "evogp",
                "D": D, "P": P, "length_cap": cfg.get("max_tree_len", 64), "generations": 1,
                "S_bar_total": f"{sbar:.2f}", "T_seconds": f"{T:.6e}",
                "gpops_s": f"{P*sbar*D/T:.3e}", "evals_per_s": f"{P/T:.3e}",
            })
            print(f"   P={P} D={D}: {P*sbar*D/T:.3e} ops/s")

def run_whole(cfg, prov, writer, out_dir):
    desc = make_descriptor(cfg)
    G = cfg.get("generations", 100)
    # jobs: either a D-sweep (contexts) or a named-dataset sweep (targets)
    if "targets" in cfg:
        dd = cfg.get("dataset_dir", "datasets/sweep")
        jobs = [(t, f"{dd}/{t}_train.csv") for t in cfg["targets"]]
    else:
        jobs = [(cfg["dataset"], D) for D in cfg["contexts"]]
    seeds = cfg.get("seeds", [cfg.get("seed", 1234)])
    for P in cfg["pops"]:
        for label, src in jobs:
            X, y = (load_csv(src) if isinstance(src, str) else get_data(cfg, src))
            problem = make_problem(cfg, X, y)
            D = int(y.shape[0])
            for seed in seeds:
                torch.manual_seed(seed)
                try:
                    alg = make_algorithm(cfg, desc, P)
                    pipe = StandardPipeline(alg, problem, generation_limit=G, is_show_details=False)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    best = pipe.run()
                    torch.cuda.synchronize(); T = time.perf_counter() - t0
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"   P={P} {label} s{seed}: OOM -- skipping ({str(e)[:60]})", flush=True)
                    torch.cuda.empty_cache(); continue
                sbar = mean_tree_size(alg.forest if hasattr(alg, "forest") else best)
                m = mse(best, X, y)
                writer.writerow(base_row(cfg, prov) | {
                    "run_id": run_id(cfg, P=P, D=D, label=label, seed=seed),
                    "mode": "whole_run", "strategy": "evogp",
                    "dataset": label, "D": D, "P": P, "seed": seed,
                    "length_cap": cfg.get("max_tree_len", 101), "generations": G,
                    "S_bar_total": f"{sbar:.2f}", "T_seconds": f"{T:.4f}",
                    "gpops_s": f"{G*P*sbar*D/T:.3e}", "evals_per_s": f"{P*G/T:.3e}",
                    "best_train_mse": f"{m:.6f}",
                })
                print(f"   P={P} {label} s{seed} D={D}: {T:.3f}s  mse={m:.6f}", flush=True)

def run_quality(cfg, prov, writer, out_dir):
    desc = make_descriptor(cfg)
    G, P = cfg.get("generations", 100), cfg.get("pop", 1000)
    for target in cfg["targets"]:
        Xtr, ytr = load_csv(f"{cfg['dataset_dir']}/{target}_train.csv")
        Xte, yte = load_csv(f"{cfg['dataset_dir']}/{target}_test.csv")
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            alg = make_algorithm(cfg, desc, P)
            best = StandardPipeline(alg, make_problem(cfg, Xtr, ytr),
                                    generation_limit=G, is_show_details=False).run()
            writer.writerow(base_row(cfg, prov) | {
                "run_id": run_id(cfg, target=target, seed=seed), "mode": "quality", "strategy": "evogp",
                "dataset": target, "D": len(ytr), "P": P, "generations": G, "seed": seed,
                "best_train_mse": f"{mse(best, Xtr, ytr):.6f}", "best_test_mse": f"{mse(best, Xte, yte):.6f}",
            })
            print(f"   {target} seed {seed}: test_mse={mse(best, Xte, yte):.6f}")

MODES = {"eval": run_eval, "whole_run": run_whole, "quality": run_quality}

def main():
    cfg = tomllib.load(open(sys.argv[1], "rb"))
    out_dir = os.path.join(ROOT, cfg.get("out_dir", f"results/{cfg['name']}"))
    os.makedirs(out_dir, exist_ok=True)
    prov = provenance()
    print(f"[{cfg['name']}] system=evogp mode={cfg['mode']} commit={prov[0]}")
    summary = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        if new: w.writeheader()
        MODES[cfg["mode"]](cfg, prov, w, out_dir)
    json.dump({"config": cfg, "commit": prov[0], "gpu": prov[1], "cuda": prov[2], "timestamp": prov[3]},
              open(os.path.join(out_dir, f"manifest_{cfg['name']}.json"), "w"), indent=2)
    print(f"  -> {summary}")

if __name__ == "__main__":
    main()
