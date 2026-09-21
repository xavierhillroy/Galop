#!/usr/bin/env python3
"""
run.py — config-driven experiment harness for GALOP.

One config (TOML) fully specifies an experiment. The harness orchestrates the
right binary, augments its output with provenance (git commit, GPU, CUDA
version), and writes a unified, self-describing summary row per run plus a
manifest. Same config + a different commit = a reproducible before/after.

    python3 run.py configs/eval_crossover.toml

Modes:
  eval       -> cuda_bench      (runtime population + contexts; no recompile)
  whole_run  -> cuda_evo_bench  (recompiled per population; D at runtime)
  quality    -> lgp_run         (full evolution; train+test MSE + per-gen curve)
"""
import sys, os, csv, json, hashlib, subprocess, tomllib, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMMARY_COLS = [
    "run_id", "system", "mode", "strategy", "dataset", "D", "features",
    "P", "length_cap", "generations", "S_bar_total", "S_bar_effective",
    "op_denominator", "T_seconds", "gpops_s", "evals_per_s",
    "best_train_mse", "best_test_mse", "seed",
    "repo_commit", "gpu_name", "cuda_version", "timestamp",
]

def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)

def provenance():
    commit = sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip() or "nogit"
    gpu = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip().splitlines()
    gpu = gpu[0] if gpu else "unknown"
    nvcc = sh(["nvcc", "--version"]).stdout
    cuda = next((t for t in nvcc.split() if t.startswith("V") and t[1:2].isdigit()), "unknown")
    return commit, gpu, cuda, datetime.datetime.now().isoformat(timespec="seconds")

def run_id(cfg, **extra):
    key = {k: cfg.get(k) for k in ("system", "mode", "dataset", "seed")}
    key.update(extra)
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]

def base_row(cfg, prov):
    commit, gpu, cuda, ts = prov
    return {c: "" for c in SUMMARY_COLS} | {
        "system": cfg["system"], "dataset": cfg["dataset"], "features": cfg.get("features", 1),
        "seed": cfg.get("seed", 1234), "repo_commit": commit, "gpu_name": gpu,
        "cuda_version": cuda, "timestamp": ts, "op_denominator": "total_executed",
    }

NVCCFLAGS = ["-std=c++20", "--expt-relaxed-constexpr", "-O2", "-Isrc", "-arch=native",
             "-Xcompiler", "-fopenmp"]   # engine host operators are OpenMP-parallel
EVO_SRCS = ["tests/cuda_evo_bench.cu", "src/EvaluatorCuda.cu", "src/Dataset.cpp",
            "src/LGPEngine.cpp", "src/EvaluatorCpu.cpp", "src/Evaluator.cpp", "src/Interpreter.cpp"]
CXXFLAGS = ["-std=c++20", "-O2", "-Isrc", "-fopenmp"]
CPU_SRCS = ["src/LGPEngine.cpp", "src/Interpreter.cpp", "src/Evaluator.cpp",
            "src/Dataset.cpp", "src/EvaluatorCpu.cpp", "src/main.cpp"]

def run_eval(cfg, prov, writer, out_dir):
    if sh(["make", "cuda_bench"]).returncode != 0:
        sys.exit("build of cuda_bench failed")
    tmp = "/tmp/galop_eval.csv"
    max_work = cfg.get("max_work")          # optional P*D cap for grid sweeps
    for P in cfg["pops"]:                   # one call per P so the cap can filter D
        contexts = [D for D in cfg["contexts"] if not max_work or float(P) * D <= max_work]
        if not contexts:
            continue
        cmd = ["./cuda_bench",
               "--pops", str(P),
               "--contexts", ",".join(map(str, contexts)),
               "--strategies", ",".join(cfg["strategies"]),
               "--length", str(cfg.get("length", 8)),
               "--reps", str(cfg.get("reps", 12)),
               "--warmup", str(cfg.get("warmup", 3)), "--csv", tmp]
        print("  ", " ".join(cmd), flush=True)
        if sh(cmd).returncode != 0:
            print(f"   P={P}: cuda_bench failed (likely OOM) -- skipping", flush=True)
            continue
        for r in csv.DictReader(open(os.path.join(ROOT, tmp))):
            Pr, D, L = int(r["pop"]), int(r["contexts"]), int(r["length"])
            writer.writerow(base_row(cfg, prov) | {
                "run_id": run_id(cfg, strategy=r["strategy"], P=Pr, D=D),
                "mode": "eval", "strategy": r["strategy"], "D": D, "P": Pr,
                "length_cap": L, "generations": 1, "S_bar_total": L,
                "T_seconds": f"{float(r['time_ms'])/1e3:.6e}", "gpops_s": r["gpops_s"],
                "evals_per_s": r["evals_per_s"],
            })

def run_whole(cfg, prov, writer, out_dir):
    G = cfg.get("generations", 100)
    # jobs: a D-sweep (contexts, synthesized) or a named-dataset sweep (targets, CSVs)
    if "targets" in cfg:
        dd = cfg.get("dataset_dir", "datasets/sweep")
        jobs = [(t, f"{dd}/{t}_train.csv") for t in cfg["targets"]]
    else:
        jobs = [(cfg["dataset"], str(D)) for D in cfg["contexts"]]
    extra = ["warp"] if cfg.get("warp_only") else []
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])   # multi-seed -> variance bands
    defs = [f"-D{d}" for d in cfg.get("extra_defines", [])]   # e.g. the §3.5 standard config
    for P in cfg["pops"]:
        print(f"   compiling whole-run at P={P} (G={G}) ...", flush=True)
        b = sh(["nvcc", *NVCCFLAGS, *defs, f"-DPOPULATION_SIZE_OVERRIDE={P}",
                f"-DMAX_GENERATIONS_OVERRIDE={G}", *EVO_SRCS, "-o", "cuda_evo_bench_tmp"])
        if b.returncode != 0:
            sys.exit("whole-run build failed:\n" + b.stderr[-800:])
        for label, arg in jobs:
            for seed in seeds:
                out = sh(["./cuda_evo_bench_tmp", arg, str(seed)] + extra)
                if out.returncode != 0:
                    print(f"     P={P} {label} s{seed}: run failed (likely OOM) -- skipping", flush=True)
                    continue
                for line in out.stdout.strip().splitlines():
                    strat, pop, d, gens, sbar, sec, gpops, mse = line.split(",")
                    writer.writerow(base_row(cfg, prov) | {
                        "run_id": run_id(cfg, strategy=strat, P=int(pop), D=int(d),
                                         label=label, seed=seed),
                        "mode": "whole_run", "strategy": strat, "dataset": label,
                        "D": int(d), "P": int(pop), "seed": seed,
                        "length_cap": cfg.get("length_cap", 50), "generations": int(gens),
                        "S_bar_total": sbar, "T_seconds": sec, "gpops_s": gpops,
                        "evals_per_s": f"{int(pop)*int(gens)/float(sec):.3e}", "best_train_mse": mse,
                    })
            print(f"     P={P} {label} done ({len(seeds)} seed(s))", flush=True)
    os.path.exists(os.path.join(ROOT, "cuda_evo_bench_tmp")) and os.remove(os.path.join(ROOT, "cuda_evo_bench_tmp"))

def run_quality(cfg, prov, writer, out_dir):
    G, P = cfg.get("generations", 100), cfg.get("pop", 1000)
    print(f"   compiling lgp_run (P={P}, G={G}) ...")
    b = sh(["g++", *CXXFLAGS, f"-DPOPULATION_SIZE_OVERRIDE={P}",
            f"-DMAX_GENERATIONS_OVERRIDE={G}", *CPU_SRCS, "-o", "lgp_run_quality"])
    if b.returncode != 0:
        sys.exit("lgp_run_quality build failed:\n" + b.stderr[-800:])
    dd = cfg.get("dataset_dir", "datasets/sweep")
    for target in cfg["targets"]:
        for seed in cfg["seeds"]:
            rid = run_id(cfg, target=target, seed=seed)
            res = os.path.join(ROOT, "/tmp/galop_quality.csv")
            os.path.exists(res) and os.remove(res)
            fit = os.path.join(out_dir, f"fitness_{rid}.csv")
            sh(["./lgp_run_quality", f"{dd}/{target}_train.csv", f"{dd}/{target}_test.csv",
                target, str(seed), "/tmp/galop_quality.csv", "/tmp/galop_prog.txt", fit])
            r = open(res).read().strip().split(",")   # target,lgp,seed,train_mse,test_mse,best_len
            row = base_row(cfg, prov) | {
                "run_id": rid, "mode": "quality", "strategy": "cpu-oracle",
                "dataset": target, "D": cfg.get("D", 100), "P": P, "length_cap": r[5],
                "generations": G, "best_train_mse": r[3], "best_test_mse": r[4], "seed": seed,
            }
            writer.writerow(row)
            print(f"     {target} seed {seed}: train_mse={r[3]} test_mse={r[4]}")
    os.path.exists(os.path.join(ROOT, "lgp_run_quality")) and os.remove(os.path.join(ROOT, "lgp_run_quality"))

MODES = {"eval": run_eval, "whole_run": run_whole, "quality": run_quality}

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: run.py <config.toml>")
    cfg = tomllib.load(open(sys.argv[1], "rb"))
    out_dir = os.path.join(ROOT, cfg.get("out_dir", f"results/{cfg['name']}"))
    os.makedirs(out_dir, exist_ok=True)
    prov = provenance()
    print(f"[{cfg['name']}] mode={cfg['mode']} commit={prov[0]} gpu={prov[1]}")

    summary = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        if new:
            w.writeheader()
        MODES[cfg["mode"]](cfg, prov, w, out_dir)

    json.dump({"config": cfg, "commit": prov[0], "gpu": prov[1], "cuda": prov[2], "timestamp": prov[3]},
              open(os.path.join(out_dir, f"manifest_{cfg['name']}.json"), "w"), indent=2)
    print(f"  -> {summary}")

if __name__ == "__main__":
    main()
