"""Kozax probe (§15.6 idea-level anchor): our CSVs, parity primitive set
(+ - * / sin cos, protected division), tournament 3, constant_optimization
OFF (pure-GP class), authors' defaults otherwise (max_nodes=15, init depth 4).
Usage: kozax_bench.py train.csv test.csv P G seed"""
import sys, time, os
import numpy as np, jax, jax.numpy as jnp, jax.random as jr
from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.SR_fitness_function import SymbolicRegressionFitnessFunction

def load(p):
    d = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
    return jnp.array(d[:, :-1]), jnp.array(d[:, -1])

tr, te, P, G, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
Xtr, ytr = load(tr); Xte, yte = load(te)
k = Xtr.shape[1]
OPS = [("+", lambda a,b: jnp.add(a,b), 2, 0.1), ("-", lambda a,b: jnp.subtract(a,b), 2, 0.1),
       ("*", lambda a,b: jnp.multiply(a,b), 2, 0.1),
       ("/", lambda a,b: jnp.where(b==0, 1.0, a/b), 2, 0.1),
       ("sin", lambda a: jnp.sin(a), 1, 0.1), ("cos", lambda a: jnp.cos(a), 1, 0.1)]
if os.environ.get("KOZAX_OPS") == "8":     # full 8-op parity set (grid arms)
    # names must be sympy-parseable identifiers: kozax's pareto extraction
    # renders trees via sympy parse_expr, and "<"/">" crash it ("lt"/"gt"
    # parse as opaque functions; identical math)
    OPS += [("lt", lambda a,b: jnp.where(a<b, 1.0, 0.0), 2, 0.1),
            ("gt", lambda a,b: jnp.where(a>b, 1.0, 0.0), 2, 0.1)]
ff = SymbolicRegressionFitnessFunction()
gp = GeneticProgramming(num_generations=G, population_size=P, fitness_function=ff,
                        operator_list=OPS, variable_list=[["x"+str(i) for i in range(k)]],
                        layer_sizes=jnp.array([1]), tournament_size=3,
                        max_nodes=int(os.environ.get("KOZAX_MAX_NODES", "15")),  # default = authors'
                        constant_optimization=False, device_type="gpu")
t0 = time.perf_counter()
gp.fit(jr.PRNGKey(seed), (Xtr, ytr), verbose=0)
wall = time.perf_counter() - t0
try:                                        # readout armored: a pareto/sympy
    fits, cands = gp.pareto_front           # failure must never void the wall
    fits = np.array(fits).reshape(len(np.array(fits)), -1)[:, 0]
    bi = int(np.argmin(fits))
    train_mse = float(fits[bi])
    try:
        test_mse = float(ff(cands[bi], (Xte, yte), gp.tree_evaluator))
    except Exception:
        test_mse = float("nan")
except Exception:
    train_mse = float("nan"); test_mse = float("nan")
print(f"KOZAXROW,{P},{G},{seed},{wall:.3f},{train_mse:.6e},{test_mse:.6e}")
