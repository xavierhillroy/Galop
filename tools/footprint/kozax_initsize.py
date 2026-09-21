#!/usr/bin/env python3
"""Kozax: initial tree size and per-individual bytes under the paper's configuration (Table 3).
Run in an environment with kozax==0.1.4 and a CUDA jax:  KOZAX_OPS=8 python tools/footprint/kozax_initsize.py"""
import numpy as np, jax.numpy as jnp, jax.random as jr
from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.SR_fitness_function import SymbolicRegressionFitnessFunction
OPS = [("+", lambda a, b: jnp.add(a, b), 2, 0.1), ("-", lambda a, b: jnp.subtract(a, b), 2, 0.1),
       ("*", lambda a, b: jnp.multiply(a, b), 2, 0.1), ("/", lambda a, b: jnp.where(b == 0, 1.0, a / b), 2, 0.1),
       ("sin", lambda a: jnp.sin(a), 1, 0.1), ("cos", lambda a: jnp.cos(a), 1, 0.1),
       ("lt", lambda a, b: jnp.where(a < b, 1.0, 0.0), 2, 0.1), ("gt", lambda a, b: jnp.where(a > b, 1.0, 0.0), 2, 0.1)]
P = 20000
for max_nodes in (101, 15):
    gp = GeneticProgramming(num_generations=1, population_size=P, fitness_function=SymbolicRegressionFitnessFunction(),
                            operator_list=OPS, variable_list=[["x0"]], layer_sizes=jnp.array([1]), tournament_size=3,
                            max_nodes=max_nodes, constant_optimization=False, device_type="gpu")
    arr = np.asarray(gp.initialize_population(jr.PRNGKey(70))).reshape(P, -1, 4)
    empty = np.array([0.0, -1.0, -1.0, 0.0], dtype=arr.dtype)            # Kozax's empty-node template
    used = (~np.all(arr == empty, axis=-1)).sum(axis=1)
    print(f"  max_nodes={max_nodes}: {arr.shape[1]} slots x 4 float32 = {arr.shape[1]*16} B per individual (fixed by the cap); "
          f"initial nodes: mean {used.mean():.2f} (min {used.min()}, max {used.max()}; max_init_depth {gp.max_init_depth})")
