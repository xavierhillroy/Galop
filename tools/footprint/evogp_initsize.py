#!/usr/bin/env python3
"""EvoGP: initial tree size and per-tree bytes under the paper's configuration (Table 3).
Run: python tools/footprint/evogp_initsize.py   (needs torch + evogp)"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "harness"))
from evogp_run import make_descriptor
from evogp.tree import Forest
desc = make_descriptor({"max_tree_len": 101, "features": 1, "using_funcs": ["+", "-", "*", "/", "sin", "cos", "<", ">"],
                        "const_samples": [0., 1., -1., 3.14159, 0.5, 2., 10., 100.]})
torch.manual_seed(70)
f = Forest.random_generate(pop_size=1_000_000, descriptor=desc)
tens = {k: v for k, v in vars(f).items() if torch.is_tensor(v)}
for k, v in tens.items(): print(f"  {k:20s} {tuple(v.shape)} {v.dtype} -> {v.numel()*v.element_size()//1_000_000} B per tree")
print(f"  forest: {sum(v.numel()*v.element_size() for v in tens.values())//1_000_000} B per tree, fixed by the 101-node cap (+4 B fitness)")
print(f"  mean initial tree size: {f.batch_subtree_size[:, 0].float().mean():.2f} nodes (min {int(f.batch_subtree_size[:, 0].min())}, max {int(f.batch_subtree_size[:, 0].max())})")
