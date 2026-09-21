"""Minimal EvoGP eval for ncu profiling: a few evaluate() launches, P=1e5 D=32."""
import torch
from evogp.tree import Forest, GenerateDescriptor
from evogp.problem import SymbolicRegression
desc = GenerateDescriptor(max_tree_len=101, input_len=1, output_len=1,
    using_funcs=["+","-","*","/","sin","cos","<",">"], max_layer_cnt=6,
    const_samples=[0.,1.,-1.,3.14159,0.5,2.,10.,100.])
f = Forest.random_generate(pop_size=100000, descriptor=desc)
X = torch.rand((32,1), device="cuda")*2-1
y = (X[:,0]**3 + X[:,0]**2 + X[:,0]).reshape(-1,1)
prob = SymbolicRegression(datapoints=X, labels=y, execute_mode="auto")
for _ in range(8):
    prob.evaluate(f)
torch.cuda.synchronize()
print("evogp eval done")
