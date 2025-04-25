import fedci
import dgp

import random
import math
import numpy as np
import polars as pl


NUM_SAMPLES = 1500
NUM_CLIENTS = 3

dir = 'experiments/datasets/data3/'
identifier = '1745511692009-7-50000-0.25-g-'

import os

files = os.listdir(dir)
files = [dir+f for f in files if identifier in f]


files1 = [f for f in files if 'd1_' in f]
files2 = [f for f in files if 'd2_' in f]

dfs1 = [pl.read_parquet(f) for f in files1]
dfs2 = [pl.read_parquet(f) for f in files2]

#print(dfs1[0].schema)
print(dfs2[0].schema)
print(dfs2[0].select(pl.all().n_unique()))

clients = {i:fedci.Client(c) for i,c in enumerate(dfs1+dfs2)}

server = fedci.Server(clients)
server.run()

experiment_tests = server.get_tests()
res = likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(server.get_tests(), test_targets=None)

for r in res:
    print(r)
