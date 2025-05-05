import dgp
from fedci import run_test, run_configured_test
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools
import random

# Run with:
# EXPAND_ORDINALS=1 LR=0.4 RIDGE=0.02 python3 run_tests.py
# EXPAND_ORDINALS=1 python3 run_tests.py
# EXPAND_ORDINALS=1 OVR=0 python3 run_tests.py
# python3 run_tests.py

# Setup Data
# ## L-B CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
nc911 = dgp.NodeCollection('C-B Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode], parents=[node1])
nc912 = dgp.NodeCollection('C-B Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc913 = dgp.NodeCollection('C-B Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.BinaryNode])
nc914 = dgp.NodeCollection('C-B Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc915 = dgp.NodeCollection('C-B Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.BinaryNode])
nc916 = dgp.NodeCollection('C-B Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## L-M CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc921 = dgp.NodeCollection('C-M Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], parents=[node1], min_categories=3)
nc922 = dgp.NodeCollection('C-M Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc923 = dgp.NodeCollection('C-M Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc924 = dgp.NodeCollection('C-M Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc925 = dgp.NodeCollection('C-M Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
nc926 = dgp.NodeCollection('C-M Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## L-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc931 = dgp.NodeCollection('C-O Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc932 = dgp.NodeCollection('C-O Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc933 = dgp.NodeCollection('C-O Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc934 = dgp.NodeCollection('C-O Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc935 = dgp.NodeCollection('C-O Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc936 = dgp.NodeCollection('C-O Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## B-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc941 = dgp.NodeCollection('B-O Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc942 = dgp.NodeCollection('B-O Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc943 = dgp.NodeCollection('B-O Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc944 = dgp.NodeCollection('B-O Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.BinaryNode])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc945 = dgp.NodeCollection('B-O Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.BinaryNode])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc946 = dgp.NodeCollection('B-O Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## M-O CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc951 = dgp.NodeCollection('M-O Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], parents=[node1], min_categories=3)
nc952 = dgp.NodeCollection('M-O Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc953 = dgp.NodeCollection('M-O Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc954 = dgp.NodeCollection('M-O Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.CategoricalNode], min_categories=3)
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.OrdinalNode], min_categories=3)
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc955 = dgp.NodeCollection('M-O Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.CategoricalNode], min_categories=3)
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.OrdinalNode], min_categories=3)
nc956 = dgp.NodeCollection('M-O Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

## L-L CASE
# Unc. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
nc961 = dgp.NodeCollection('C-C Unc. Indep. : X Y', [node1, node2])
# Unc. Dep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node], parents=[node1])
nc962 = dgp.NodeCollection('C-C Unc. Dep. : X -> Y', [node1, node2])
# Con. Dep. Case given Z
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc963 = dgp.NodeCollection('C-C Con. Dep. : X -> Z <- Y', [node1, node2, node3])
# Con. Indep. Case given Z
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.Node])
nc964 = dgp.NodeCollection('C-C Con. Indep. : X <- Z -> Y', [node1, node2, node3])
# Unc. Conf. Indep. Case
node1 = dgp.GenericNode('X', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('Y', node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Z', parents=[node1, node2], node_restrictions=[dgp.Node])
nc965 = dgp.NodeCollection('C-C Unc. Conf. Indep. : X (-> Z <-) Y', [node1, node2, node3], drop_vars=['Z'])
# Unc. Conf. Dep. Case
node1 = dgp.GenericNode('Z', node_restrictions=[dgp.Node])
node2 = dgp.GenericNode('X', parents=[node1], node_restrictions=[dgp.Node])
node3 = dgp.GenericNode('Y', parents=[node1], node_restrictions=[dgp.Node])
nc966 = dgp.NodeCollection('C-C Unc. Conf. Dep. : X (<- Z ->) Y', [node1, node2, node3], drop_vars=['Z'])

# Setup Configs
node_collections = [
    nc911, nc912, nc913, nc914, #nc915, nc916,
    nc921, nc922, nc923, nc924, #nc925, nc926,
    nc931, nc932, nc933, nc934, #nc935, nc936,
    nc941, nc942, nc943, nc944, #nc945, nc946,
    nc951, nc952, nc953, nc954, #nc955, nc956,
    #nc961, nc962, nc963, nc964, nc965, nc966,
]


def run_mxm_ci_test():
    pass

def run():
    pass



num_samples = [
    100,
    200, 300, 400,
    500, #600, 700, 800,
    750,
    #900,
    1000,
    #1250,
    #1500,
    #1750,
    2000,
    #2500,
    3000,

    #5000,
    #7500,
    #10000
]
num_clients = [
    #1, 3, 5,
    2
]

file_info = ('./experiments/fed-v-fisher/', 'tests.ndjson')

configurations = list(itertools.product(node_collections, num_samples, num_clients))
configurations = [c + file_info for c in configurations]
test_targets_uncon = [('X', 'Y', ())]
test_targets_con = [('X', 'Y', ('Z',))]
#configurations = [c + (test_targets_uncon,) if 'Unc.' in c[0].name else c + (test_targets_con,) for c in configurations]

configurations = [c + (None,) for c in configurations]

num_runs = 1

configurations *= num_runs

# Run tests
#process_map(run_configured_test, configurations, max_workers=5, chunksize=1)
for i, configuration in enumerate(tqdm(configurations[::-1])):
    run_configured_test(configuration)
    break
