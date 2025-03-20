# Load PAGs
from typing_extensions import Doc
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

import string
import random
import copy
import json
import itertools
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
import fcntl
import polars.selectors as cs


import dgp
import fedci

# supress R log
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

#ro.r['source']('./load_pags.r')
#load_pags = ro.globalenv['load_pags']

# 1. removed R multiprocessing (testing tn)
# 2. put rpy2 source file open into mp function
# 3. from rpy2.rinterface_lib import openrlib
# with openrlib.rlock:
#     # Your R function call here
#     pass

# load local-ci script
ro.r['source']('./ci_functions.r')
# load function from R script
get_data_f = ro.globalenv['get_data']
load_pags = ro.globalenv['load_pags']
run_ci_test_f = ro.globalenv['run_ci_test']
aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

truePAGs, subsetsList = load_pags()

subsetsList = [(sorted(tuple(x[0])), sorted(tuple(x[1]))) for x in subsetsList]

def floatmatrix_to_2dlist(r_floatmatrix):
    numpy_matrix = numpy2ri.rpy2py(r_floatmatrix)
    return numpy_matrix.astype(int).tolist()
#truePAGs = [floatmatrix_to_2dlist(pag) for pag in truePAGs]

# Adjacency Matrix Arrowheads:
# 0: Missing Edge
# 1: Dot Head
# 2: Arrow Head
# 3: Tail
def pag_to_node_collection(pag):
    alphabet = string.ascii_uppercase

    def get_node_collection(pag):
        nodes = []
        for i in range(len(pag)):
            nodes.append(dgp.GenericNode(name=alphabet[i]))

        for i in range(len(pag)):
            for j in range(i, len(pag)):
                # Arrowhead on Node i
                marker_1 = pag[i][j]
                # Arrowhead on Node j
                marker_2 = pag[j][i]

                assert (marker_1 != 0 and marker_2 != 0) or marker_1 == marker_2, 'If one is 0, the other needs to be as well'

                # no edge
                if marker_1 == 0 or marker_2 == 0:
                    continue

                # Turn odot ends into tails
                marker_1 = 3 if marker_1 == 1 else marker_1
                marker_2 = 3 if marker_2 == 1 else marker_2

                # edges must have at least one arrow
                assert marker_1 != 3 or marker_2 != 3, 'If one is tail, the other can not be'

                assert marker_1 in [2,3] and marker_2 in [2,3], 'Only tails and arrows allowed after this point'

                ## start adding parents
                if marker_1 == 2 and marker_2 == 2:
                    # add latent confounder
                    # TODO: Maybe make this only continuos values
                    confounder = dgp.GenericNode(name=f'L_{alphabet[i]}{alphabet[j]}')
                    nodes.append(confounder)
                    nodes[i].parents.append(confounder)
                    nodes[j].parents.append(confounder)
                elif marker_1 == 3 and marker_2 == 2:
                    nodes[i].parents.append(nodes[j])
                elif marker_1 == 2 and marker_2 == 3:
                    nodes[j].parents.append(nodes[i])
                else:
                    raise Exception('Two tails on one edge are not allowed at this point')
        nc = dgp.NodeCollection(
            name='test',
            nodes=nodes,
            drop_vars=[n.name for n in nodes[len(pag):]] # drop all vars outside the adjacency matrix -> confounders
        )
        return nc


    # TODO: AVOID - NEW COLLIDERS    (done)
    #             - CYCLES           (done)
    #             - UNDIRECTED EDGES (done)

    # Fix odot to odot edges by trying both
    def get_options_for_odot_edges(true_pag, pag):
        pags = []
        for i in range(len(pag)):
            for j in range(i, len(pag)):
                # Arrowhead on Node i
                marker_1 = pag[i][j]
                # Arrowhead on Node j
                marker_2 = pag[j][i]

                if marker_1 == 1 and marker_2 == 1:
                    pag_array = np.array(pag)
                    _pag_1 = pag_array.copy()
                    if np.sum((_pag_1[:,j] == 2) * (true_pag[:,j] == 1)) == 0:
                        _pag_1[i,j] = 2
                        _pag_1[j,i] = 3
                        pags.extend(get_options_for_odot_edges(true_pag, _pag_1.tolist()))

                    _pag_2 = pag_array.copy()
                    if np.sum((_pag_2[:,i] == 2) * (true_pag[:,i] == 1)) == 0:
                        _pag_2[i,j] = 3
                        _pag_2[j,i] = 2
                        pags.extend(get_options_for_odot_edges(true_pag, _pag_2.tolist()))

                    _pag_3 = pag_array.copy()
                    if (np.sum((_pag_3[:,i] == 2) * (true_pag[:,i] == 1)) == 0) and \
                        (np.sum((_pag_3[:,j] == 2) * (true_pag[:,j] == 1)) == 0):
                        _pag_3[i,j] = 2
                        _pag_3[j,i] = 2
                        pags.extend(get_options_for_odot_edges(true_pag, _pag_3.tolist()))

                    return pags
        return [pag]

    pags = get_options_for_odot_edges(np.array(copy.deepcopy(pag)), copy.deepcopy(pag))
    ncs = []
    for pag in pags:
        try:
            nc = get_node_collection(pag)
            ncs.append(nc)
        except:
            continue
    assert len(ncs) > 0, 'At least one result is required'
    nc = random.choice(ncs)
    return nc.reset()


def get_dataframe_from_r(test_setup, num_samples):
    raw_true_pag = test_setup[0]
    labels = sorted(list(set(test_setup[1][0] + test_setup[1][1])))
    potential_var_types = {'continuous': [1], 'binary': [2], 'ordinal': [3,4], 'nominal': [3,4]}
    var_types = {}
    var_levels = []
    for label in labels:
        var_type = random.choice(list(potential_var_types.keys()))
        var_types[label] = var_type
        var_levels += [random.choice(potential_var_types[var_type])]
    succeeded = False
    cnt = 0
    #print(var_types)
    while not succeeded:
        cnt += 1
        try:
            dat = get_data_f(raw_true_pag, num_samples, var_levels, 'mixed' if cnt < 3 else 'continuous')
        except ro.rinterface_lib.embedded.RRuntimeError as e:
            continue
        #dat = get_data_f(raw_true_pag, num_samples, var_levels, 'mixed') # continuos mixed
        #print('---')
        #print(dat)
        # "CAN ACTUALLY BE NULL, JUST RETRY"
        with (ro.default_converter + pandas2ri.converter).context():
            df = ro.conversion.get_conversion().rpy2py(dat[0])
        #print(df)
        try:
            df = pl.from_pandas(df)
        except:
            if cnt > 10:
                print(raw_true_pag)
            continue

        succeeded = True
    for var_name, var_type in var_types.items():
        if var_type == 'continuous':
            df = df.with_columns(pl.col(var_name).cast(pl.Float64))
        elif var_type == 'binary':
            df = df.with_columns(pl.col(var_name) == 'A')
        elif var_type == 'ordinal':
            repl_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            df = df.with_columns(pl.col(var_name).cast(pl.Utf8).replace(repl_dict).cast(pl.Int32))
        elif var_type == 'nominal':
            df = df.with_columns(pl.col(var_name).cast(pl.Utf8))
    return df

def get_data(test_setup, num_samples, num_clients):
    def split_dataframe(df, n):
        if n <= 0:
            raise ValueError("The number of splits 'n' must be greater than 0.")

        min_perc = 0.03
        percentiles = np.random.uniform(0,1,n)
        percentiles = (percentiles+min_perc)
        percentiles = percentiles/np.sum(percentiles)
        split_percentiles = percentiles.tolist()
        percentiles = np.cumsum(percentiles)
        percentiles = [0] + percentiles.tolist()[:-1] + [1]

        splits = []
        for i in range(n):
            start = int(percentiles[i]*len(df))
            end = int(percentiles[i+1]*len(df))
            splits.append(df[start:end])

        return splits, split_percentiles

    #pag = floatmatrix_to_2dlist(test_setup[0])
    #nc = pag_to_node_collection(pag)

    pag = test_setup[0]

    is_valid = False
    while not is_valid:
        #data = nc.reset().get(num_samples)
        data = get_dataframe_from_r(test_setup, num_samples)

        cols = data.columns
        cols_c1 = test_setup[1][0]
        cols_c2 = test_setup[1][1]
        cols_cx = [test_setup[1][i%2] for i in range(num_clients-2)]
        #cols_cx = [sorted(cols, key=lambda k: random.random())[:-1] for _ in range(num_clients-2)]

        split_dfs, split_percs = split_dataframe(data, num_clients)
        client_data = [df.select(c) for df, c in zip(split_dfs, [cols_c1, cols_c2] + cols_cx)]

        is_valid = not any([split.select(fail=pl.any_horizontal((cs.boolean() | cs.string() | cs.integer()).n_unique() == 1))['fail'][0] for split in client_data])
        if not is_valid:
            continue

        for d in client_data:
            if len(d) < 5:
                is_valid = False
                break
            if len(d.select(cs.boolean())) == 0:
                continue
            if any([l < 2 for l in d.group_by(cs.boolean()).len()['len'].to_list()]):
                is_valid = False
                break
    # 126

    return (pag, sorted(data.columns)), (client_data, split_percs)


def setup_server(client_data):
    # Create Clients
    clients = [fedci.Client(d) for d in client_data]

    #for cd in client_data:
    #    print(cd)

    # Create Server
    server = fedci.Server(
        {
            str(i): c for i, c in enumerate(clients)
        }
    )

    return server

def server_results_to_dataframe(labels, results):
    likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(results)

    columns = ('ord', 'X', 'Y', 'S', 'pvalue')
    rows = []

    lrt_ord_0 = [(lrt.v0, lrt.v1) for lrt in likelihood_ratio_tests if len(lrt.conditioning_set) == 0]
    label_combinations = itertools.combinations(labels, 2)
    missing_base_rows = []
    for label_combination in label_combinations:
        if label_combination in lrt_ord_0:
            continue
        #print('MISSING', label_combination)
        l0, l1 = label_combination
        missing_base_rows.append((0, labels.index(l0)+1, labels.index(l1)+1, "", 1))
    rows += missing_base_rows

    for test in likelihood_ratio_tests:
        s_labels_string = ','.join(sorted([str(labels.index(l)+1) for l in test.conditioning_set]))
        rows.append((len(test.conditioning_set), labels.index(test.v0)+1, labels.index(test.v1)+1, s_labels_string, test.p_val))

    df = pd.DataFrame(data=rows, columns=columns)
    return df
# Run fedci
#server.run()

# Run MXM local-ci.r per Client

def mxm_ci_test(df):
    df = df.with_columns(cs.string().cast(pl.Categorical()))
    df = df.to_pandas()
    with (ro.default_converter + pandas2ri.converter).context():
        # # load local-ci script
        # ro.r['source']('./local-ci.r')
        # # load function from R script
        # run_ci_test_f = ro.globalenv['run_ci_test']
        #converting it into r object for passing into r function
        df_r = ro.conversion.get_conversion().py2rpy(df)
        #Invoking the R function and getting the result
        result = run_ci_test_f(df_r, 999, "./examples/", 'dummy')
        #Converting it back to a pandas dataframe.
        df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
        labels = list(result['labels'])
    return df_pvals, labels

def run_pval_agg_iod(true_pag, true_labels, users, dfs, client_labels, alpha, procedure):
    #ro.r['source']('./aggregation.r')
    #aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        #r_dfs = ro.ListVector(r_dfs)
        label_list = [ro.StrVector(v) for v in client_labels]

        true_pag_np = np.array(true_pag)
        r_matrix = ro.r.matrix(ro.FloatVector(true_pag_np.flatten()), nrow=len(true_labels), ncol=len(true_labels))
        colnames = ro.StrVector(true_labels)

        result = aggregate_ci_results_f(r_matrix, colnames, label_list, r_dfs, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
        g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]

        found_correct_pag = bool(result['found_correct_pag'][0])
        g_pag_shd = [x[1][0].item() for x in result['G_PAG_SHD'].items()]
        g_pag_for = [x[1][0].item() for x in result['G_PAG_FOR'].items()]
        g_pag_fdr = [x[1][0].item() for x in result['G_PAG_FDR'].items()]

    return {
        "found_correct": found_correct_pag,

        "SHD": g_pag_shd,
        "FOR": g_pag_for,
        "FDR": g_pag_fdr,

        "MEAN_SHD": sum(g_pag_shd)/len(g_pag_shd) if len(g_pag_shd) > 0 else None,
        "MEAN_FOR": sum(g_pag_for)/len(g_pag_for) if len(g_pag_for) > 0 else None,
        "MEAN_FDR": sum(g_pag_fdr)/len(g_pag_fdr) if len(g_pag_fdr) > 0 else None,

        "MIN_SHD": min(g_pag_shd) if len(g_pag_shd) > 0 else None,
        "MIN_FOR": min(g_pag_for) if len(g_pag_for) > 0 else None,
        "MIN_FDR": min(g_pag_fdr) if len(g_pag_fdr) > 0 else None,

        "MAX_SHD": max(g_pag_shd) if len(g_pag_shd) > 0 else None,
        "MAX_FOR": max(g_pag_for) if len(g_pag_for) > 0 else None,
        "MAX_FDR": max(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
    }

def run_riod(true_pag, true_labels, df, labels, client_labels, alpha, procedure):
    # ro.r['source']('./aggregation.r')
    # iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']
    # Reading and processing data
    #df = pl.read_csv("./random-data-1.csv")

    # let index start with 1
    df.index += 1

    label_list = [ro.StrVector(v) for v in client_labels.values()]
    users = list(client_labels.keys())

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        #converting it into r object for passing into r function
        suff_stat = [
            ('citestResults', ro.conversion.get_conversion().py2rpy(df)),
            ('all_labels', ro.StrVector(labels)),
        ]
        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)

        true_pag_np = np.array(true_pag)
        r_matrix = ro.r.matrix(ro.FloatVector(true_pag_np.flatten()), nrow=len(true_labels), ncol=len(true_labels))
        colnames = ro.StrVector(true_labels)

        result = iod_on_ci_data_f(r_matrix, colnames, label_list, suff_stat, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
        g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
        g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
        gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

        #print(true_labels, labels, g_pag_labels)
        found_correct_pag = bool(result['found_correct_pag'][0])
        g_pag_shd = [x[1][0].item() for x in result['G_PAG_SHD'].items()]
        g_pag_for = [x[1][0].item() for x in result['G_PAG_FOR'].items()]
        g_pag_fdr = [x[1][0].item() for x in result['G_PAG_FDR'].items()]

    return {
        "found_correct": found_correct_pag,

        "SHD": g_pag_shd,
        "FOR": g_pag_for,
        "FDR": g_pag_fdr,

        "MEAN_SHD": sum(g_pag_shd)/len(g_pag_shd) if len(g_pag_shd) > 0 else None,
        "MEAN_FOR": sum(g_pag_for)/len(g_pag_for) if len(g_pag_for) > 0 else None,
        "MEAN_FDR": sum(g_pag_fdr)/len(g_pag_fdr) if len(g_pag_fdr) > 0 else None,

        "MIN_SHD": min(g_pag_shd) if len(g_pag_shd) > 0 else None,
        "MIN_FOR": min(g_pag_for) if len(g_pag_for) > 0 else None,
        "MIN_FDR": min(g_pag_fdr) if len(g_pag_fdr) > 0 else None,

        "MAX_SHD": max(g_pag_shd) if len(g_pag_shd) > 0 else None,
        "MAX_FOR": max(g_pag_for) if len(g_pag_for) > 0 else None,
        "MAX_FDR": max(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
    }

def filter_adjacency_matrices(pag, pag_labels, filter_labels):
    # Convert to numpy arrays for easier manipulation
    pag = np.array(pag)

    # Find indices of pred_labels in true_labels to maintain the order of pred_labels
    indices = [pag_labels.index(label) for label in filter_labels if label in pag_labels]

    # Filter the rows and columns of true_pag to match the order of pred_labels
    filtered_pag = pag[np.ix_(indices, indices)]

    # Extract the corresponding labels
    filtered_true_labels = [pag_labels[i] for i in indices]

    return filtered_pag.tolist(), filtered_true_labels

def evaluate_prediction(true_pag, pred_pag, true_labels, pred_labels):
    shd = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    correct_edges = 0
    other = 0

    true_sub_pag, true_sub_labels = filter_adjacency_matrices(true_pag, true_labels, pred_labels)
    if len(pred_pag) > len(pred_labels):
        pred_pag, _ = filter_adjacency_matrices(pred_pag, true_labels, pred_labels)

    assert tuple(true_sub_labels) == tuple(pred_labels), 'When evaluating, subgraph of true PAG needs to match vertices of predicted PAG'

    for i in range(len(true_sub_pag)):
        for j in range(i, len(true_sub_pag)):
            true_edge_start = true_sub_pag[i][j]
            true_edge_end = true_sub_pag[j][i]

            assert (true_edge_start != 0 and true_edge_end != 0) or true_edge_start == true_edge_end, 'Missing edges need to be symmetric'

            pred_edge_start = pred_pag[i][j]
            pred_edge_end = pred_pag[j][i]

            assert (pred_edge_start != 0 and pred_edge_end != 0) or pred_edge_start == pred_edge_end, 'Missing edges need to be symmetric'

            # Missing edge in both
            if true_edge_start == 0 and pred_edge_start == 0:
                tn += 1
                continue

            # False Positive
            if true_edge_start == 0 and pred_edge_start != 0:
                fp += 1
                shd += 1
                continue

            # False Negative
            if true_edge_start != 0 and pred_edge_start == 0:
                fn += 1
                shd += 1
                continue
            # True Positive
            if true_edge_start != 0 and pred_edge_start != 0:
                tp += 1
                continue

            # Same edge in both
            if true_edge_start == pred_edge_start and true_edge_end == pred_edge_end:
                correct_edges += 1
                continue

            other += 1
            shd += 1

    return shd, tp, tn, fp, fn, other, correct_edges

def log_results(
    target_dir,
    target_file,
    metrics_fedci,
    metrics_fedci_ot,
    metrics_fisher,
    metrics_fisher_ot,
    alpha,
    num_samples,
    num_clients,
    data_percs
):
    result = {
        "alpha": alpha,
        "num_samples": num_samples,
        "num_clients": num_clients,
        "split_percentiles": data_percs,
        "metrics_fedci": metrics_fedci,
        "metrics_fedci_ot": metrics_fedci_ot,
        "metrics_fisher": metrics_fisher,
        "metrics_fisher_ot": metrics_fisher_ot
    }

    with open(Path(target_dir) / target_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(result) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)


test_setups = list(zip(truePAGs, subsetsList))

#test_setups = test_setups[:1]
NUM_TESTS = 1
ALPHA = 0.05

# TODO: run the tests done so far for fedci with colliders with order IOD
# 500,1000,5000,10000 with 2,4 clients 10 times

#test_setups = test_setups[5:10]
data_dir = './experiments/simulation/iod_comparison'
data_file_pattern = '{}-{}-{}.ndjson'

import datetime
import polars as pl

now = int(datetime.datetime.utcnow().timestamp()*1e3)
data_file_pattern = str(now) + '-' + data_file_pattern

def run_comparison(setup):
    #print(setup)

    #np_matrix = np.array(setup[3][0])
    #print(np_matrix)
    idx, data_dir, data_file_pattern, test_setup, num_samples, num_clients = setup
    data_file = data_file_pattern.format(idx, num_samples, num_clients)
    (true_pag, all_labels), (client_data, data_percs) = get_data(test_setup, num_samples, num_clients)

    server = setup_server(client_data)

    results_fedci = server.run()
    all_labels_fedci = sorted(list(server.schema.keys()))
    client_labels = {id: sorted(list(schema.keys())) for id, schema in server.client_schemas.items()}
    df_fedci = server_results_to_dataframe(all_labels_fedci, results_fedci)

    metrics_fedci = run_riod(
        true_pag,
        all_labels,
        df_fedci,
        all_labels_fedci,
        client_labels,
        ALPHA,
        procedure='original'
    )

    metrics_fedci_ot = run_riod(
        true_pag,
        all_labels,
        df_fedci,
        all_labels_fedci,
        client_labels,
        ALPHA,
        procedure='orderedtriplets'
    )

    def run_client(client_data):
        server = fedci.Server({'1': fedci.Client(client_data)})
        results = server.run()
        client_labels = sorted(client_data.columns)
        df = server_results_to_dataframe(client_labels, results)
        return df, client_labels


    ## Run p val agg IOD

    # since use of own CI test, this throws errors on small sample sizes
    #try:
    client_ci_info = [mxm_ci_test(d) for d in client_data]
    #client_ci_info = [run_client(d) for d in client_data]
    client_ci_dfs, client_ci_labels = zip(*client_ci_info)

    metrics_fisher_ot = run_pval_agg_iod(
        true_pag,
        all_labels,
        list(client_labels.keys()),
        client_ci_dfs,
        client_ci_labels,
        ALPHA,
        procedure='orderedtriplets'
    )
        #except:
        #metrics_pvalagg = None

    metrics_fisher = run_pval_agg_iod(
        true_pag,
        all_labels,
        list(client_labels.keys()),
        client_ci_dfs,
        client_ci_labels,
        ALPHA,
        procedure='original'
    )

    #print(found_correct_pag_fedci, found_correct_pag_pvalagg)

    log_results(data_dir, data_file, metrics_fedci, metrics_fedci_ot, metrics_fisher, metrics_fisher, ALPHA, num_samples, num_clients, data_percs)

num_clients_options = [2,4] #,10] #,10]
num_samples_options = [500,1000,5000,10000] #,5000][50_000]

#pl.Config.set_tbl_rows(20)

configurations = list(itertools.product(test_setups, num_samples_options, num_clients_options))

configurations = [(data_dir, data_file_pattern) + c for c in configurations]

configurations = [(i,) + c for i in range(NUM_TESTS) for c in configurations]

from tqdm.contrib.concurrent import process_map
#from fedci.env import OVR, EXPAND_ORDINALS
#print(OVR, EXPAND_ORDINALS)

#for configuration in tqdm(configurations):
#    run_comparison(configuration)

process_map(run_comparison, configurations, max_workers=4, chunksize=1)
