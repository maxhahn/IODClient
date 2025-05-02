import math
import os
import copy
import fcntl
import json
from pathlib import Path
import numpy as np
import random

import polars as pl
import polars.selectors as cs

from dgp import NodeCollection

from .server import Server
from .client import Client
from .evaluation import get_riod_tests, compare_tests_to_truth, fisher_test_combination
from .env import DEBUG, EXPAND_ORDINALS, LOG_R, LR, RIDGE, NO_WRITE

import rpy2.rinterface_lib.callbacks as cb

def partition_dataframe_advanced(dgp_nodes, n_samples, n_clients):
    def split_dataframe(df, n):
        if n <= 0:
            raise ValueError("The number of splits 'n' must be greater than 0.")

        min_perc = 0.15
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

    is_valid = False
    while not is_valid:
        df = dgp_nodes.reset().get(n_samples)
        client_data, split_percs = split_dataframe(df, n_clients)

        is_valid = not any([split.select(fail=pl.any_horizontal((cs.boolean() | cs.string() | cs.integer()).n_unique() == 1))['fail'][0] for split in client_data])
        if not is_valid:
            continue

        for d in client_data:
            if len(d) < 2:
                is_valid = False
                break
            if len(d.select(cs.boolean())) == 0:
                continue
            if any([l < 2 for l in d.group_by(cs.boolean()).len()['len'].to_list()]):
                is_valid = False
                break
    return df, client_data, split_percs

def partition_dataframe(df, n, partition_ratios):
    assert partition_ratios is None or (sum(partition_ratios) == 1 and len(partition_ratios) == n), 'Malformed partition_ratios'

    total_rows = len(df)

    partitions = []
    if partition_ratios is None:
        partition_size = math.ceil(total_rows / n)

        for i in range(n):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size, total_rows)
            partition = df[start_idx:end_idx]
            partitions.append(partition)
    else:
        partition_offsets = [int(sr*total_rows) for sr in partition_ratios]
        partition_offsets = [0] + partition_offsets

        client_data = []
        for i in range(n):
            split_offset = partition_offsets[i]
            split_length = None
            if i+1 != n:
                split_length = partition_offsets[i+1] - partition_offsets[i]
            partitions.append(df.slice(split_offset, split_length))

    return partitions

def write_result(result, directory, file):
    with open(Path(directory) / file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(result) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

def run_configured_test(config, partition_ratios=None, seed=None):
    node_collection, num_samples, num_clients, target_directory, target_file, test_targets = config
    if not os.path.exists(target_directory) and not (DEBUG >= 1):
        os.makedirs(target_directory, exist_ok=True)
    target_file = f'{os.getpid()}-{target_file}'
    return run_test(dgp_nodes=node_collection,
                    num_samples=num_samples,
                    num_clients=num_clients,
                    target_directory=target_directory,
                    target_file=target_file,
                    test_targets=test_targets,
                    partition_ratios=partition_ratios,
                    seed=seed
                    )

def run_test(dgp_nodes: NodeCollection,
             num_samples,
             num_clients,
             target_directory,
             target_file,
             max_regressors=None,
             test_targets=None,
             partition_ratios=None,
             seed=None
             ):
    if seed is None:
        seed = random.randrange(2**30)
    if DEBUG >= 1: print(f'Current seed: {seed}')

    random.seed(2*seed)
    np.random.seed(seed)

    dgp_nodes = copy.deepcopy(dgp_nodes)

    #dgp_nodes.reset()
    #data = dgp_nodes.get(num_samples)

    return run_test_on_data(dgp_nodes,
                            num_samples,
                            dgp_nodes.name,
                            num_clients,
                            target_directory,
                            target_file,
                            max_regressors,
                            seed=seed,
                            test_targets=test_targets,
                            partition_ratios=partition_ratios
                            )

def run_test_on_data(dgp_nodes,
                     num_samples,
                     data_name,
                     num_clients,
                     target_directory,
                     target_file,
                     max_regressors=None,
                     seed=None,
                     test_targets=None,
                     partition_ratios=None
                     ):

    if LOG_R == 0:
        cb.consolewrite_print = lambda x: None
        cb.consolewrite_warnerror = lambda x: None

    #client_data_chunks = [chunk  for chunk in partition_dataframe(data, num_clients, partition_ratios)]
    data, client_data_chunks, split_percentages = partition_dataframe_advanced(dgp_nodes, num_samples, num_clients)

    if DEBUG >= 1:
        print("*** Data schema")
        for col, dtype in sorted(data.schema.items(), key=lambda x: x[0]):
            print(f"{col} - {dtype}")

    clients = {i:Client(chunk) for i, chunk in enumerate(client_data_chunks)}

    server = Server(
        clients,
        max_regressors=max_regressors,
        test_targets=test_targets,
        max_iterations=25
        )

    server.run()

    fed_tests = server.get_likelihood_ratio_tests(symmetric=True)
    baseline_tests = get_riod_tests(data, max_regressors=max_regressors, test_targets=test_targets)
    fisher_tests = [get_riod_tests(d, max_regressors=max_regressors, test_targets=test_targets) for d in client_data_chunks]

    fisher_tests = fisher_test_combination(fisher_tests)

    federated_p_values, fisher_p_values, baseline_p_values = compare_tests_to_truth(fed_tests, fisher_tests, baseline_tests, test_targets)

    result = {
        'name': data_name,
        'num_clients': num_clients,
        'num_samples': len(data),
        'max_regressors': max_regressors,
        'expanded_ordinals': True if EXPAND_ORDINALS == 1 else False,
        'lr': LR,
        'ridge': RIDGE,
        'seed': seed,
        'fisher_p_values': fisher_p_values,
        'federated_p_values': federated_p_values,
        'baseline_p_values': baseline_p_values,
        'test_targets': test_targets,
        'split_percentages': split_percentages
    }

    if DEBUG == 0 and NO_WRITE == 0:
        write_result(result, target_directory, target_file)

    return list(zip(sorted(fed_tests), sorted(fisher_tests), sorted(baseline_tests)))
