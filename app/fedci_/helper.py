import math
import os
import copy
import fcntl
import json
from pathlib import Path
import numpy as np
import random

from dgp import NodeCollection

from .server import Server
from .client import Client
from .evaluation import get_symmetric_likelihood_tests, get_riod_tests, compare_tests_to_truth
from .env import DEBUG, EXPAND_ORDINALS, LOG_R, SEEDED, LR, RIDGE

import rpy2.rinterface_lib.callbacks as cb

def partition_dataframe(df, n):
    total_rows = len(df)
    partition_size = math.ceil(total_rows / n)

    partitions = []
    for i in range(n):
        start_idx = i * partition_size
        end_idx = min((i + 1) * partition_size, total_rows)
        partition = df[start_idx:end_idx]
        partitions.append(partition)

    return partitions

def write_result(result, directory, file):
    with open(Path(directory) / file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(result) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

def run_configured_test(config, seed=None):
    node_collection, num_samples, num_clients, target_directory, target_file = config
    if seed is not None:
        if DEBUG >= 1: print(f'Current seed: {seed}')
        np.random.seed(seed)
    if not os.path.exists(target_directory) and not (DEBUG >= 1):
        os.makedirs(target_directory, exist_ok=True)
    target_file = f'{os.getpid()}-{target_file}'
    return run_test(dgp_nodes=node_collection,
                    num_samples=num_samples,
                    num_clients=num_clients,
                    target_directory=target_directory,
                    target_file=target_file
                    )

def run_test(dgp_nodes: NodeCollection,
             num_samples,
             num_clients,
             target_directory,
             target_file,
             max_regressors=None
             ):
    if SEEDED >= 1:
        seed = random.randrange(2**32)
        np.random.seed(seed)
    dgp_nodes = copy.deepcopy(dgp_nodes)
    dgp_nodes.reset()
    data = dgp_nodes.get(num_samples)

    return run_test_on_data(data,
                            dgp_nodes.name,
                            num_clients,
                            target_directory,
                            target_file,
                            max_regressors,
                            seed=None if SEEDED==0 else seed
                            )

def run_test_on_data(data,
                     data_name,
                     num_clients,
                     target_directory,
                     target_file,
                     max_regressors=None,
                     seed=None
                     ):
    if DEBUG >= 1:
        print("*** Data schema")
        for col, dtype in sorted(data.schema.items(), key=lambda x: x[0]):
            print(f"{col} - {dtype}")

    if LOG_R == 0:
        cb.consolewrite_print = lambda x: None
        cb.consolewrite_warnerror = lambda x: None

    clients = {i:Client(chunk) for i, chunk in enumerate(partition_dataframe(data, num_clients))}
    server = Server(
        clients,
        max_regressors=max_regressors
        )

    server.run()

    likelihood_ratio_tests = get_symmetric_likelihood_tests(server.get_tests())
    ground_truth_tests = get_riod_tests(data, max_regressors=max_regressors)
    predicted_p_values, true_p_values = compare_tests_to_truth(likelihood_ratio_tests, ground_truth_tests)

    result = {
        'name': data_name,
        'num_clients': num_clients,
        'num_samples': len(data),
        'max_regressors': max_regressors,
        'expanded_ordinals': True if EXPAND_ORDINALS == 1 else False,
        'lr': LR,
        'ridge': RIDGE,
        'seed': seed,
        'predicted_p_values': predicted_p_values,
        'true_p_values': true_p_values
    }

    if DEBUG == 0:
        write_result(result, target_directory, target_file)

    return list(zip(sorted(likelihood_ratio_tests), sorted(ground_truth_tests)))
