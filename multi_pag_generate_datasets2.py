# Load PAGs
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

COEF_THRESHOLD = 0.2

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
msep_f = ro.globalenv['msep']
load_pags = ro.globalenv['load_pags']
run_ci_test_f = ro.globalenv['run_ci_test']
#aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
#iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

truePAGs, subsetsList = load_pags()

subsetsList = [(sorted(tuple(x[0])), sorted(tuple(x[1]))) for x in subsetsList]


def get_dataframe_from_r(test_setup, num_samples, mode='mixed'):
    raw_true_pag = test_setup[0]
    labels = sorted(list(set(test_setup[1][0] + test_setup[1][1])))
    potential_var_types = {'continuous': [1], 'binary': [2], 'ordinal': [3,4], 'nominal': [3,4]}

    var_types = {}
    var_levels = []
    for label in labels:
        var_type = random.choice(list(potential_var_types.keys()))
        var_types[label] = var_type
        var_levels += [random.choice(potential_var_types[var_type])]

    dat = get_data_f(raw_true_pag, num_samples, var_levels, mode, COEF_THRESHOLD)
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(dat[0])
    df = pl.from_pandas(df)
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

def test_faithfulness(df, df_msep, antijoin_df=None):
    result_df, result_labels = mxm_ci_test(df)
    result_df = pl.from_pandas(result_df)
    result_df = result_df.with_columns(indep=pl.col('pvalue')>ALPHA)
    mapping = {str(i):l for i,l in enumerate(df.columns, start=1)}
    result_df = result_df.with_columns(
        pl.col('X').cast(pl.Utf8).replace(mapping),
        pl.col('Y').cast(pl.Utf8).replace(mapping),
        pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
    )

    if antijoin_df is not None:
        result_df = result_df.join(antijoin_df, on=['ord', 'X', 'Y', 'S'], how='anti')

    faithful_df = result_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='inner', coalesce=True)
    faithful_df = faithful_df.with_columns(is_faithful=(pl.col('indep') == pl.col('MSep')))
    is_faithful = faithful_df['is_faithful'].sum() == len(faithful_df)

    return is_faithful, result_df, faithful_df

def split_data(test_setup, df, df_msep, splits):
    is_fully_faithful_list = []
    partition_1_labels = sorted(test_setup[1][0])
    partition_2_labels = sorted(test_setup[1][1])
    intersection_labels = sorted(list(set(partition_1_labels) & set(partition_2_labels)))
    dfs = []

    is_faithful, overlap_df, _ = test_faithfulness(df.select(intersection_labels), df_msep)
    if not is_faithful:
        #print('...... Intersection of partitions not faithful. Skipping...')
        #is_fully_faithful = False
        pass
        #return dfs

    for split in splits:
        total_split = sum(split[0])+sum(split[1])
        split_frac = sum(split[0])/total_split
        df1 = df[:int(split_frac*len(df))]
        df2 = df[int(split_frac*len(df)):]

        dfs1 = []
        split_acc = 0
        split_percs = split[0]
        split_percs = [s/sum(split_percs) for s in split_percs]
        for split_perc in split_percs:
            cutoff_from = int(split_acc * len(df1))
            cutoff_to = int((split_acc+split_perc) * len(df1))
            split_acc += split_perc
            _df = df1[cutoff_from:cutoff_to]
            dfs1.append(_df.select(partition_1_labels))

        dfs2 = []
        split_acc = 0
        split_percs = split[1]
        split_percs = [s/sum(split_percs) for s in split_percs]
        for split_perc in split_percs:
            cutoff_from = int(split_acc * len(df2))
            cutoff_to = int((split_acc+split_perc) * len(df2))
            split_acc += split_perc
            _df = df2[cutoff_from:cutoff_to]
            dfs2.append(_df.select(partition_2_labels))

        all_partitions_faithful = True
        faithfulness_dfs = []
        for i, df_i in enumerate(dfs1, start=1):
            is_faithful, _, faithfulness_df = test_faithfulness(df_i.select(intersection_labels), df_msep)
            faithfulness_dfs.append(faithfulness_df.filter(~pl.col('is_faithful')).select('ord', 'X', 'Y', 'S', client_number=pl.lit(f'1-{i}'), split_portion=pl.lit(split[0][i-1])))
            all_partitions_faithful = all_partitions_faithful and is_faithful
        for i, df_i in enumerate(dfs2, start=1):
            is_faithful, _, faithfulness_df = test_faithfulness(df_i.select(intersection_labels), df_msep)
            faithfulness_dfs.append(faithfulness_df.filter(~pl.col('is_faithful')).select('ord', 'X', 'Y', 'S', client_number=pl.lit(f'2-{i}'), split_portion=pl.lit(split[1][i-1])))
            all_partitions_faithful = all_partitions_faithful and is_faithful
        if all_partitions_faithful:
            #print('...... All partitions are faithful. Skipping...')
            #is_fully_faithful = False
            #continue
            pass

        is_faithful1, _, _ = test_faithfulness(pl.concat(dfs1), df_msep, overlap_df)
        is_faithful2, _, _ = test_faithfulness(pl.concat(dfs2), df_msep, overlap_df)

        if not is_faithful1 or not is_faithful2:
            #print('...... Data is not faithful globally. Skipping...')
            #is_fully_faithful = False
            pass
            #continue
        dfs.append((split, (dfs1, dfs2), pl.concat(faithfulness_dfs)))
        is_fully_faithful_list.append(is_faithful & is_faithful1 & is_faithful2)
    return dfs, is_fully_faithful_list

from itertools import chain, combinations
def is_m_separable(test_setup):
    pag = test_setup[0]
    labels = set(sorted(list(set(test_setup[1][0] + test_setup[1][1]))))

    cnt = 0
    result = []
    for x in labels:
        label_wo_x = labels - {x}
        for y in label_wo_x:
            if x > y:
                continue
            conditioning_set = chain.from_iterable(combinations(label_wo_x - {y}, r) for r in range(0, len(label_wo_x)))
            for s in conditioning_set:

                cnt += 1
                is_msep = msep_f(pag, x, y, list(s))
                r = {
                    'X': x,
                    'Y': y,
                    'S': sorted(list(s)),
                    'MSep': bool(is_msep[0])
                }
                result.append(r)
                #print(x,y,s, bool(is_msep[0]))
    df = pl.from_dicts(result)
    return df

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

def test_cix(df_msep, test_setup, num_samples, perc_split, alpha = 0.05):

    if len(dfs) == 0:
        return None



test_setups = [(pag, subset, i) for i,(pag,subset) in enumerate(zip(truePAGs, subsetsList))]
#test_setups = test_setups[:1]

#test_setups = test_setups[:1]
NUM_TESTS = 10
# ls -la experiments/datasets/*/*-100000-faith.parquet | wc -l
ALPHA = 0.05

# TODO: run the tests done so far for fedci with colliders with order IOD
# 500,1000,5000,10000 with 2,4 clients 10 times
data_dir = './experiments/datasets/'
data_file_pattern = '{}-{}.ndjson'

import datetime
import polars as pl

now = int(datetime.datetime.utcnow().timestamp()*1e3)
data_file_pattern = str(now) + '-' + data_file_pattern

def generate_dataset(setup):
    idx, data_dir, data_file_pattern, test_setup, num_samples, perc_split = setup

    df_msep = pl.read_parquet('./experiments/pag_msep/pag-{}.parquet'.format(test_setup[2]))
    df_msep = df_msep.with_columns(pl.col('S').list.join(','))
    df_msep = df_msep.with_columns(
        ord=pl.when(
            pl.col('S').str.len_chars() == 0).then(
                pl.lit(0)
            ).otherwise(
                pl.col('S').str.count_matches(',') + 1
            )
    )

    is_faithful = False
    while not is_faithful:
        df = get_dataframe_from_r(test_setup, num_samples)
        dfs, is_faithful = split_data(test_setup, df, df_msep, perc_split)

        is_faithful = is_faithful[0]

    #if len(dfs) == 0:
    #    #print('... Not faithful')
    #    return

    #print('!!! Faithful')

    now = int(datetime.datetime.utcnow().timestamp()*1e3)
    ds_file_pattern = './experiments/datasets/mixed_pag/{}-{}-{}-{}-{}.parquet'
    if is_faithful:
        faith_id = 'g'
    else:
        faith_id = 'n'

    for split, (dfs1, dfs2), faith_df in dfs:
        for i, df1 in enumerate(dfs1, start=1):
            df1.write_parquet(ds_file_pattern.format(now, test_setup[2], num_samples, faith_id, f'p1-{i}'))
        for i, df2 in enumerate(dfs2, start=1):
            df2.write_parquet(ds_file_pattern.format(now, test_setup[2], num_samples, faith_id, f'p2-{i}'))

#pl.Config.set_tbl_rows(20)

#num_client_options = [4]
num_samples_options = [4_000] #, 50_000, 100_000]
#split_options = [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]#[0.1,0.5]
split_options = [[[(1,1),(1,1)]]]#[0.1,0.5]


# 10_000 globally -> does it give faithfulness?
#
#

# THREE TAIL PAGS
#  [1]  2 16 18 19 20 23 29 31 37 42 44 53 57 58 62 64 66 69 70 72 73 74 75 79 81 82 83 84 93 98
three_tail_pags = [2, 16, 18, 19, 20, 23, 29, 31, 37, 42, 44, 53, 57, 58, 62, 64, 66, 69, 70, 72, 73, 74, 75, 79, 81, 82, 83, 84, 93, 98]
three_tail_pags = [t-1 for t in three_tail_pags]
assert len(three_tail_pags) == 30

#three_tail_pags = [1, 41, 83, 69, 81, 19, 36]
#three_tail_pags = [81, 1, 83, 36, 69]
#three_tail_pags = [81, 83, 36, 69]
#three_tail_pags = [69]
#three_tail_pags = [30, 41, 1, 81, 69, 65, 56, 92, 28, 83]
#three_tail_pags = [69, 30, 61, 28, 80, 1, 83, 18, 92, 22, 78, 19, 81]
#three_tail_pags = [69, 30, 61, 80, 83, 92, 28, 78, 1, 81]
#three_tail_pags = [69, 30, 61, 80, 83, 92, 78, 81]

test_setups = [t for t in test_setups if t[2] in three_tail_pags]

configurations = list(itertools.product(test_setups, num_samples_options, split_options))
configurations = [(data_dir, data_file_pattern) + c for c in configurations]
configurations = [(i,) + c for i in range(NUM_TESTS) for c in configurations]


#configurations = configurations[20:-20]

#from tqdm.contrib.concurrent import process_map
#from fedci.env import OVR, EXPAND_ORDINALS
#print(OVR, EXPAND_ORDINALS)

import random
#random.shuffle(configurations)

for configuration in tqdm(configurations):
    generate_dataset(configuration)

#process_map(run_comparison, configurations, max_workers=4, chunksize=1)
