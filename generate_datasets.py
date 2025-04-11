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
msep_f = ro.globalenv['msep']
load_pags = ro.globalenv['load_pags']
run_ci_test_f = ro.globalenv['run_ci_test']
#aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
#iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

truePAGs, subsetsList = load_pags()

subsetsList = [(sorted(tuple(x[0])), sorted(tuple(x[1]))) for x in subsetsList]

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
        # get data from R function
        try:
            dat = get_data_f(raw_true_pag, num_samples, var_levels, 'continuous' if cnt > 2 else 'mixed')
        except:# ro.rinterface_lib.embedded.RRuntimeError as e:
            continue

        # convert R result to pandas
        with (ro.default_converter + pandas2ri.converter).context():
            df = ro.conversion.get_conversion().rpy2py(dat[0])

        # attempt to get correct dataframe
        try:
            df = pl.from_pandas(df)
        except:
            continue

        succeeded = True
    for var_name, var_type in var_types.items():
        if cnt > 2 or var_type == 'continuous':
            df = df.with_columns(pl.col(var_name).cast(pl.Float64))
        elif var_type == 'binary':
            df = df.with_columns(pl.col(var_name) == 'A')
        elif var_type == 'ordinal':
            repl_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            df = df.with_columns(pl.col(var_name).cast(pl.Utf8).replace(repl_dict).cast(pl.Int32))
        elif var_type == 'nominal':
            df = df.with_columns(pl.col(var_name).cast(pl.Utf8))
    return df

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

def test_ci(num_samples, test_setup, alpha = 0.05):
    all_labels = set(sorted(list(set(test_setup[1][0] + test_setup[1][1]))))

    labels1 = set(test_setup[1][0])
    labels2 = set(test_setup[1][1])

    label_intersect = labels1 & labels2
    #labels1_only = labels1 - label_intersect
    #labels2_only = labels2 - label_intersect

    labels1 = sorted(list(labels1))
    labels2 = sorted(list(labels2))
    label_intersect = sorted(list(label_intersect))

    #n = df.height
    #half = n // 2

    # loop here
    cnt = 0
    while True:

        df = data = get_dataframe_from_r(test_setup, num_samples)

        df1 = df[:len(df)//2]
        df2 = df[len(df)//2:]

        pl.Config.set_tbl_rows(50)

        result_df_intersect, result_labels_intersect = mxm_ci_test(df.select(label_intersect))
        result_df1, result_labels1 = mxm_ci_test(df1.select(labels1))
        result_df2, result_labels2 = mxm_ci_test(df2.select(labels2))

        result_df_intersect = pl.from_pandas(result_df_intersect).lazy()
        result_df1 = pl.from_pandas(result_df1).lazy()
        result_df2 = pl.from_pandas(result_df2).lazy()

        mapping1 = {str(i):l for i,l in enumerate(result_labels1, start=1)}
        result_df1 = result_df1.with_columns(
            pl.col('X').cast(pl.Utf8).replace(mapping1),
            pl.col('Y').cast(pl.Utf8).replace(mapping1),
            pl.col('S').str.split(',').list.eval(pl.element().replace(mapping1)).list.sort().list.join(','),
        )

        mapping2 = {str(i):l for i,l in enumerate(result_labels2, start=1)}
        result_df2 = result_df2.with_columns(
            pl.col('X').cast(pl.Utf8).replace(mapping2),
            pl.col('Y').cast(pl.Utf8).replace(mapping2),
            pl.col('S').str.split(',').list.eval(pl.element().replace(mapping2)).list.sort().list.join(','),
        )

        mapping_intersect = {str(i):l for i,l in enumerate(result_labels_intersect, start=1)}
        result_df_intersect = result_df_intersect.with_columns(
            pl.col('X').cast(pl.Utf8).replace(mapping_intersect),
            pl.col('Y').cast(pl.Utf8).replace(mapping_intersect),
            pl.col('S').str.split(',').list.eval(pl.element().replace(mapping_intersect)).list.sort().list.join(','),
        )

        #print(result_df1)
        #print(result_df1.schema)
        #print(result_df2)

        result_df = result_df1.join(result_df2, on=['ord', 'X', 'Y', 'S'], how='full', coalesce=True)

        result_df = result_df.with_columns(indep=pl.when(
            (pl.col('pvalue') > alpha) & (pl.col('pvalue_right') > alpha)
        ).then(pl.lit(True)).when(
            (pl.col('pvalue') < alpha) & (pl.col('pvalue_right') < alpha)
        ).then(pl.lit(False)).otherwise(pl.lit(None)))

        result_df = result_df.drop('pvalue', 'pvalue_right')

        result_df_intersect = result_df_intersect.with_columns(indep=(pl.col('pvalue') > alpha))
        result_df_intersect = result_df_intersect.drop('pvalue')

        result_df = result_df.join(result_df_intersect, on=['ord', 'X', 'Y', 'S'], how='anti')
        result_df = pl.concat([result_df, result_df_intersect]).collect()

        print(f'{100-(result_df["indep"].null_count()/len(result_df))*100:.2f}% data filled')
        if result_df['indep'].mean() > 0.8:
            break
        if cnt > 20:
            print(f'Stopping with {result_df["indep"].null_count()/len(result_df)}% data filled')
        cnt += 1

    return df1, df2, result_df


test_setups = [(pag, subset, i) for i,(pag,subset) in enumerate(zip(truePAGs, subsetsList))]
#test_setups = test_setups[:1]

#test_setups = test_setups[:1]
NUM_TESTS = 1
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
    idx, data_dir, data_file_pattern, test_setup, num_samples = setup
    #data_file = data_file_pattern.format(idx, num_samples)
    data = get_dataframe_from_r(test_setup, num_samples)

    data1, data2, df_faithful = test_ci(data, test_setup)


    df_msep = pl.read_parquet('./experiments/pag_msep/pag-{}.parquet'.format(test_setup[2]))
    df_msep = df_msep.with_columns(pl.col('S').list.join(','))

    df_faithful = df_faithful.join(df_msep, on=['X', 'Y', 'S'], how='left', coalesce=True)
    print(df_faithful)
    #print(data)
    asd
    # GET M SEPARABILITY
    #df = is_m_separable(test_setup)
    #df.write_parquet('./experiments/pag_msep/pag-{}.parquet'.format(test_setup[2]))

#pl.Config.set_tbl_rows(20)

num_samples_options = [10_000] #, 50_000, 100_000]

configurations = list(itertools.product(test_setups, num_samples_options))
configurations = [(data_dir, data_file_pattern) + c for c in configurations]
configurations = [(i,) + c for i in range(NUM_TESTS) for c in configurations]

#from tqdm.contrib.concurrent import process_map
#from fedci.env import OVR, EXPAND_ORDINALS
#print(OVR, EXPAND_ORDINALS)

for configuration in tqdm(configurations):
    generate_dataset(configuration)

#process_map(run_comparison, configurations, max_workers=4, chunksize=1)
