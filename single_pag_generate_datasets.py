# Load PAGs
from jinja2.runtime import V
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
get_data_f = ro.globalenv['get_data_for_single_pag']
msep_f = ro.globalenv['msep_for_single_pag']
run_ci_test_f = ro.globalenv['run_ci_test']
#aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
#iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

subsetA = ['A', 'C', 'D', 'E']
subsetB = ['A', 'B', 'C', 'E']

def get_dataframe_from_r(num_samples, mode='mixed'):
    labels = sorted(list(set(subsetA + subsetB)))
    if mode == 'continuous':
        potential_var_types = {'continuous': [1]}
    else:
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
            dat = get_data_f(num_samples, var_levels, 'continuous' if cnt > 2 else mode, 0.2)
        except:# ro.rinterface_lib.embedded.RRuntimeError as e:
            print('Failed to do mixed graph')
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
def is_m_separable():
    labels = set(sorted(list(set(subsetA + subsetB))))

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
                is_msep = msep_f(x, y, list(s))
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

def test_ci(df_msep, num_samples, perc_split=[1,1], alpha = 0.05):
    labels1 = set(subsetA)
    labels2 = set(subsetB)
    label_intersect = labels1 & labels2

    labels1 = sorted(list(labels1))
    labels2 = sorted(list(labels2))
    label_intersect = sorted(list(label_intersect))


    CNT_MAX = 1000
    cnt = 0
    cnt_split_attempt = 0
    while True:
        df = get_dataframe_from_r(num_samples, mode='continuous')

        # test overlap
        result_intersect_df, result_intersect_labels = mxm_ci_test(df.select(label_intersect))
        result_intersect_df = pl.from_pandas(result_intersect_df)
        result_intersect_df = result_intersect_df.with_columns(indep=pl.col('pvalue')>ALPHA)

        mapping = {str(i):l for i,l in enumerate(result_intersect_labels, start=1)}
        result_intersect_df = result_intersect_df.with_columns(
            pl.col('X').cast(pl.Utf8).replace(mapping),
            pl.col('Y').cast(pl.Utf8).replace(mapping),
            pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
        )

        faithful_df = result_intersect_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='inner', coalesce=True)
        is_faithful3 = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)

        if not is_faithful3 and cnt < CNT_MAX:
            cnt += 1
            print(f'retry - {cnt} - overlap not faithful')
            # cnt += 1 # no cnt++ when overlap already fails
            continue

        # test subset1 and subset2
        subset1_frac = sum(perc_split[::2])/sum(perc_split)

        cutoff = int(subset1_frac*len(df))
        is_faithful1 = True
        is_faithful2 = True
        for _ in range(2):
            _df = df.sample(fraction=1, shuffle=True)
            df1 = _df[:cutoff].select(labels1)
            df2 = _df[cutoff:].select(labels2)

            result_subset1_df, result_subset1_labels = mxm_ci_test(df1)
            result_subset1_df = pl.from_pandas(result_subset1_df)
            result_subset1_df = result_subset1_df.with_columns(indep=pl.col('pvalue')>ALPHA)

            mapping = {str(i):l for i,l in enumerate(result_subset1_labels, start=1)}
            result_subset1_df = result_subset1_df.with_columns(
                pl.col('X').cast(pl.Utf8).replace(mapping),
                pl.col('Y').cast(pl.Utf8).replace(mapping),
                pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
            )

            result_subset1_df = result_subset1_df.join(result_intersect_df, on=['ord', 'X', 'Y', 'S'], how='anti')

            faithful_df = result_subset1_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='inner', coalesce=True)
            is_faithful1 = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)

            if not is_faithful1 and cnt < CNT_MAX:
                print(f'retry - {cnt} - subset 1 not faithful')
                #cnt += 1
                continue

            result_subset2_df, result_subset2_labels = mxm_ci_test(df2)
            result_subset2_df = pl.from_pandas(result_subset2_df)
            result_subset2_df = result_subset2_df.with_columns(indep=pl.col('pvalue')>ALPHA)

            mapping = {str(i):l for i,l in enumerate(result_subset2_labels, start=1)}
            result_subset2_df = result_subset2_df.with_columns(
                pl.col('X').cast(pl.Utf8).replace(mapping),
                pl.col('Y').cast(pl.Utf8).replace(mapping),
                pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
            )

            result_subset2_df = result_subset2_df.join(result_intersect_df, on=['ord', 'X', 'Y', 'S'], how='anti')

            faithful_df = result_subset2_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
            is_faithful2 = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)

            if not is_faithful2 and cnt < CNT_MAX:
                print(f'retry - {cnt} - subset 2 not faithful')
                #cnt += 1
                continue
        if (not is_faithful1 or not is_faithful2) and cnt < CNT_MAX:
            print(f'retry - {cnt} - subsets not faithful')
            cnt += 1
            continue

        # dfs1 = []
        # split_acc = 0
        # split_percs = perc_split[0::2]
        # split_percs = [s/sum(split_percs) for s in split_percs]
        # for split_perc in split_percs:#perc_split[0::2]:
        #     cutoff_from = int(split_acc * len(df1))
        #     cutoff_to = int((split_acc+split_perc) * len(df1))
        #     split_acc += split_perc
        #     _df = df1[cutoff_from:cutoff_to]
        #     dfs1.append(_df)

        # dfs2 = []
        # split_acc = 0
        # split_percs = perc_split[1::2]
        # split_percs = [s/sum(split_percs) for s in split_percs]
        # for split_perc in split_percs:#perc_split[1::2]:
        #     cutoff_from = int(split_acc * len(df2))
        #     cutoff_to = int((split_acc+split_perc) * len(df2))
        #     split_acc += split_perc
        #     _df = df2[cutoff_from:cutoff_to]
        #     dfs2.append(_df)

        return df1, df2, None, is_faithful1, is_faithful2, is_faithful3




def test_ci2(df_msep, num_samples, test_setup, perc_split, alpha = 0.05):
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

    CNT_MAX = 30
    # loop here
    cnt = 0
    cnt_split_attempt = 0

    resample = False
    while True:
        reused_df = False
        print(cnt, cnt_split_attempt)

        if cnt+cnt_split_attempt % 5 == 0 or resample:
            df = get_dataframe_from_r(test_setup, num_samples)#, mode='continuos')
            resample = False
        else:
            df = df.sample(fraction=1, shuffle=True)
            reused_df = True

        dfs1 = []
        dfs2 = []

        split_acc = 0
        for i, split_perc in enumerate(perc_split):
            cutoff_from = int(split_acc * len(df))
            cutoff_to = int((split_acc+split_perc) * len(df))
            split_acc += split_perc
            _df = df[cutoff_from:cutoff_to]
            if i % 2 == 0:
                dfs1.append(_df.select(labels1))
            else:
                dfs2.append(_df.select(labels2))

        #cutoff = int(perc_split * len(df))
        #df1 = df[:cutoff]
        #df2 = df[cutoff:]

        retry = False
        for df1 in dfs1:
            if not df1.select((cs.by_name(labels1) - cs.float()).n_unique()).equals(df.select((cs.by_name(labels1) - cs.float()).n_unique())):
                retry = True
        for df2 in dfs2:
            if not df2.select((cs.by_name(labels2) - cs.float()).n_unique()).equals(df.select((cs.by_name(labels2) - cs.float()).n_unique())):
                retry = True
        if retry:
            cnt_split_attempt += 1
            if cnt_split_attempt <= 10:
                continue

        # if not(df1.select((cs.by_name(labels1) - cs.float()).n_unique()).equals(df.select((cs.by_name(labels1) - cs.float()).n_unique()))\
        #     & df2.select((cs.by_name(labels2) - cs.float()).n_unique()).equals(df.select((cs.by_name(labels2) - cs.float()).n_unique()))):
        #         cnt_split_attempt += 1
        #         if cnt_split_attempt > 10:
        #             print('Problem creating splits', cnt_split_attempt)
        #         continue
        cnt_split_attempt = 0

        if not reused_df:
            intersect_labels = sorted(list(set(labels1) & set(labels2)))
            intersect_df = pl.concat([d.select(intersect_labels) for d in dfs1+dfs2])
            result_intersect_df, result_intersect_labels = mxm_ci_test(intersect_df)
            result_intersect_df = pl.from_pandas(result_intersect_df)
            result_intersect_df = result_intersect_df.with_columns(indep=pl.col('pvalue')>ALPHA)

            mapping = {str(i):l for i,l in enumerate(result_intersect_labels, start=1)}
            result_intersect_df = result_intersect_df.with_columns(
                pl.col('X').cast(pl.Utf8).replace(mapping),
                pl.col('Y').cast(pl.Utf8).replace(mapping),
                pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
            )

            faithful_df = result_intersect_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
            #pl.Config.set_tbl_rows(50)
            #print(faithful_df)
            #print(faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum(), len(faithful_df))
            is_faithful3 = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)

            if not is_faithful3 and cnt < CNT_MAX:
                print(f'overlap faithful: {is_faithful3}')
                resample = True
                cnt += 1
                continue

        subset1_df = pl.concat(dfs1)
        result_subset1_df, result_subset1_labels = mxm_ci_test(subset1_df)
        result_subset1_df = pl.from_pandas(result_subset1_df)
        result_subset1_df = result_subset1_df.with_columns(indep=pl.col('pvalue')>ALPHA)

        subset2_df = pl.concat(dfs2)
        result_subset2_df, result_subset2_labels = mxm_ci_test(subset2_df)
        result_subset2_df = pl.from_pandas(result_subset2_df)
        result_subset2_df = result_subset2_df.with_columns(indep=pl.col('pvalue')>ALPHA)

        mapping = {str(i):l for i,l in enumerate(result_subset1_labels, start=1)}
        result_subset1_df = result_subset1_df.with_columns(
            pl.col('X').cast(pl.Utf8).replace(mapping),
            pl.col('Y').cast(pl.Utf8).replace(mapping),
            pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
        )

        mapping = {str(i):l for i,l in enumerate(result_subset2_labels, start=1)}
        result_subset2_df = result_subset2_df.with_columns(
            pl.col('X').cast(pl.Utf8).replace(mapping),
            pl.col('Y').cast(pl.Utf8).replace(mapping),
            pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
        )

        faithful_df = result_subset1_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
        is_faithful1 = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)

        faithful_df = result_subset2_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
        is_faithful2 = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)


        if (not is_faithful1 or not is_faithful2) and cnt < CNT_MAX:
            print(f'dfs1 faithful: {is_faithful1} - dfs2 faithful: {is_faithful2}')
            #resample = True
            cnt += 1
            continue

        return dfs1, dfs2, None, is_faithful1, is_faithful2, is_faithful3

        #pl.Config.set_tbl_rows(50)

        result_dfs1 = []
        result_labels1_list = []
        result_dfs2 = []
        result_labels2_list = []

        for df1 in dfs1:
            result_df1, result_labels1 = mxm_ci_test(df1)
            result_df1 = pl.from_pandas(result_df1)
            result_df1 = result_df1.with_columns(indep=pl.col('pvalue')>ALPHA).drop('pvalue')
            result_dfs1.append(result_df1)
            result_labels1_list.append(result_labels1)
        for df2 in dfs2:
            result_df2, result_labels2 = mxm_ci_test(df2)
            result_df2 = pl.from_pandas(result_df2)
            result_df2 = result_df2.with_columns(indep=pl.col('pvalue')>ALPHA).drop('pvalue')
            result_dfs2.append(result_df2)
            result_labels2_list.append(result_labels2)

        result_dfs1_new = []
        for result_df1, result_labels1 in zip(result_dfs1, result_labels1_list):
            mapping = {str(i):l for i,l in enumerate(result_labels1, start=1)}
            result_df1 = result_df1.with_columns(
                pl.col('X').cast(pl.Utf8).replace(mapping),
                pl.col('Y').cast(pl.Utf8).replace(mapping),
                pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
            )
            result_dfs1_new.append(result_df1)
        result_dfs1 = result_dfs1_new

        result_dfs2_new = []
        for result_df2, result_labels2 in zip(result_dfs2, result_labels2_list):
            mapping = {str(i):l for i,l in enumerate(result_labels2, start=1)}
            result_df2 = result_df2.with_columns(
                pl.col('X').cast(pl.Utf8).replace(mapping),
                pl.col('Y').cast(pl.Utf8).replace(mapping),
                pl.col('S').str.split(',').list.eval(pl.element().replace(mapping)).list.sort().list.join(','),
            )
            result_dfs2_new.append(result_df2)
        result_dfs2 = result_dfs2_new

        faithful_client_cnt = 0
        for df1 in result_dfs1:
            faithful_df = df1.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
            is_faithful = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)
            if is_faithful:
                faithful_client_cnt += 1

        for df2 in result_dfs2:
            faithful_df = df2.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
            is_faithful = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)
            if is_faithful:
                faithful_client_cnt += 1

        print('Faithful clients:', faithful_client_cnt)
        is_any_client_faithful = faithful_client_cnt > 0

        if is_any_client_faithful and cnt < CNT_MAX:
            cnt += 1
            continue

        print(is_any_client_faithful, is_faithful1, is_faithful2, cnt)
        break
    return dfs1, dfs2, is_any_client_faithful, is_faithful1, is_faithful2, is_faithful3


NUM_TESTS = 100
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
    idx, data_dir, data_file_pattern, num_samples = setup
    #data_file = data_file_pattern.format(idx, num_samples)
    #data = get_dataframe_from_r(test_setup, num_samples)

    #data1, data2, df_faithful = test_ci(data, test_setup)

    # GET M SEPARABILITY
    #df = is_m_separable()
    #df.write_parquet('./experiments/pag_msep/pag-slides.parquet')
    #pl.Config.set_tbl_rows(80)
    #print(df)

    df_msep = pl.read_parquet('./experiments/pag_msep/pag-slides.parquet')
    df_msep = df_msep.with_columns(pl.col('S').list.join(','))
    df_msep = df_msep.with_columns(
        ord=pl.when(
            pl.col('S').str.len_chars() == 0).then(
                pl.lit(0)
            ).otherwise(
                pl.col('S').str.count_matches(',') + 1
            )
    )

    df1, df2, is_any_client_faithful, is_faithful1, is_faithful2, is_faithful_overlap = test_ci(df_msep, num_samples)

    print(df1.head())
    print(df2.head())
    #print(len(df_msep))
    # do full outer join here if you want to check all possible combinations

    #print(data1.columns, data2.columns)
    #print(df_faithful)
    #print(data)
    #asd

    #is_faithful = df_faithful.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(df_faithful)
    faith_id = ''
    if is_faithful1 and is_faithful2 and is_faithful_overlap:
        faith_id += 'g'
    if is_any_client_faithful:
        faith_id += 'l'
    if faith_id == '':
        faith_id = 'n'

    now = int(datetime.datetime.utcnow().timestamp()*1e3)

    ds_file_pattern = './experiments/datasets/data_slides/{}-slides-{}-{}-{}.parquet'

    df1.write_parquet(ds_file_pattern.format(now, num_samples, faith_id, 'p1'))
    df2.write_parquet(ds_file_pattern.format(now, num_samples, faith_id, 'p2'))

    # for i,df1 in enumerate(dfs1):
    #     _split_perc = f'{len(df1)/num_samples:.2f}'
    #     df1.write_parquet(ds_file_pattern.format(now, test_setup[2], num_samples, _split_perc, faith_id, f'd1_{i}'))
    # for i,df2 in enumerate(dfs2):
    #     _split_perc = f'{len(df2)/num_samples:.2f}'
    #     df2.write_parquet(ds_file_pattern.format(now, test_setup[2], num_samples, _split_perc, faith_id, f'd2_{i}'))



#pl.Config.set_tbl_rows(20)


num_samples_options = [4_000]#[4_000, 6_000, 8_000, 10_000]


configurations = [(data_dir, data_file_pattern, c) for c in num_samples_options]
configurations = [(i,) + c for i in range(NUM_TESTS) for c in configurations]


#configurations = configurations[20:-20]

#from tqdm.contrib.concurrent import process_map
#from fedci.env import OVR, EXPAND_ORDINALS
#print(OVR, EXPAND_ORDINALS)

import random
random.shuffle(configurations)

for configuration in tqdm(configurations):
    generate_dataset(configuration)

#process_map(run_comparison, configurations, max_workers=4, chunksize=1)
