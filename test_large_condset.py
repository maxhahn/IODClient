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

df2 = pl.concat(dfs2)

test_targets = [('B', 'E', ('C', 'D'))]

#print(dfs1[0].schema)
print(dfs2[0].schema)
print(dfs2[0].select(pl.all().n_unique()))

clients = {i:fedci.Client(c) for i,c in enumerate(dfs1+dfs2)}

server = fedci.Server(clients, test_targets=test_targets)
# server.run()

# experiment_tests = server.get_tests()
# res = likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(server.get_tests(), test_targets=None)

# for r in res:
#     print(r)

print('---'*5)
print('---'*5)
print('---'*5)

clients = {1:fedci.Client(df2)}
server = fedci.Server(clients, test_targets=test_targets)
# server.run()

# experiment_tests = server.get_tests()
# res = likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(server.get_tests(), test_targets=None)

# for r in res:
#     print(r)

print('---'*5)
print('---'*5)
print('---'*5)

import rpy2.rinterface_lib.callbacks as cb
#cb.consolewrite_print = lambda x: None
#cb.consolewrite_warnerror = lambda x: None

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import polars.selectors as cs

ro.r['source']('./ci_functions.r')
run_ci_test_f = ro.globalenv['run_ci_test']
run_ci_test2_f = ro.globalenv['run_ci_test2']

#categories=['low', 'medium', 'high'], ordered=True
import pandas as pd
def cast_int_columns_to_ordered(df):
    for col in df.select_dtypes(include='int').columns:
        unique_vals = sorted(df[col].unique())
        df[col] = pd.Categorical(df[col], categories=unique_vals, ordered=True)
    return df

def mxm_ci_test(df):
    df = df.with_columns(cs.string().cast(pl.Categorical()))
    df = df.to_pandas()
    #df = cast_int_columns_to_ordered(df)
    with (ro.default_converter + pandas2ri.converter).context():
        #converting it into r object for passing into r function
        df_r = ro.conversion.get_conversion().py2rpy(df)
        #Invoking the R function and getting the result

        run_ci_test2_f('B','E',ro.StrVector(['C','D']),df_r)
        print('next')
        run_ci_test2_f('E','B',ro.StrVector(['C','D']),df_r)
        result = run_ci_test_f(df_r, 999, "./examples/", 'dummy')

        #Converting it back to a pandas dataframe.
        df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
        labels = list(result['labels'])
    return df_pvals, labels





res, labels = mxm_ci_test(df2)
res = pl.from_pandas(res)
label_mapping = {str(i):l for i,l in enumerate(labels, start=1)}
res = res.with_columns(
    pl.col('X').cast(pl.Utf8).replace(label_mapping),
    pl.col('Y').cast(pl.Utf8).replace(label_mapping),
    pl.col('S').str.split(',').list.eval(pl.element().replace(label_mapping)).list.sort().list.join(','),
)
print(res.filter((pl.col('X')=='B') & (pl.col('Y')=='E')))
