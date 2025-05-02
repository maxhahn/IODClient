from .testing import Test, SymmetricLikelihoodRatioTest, LikelihoodRatioTest, EmptyLikelihoodRatioTest
from .env import DEBUG

import scipy
import numpy as np
import polars as pl
import polars.selectors as cs
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import os
from typing import List

def _transform_dataframe(df):
    pandas2ri.activate()
    base = importr('base')
    df_copy = df.copy()
    r_list = {}#ro.ListVector({})

    for col in df_copy.columns:
        if df_copy[col].dtype == 'float64':
            r_list[col] = pandas2ri.py2rpy(df_copy[col])
        elif df_copy[col].dtype == 'object':
            r_list[col] = base.as_factor(pandas2ri.py2rpy(df_copy[col]))
        elif df_copy[col].dtype == 'int64':
            unique_values = sorted(df_copy[col].unique())
            r_list[col] = base.factor(pandas2ri.py2rpy(df_copy[col]),
                                    levels=ro.IntVector(unique_values),
                                    ordered=True)
        else:
            raise Exception(f'Could not properly convert column {col} with type {df_copy[col].dtype}')

    r_list = ro.ListVector(r_list)
    r_dataframe = base.as_data_frame(r_list)
    return r_dataframe

def get_riod_tests(data, max_regressors=None, test_targets=None):
    if max_regressors is None:
        max_regressors = 999

    data = data.with_columns(cs.integer().cast(pl.Int64))
    ground_truth_tests = []

    with (ro.default_converter + pandas2ri.converter).context():
        ro.r['source']('./ci_functions.r')
        run_ci_test_f = ro.globalenv['run_ci_test']
        df_r = _transform_dataframe(data.with_columns(cs.boolean().cast(pl.Utf8)).to_pandas())

        # delete potentially existing file
        pid = os.getpid()
        if os.path.exists(f'./tmp/citestResults_dummy_{pid}.csv'):
            os.remove(f'./tmp/citestResults_dummy_{pid}.csv')
        result = run_ci_test_f(df_r, max_regressors, "./tmp/", f'dummy_{pid}')
        df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
        labels = list(result['labels'])

    df = pl.from_pandas(df_pvals)
    df = df.drop('ord')
    df = df.with_columns(pl.col('S').str.split(',').cast(pl.List(pl.Int64)))
    df = df.with_columns(pl.col('X', 'Y').cast(pl.Int64))

    for row in df.rows():
        x = labels[row[0]-1]
        y = labels[row[1]-1]
        if x > y:
            x,y = y,x
        s = [str(labels[r-1]) for r in row[2] if r is not None]
        s = sorted(s)
        pval = row[3]
        ground_truth_tests.append(EmptyLikelihoodRatioTest(x, y, s, pval))

    if test_targets is not None:
        ground_truth_tests = [t for t in ground_truth_tests if (t.v0, t.v1, tuple(sorted(t.conditioning_set))) in test_targets]

    return sorted(ground_truth_tests)

def fisher_test_combination(tests: List[List[SymmetricLikelihoodRatioTest]]):
    tests = list(zip(*tests))
    results = []
    for _tests in tests:
        v0 = _tests[0].v0
        v1 = _tests[0].v1
        conditioning_set = _tests[0].conditioning_set
        assert all([t.v0 == v0 and t.v1 == v1 and t.conditioning_set == conditioning_set for t in _tests]), 'Unmatched tests found'
        p_values = [t.p_val for t in _tests]
        stat = -2 * np.sum(np.log(np.clip(np.array(p_values), 1e-12, 1)))
        new_p_val = scipy.stats.chi2.sf(stat, 2*len(p_values))
        results.append(EmptyLikelihoodRatioTest(v0, v1, conditioning_set, new_p_val))
    return results

def compare_tests_to_truth(
    federated_tests: List[SymmetricLikelihoodRatioTest],
    fisher_tests: List[SymmetricLikelihoodRatioTest],
    ground_truth_tests: List[SymmetricLikelihoodRatioTest],
    test_targets
):
    p_values = []
    for gt_test in ground_truth_tests:
        fed_test = [t for t in federated_tests if t == gt_test]
        assert len(fed_test) == 1, 'Exactly one match expected'
        fed_test = fed_test[0]

        fisher_test = [t for t in fisher_tests if t == gt_test]
        assert len(fisher_test) == 1, 'Exactly one match expected'
        fisher_test = fisher_test[0]

        p_values.append((fed_test.p_val, fisher_test.p_val, gt_test.p_val))

    return list(zip(*p_values))
