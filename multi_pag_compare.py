import polars as pl
import polars.selectors as cs
import pandas as pd
import os
import glob
import numpy as np
import itertools

from collections import OrderedDict

import fedci

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

import json

# load local-ci script
ro.r['source']('./ci_functions.r')
# load function from R script
run_ci_test_f = ro.globalenv['run_ci_test']
aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']
load_pags = ro.globalenv['load_pags']

truePAGs, subsetsList = load_pags()
subsetsList = [(sorted(tuple(x[0])), sorted(tuple(x[1]))) for x in subsetsList]
truePAGs_map = {i:pag for i, pag in enumerate(truePAGs, start=0)}

DATA_DIR = 'experiments/datasets/mixed_pag'
ALPHA = 0.05
PROCEDURE = 'original'

LOGFILE = './mixed_pag_results_non_faithful.json'

"""
from collections import Counter
import os
DATA_DIR = 'experiments/datasets/mixed_pag'
files = os.listdir(DATA_DIR)
files = set([f.rpartition('-p')[0] for f in files])
files = sorted(files)
pagids = [f.split('-')[1] for f in files]

three_tail_pags = [2, 16, 18, 19, 20, 23, 29, 31, 37, 42, 44, 53, 57, 58, 62, 64, 66, 69, 70, 72, 73, 74, 75, 79, 81, 82, 83, 84, 93, 98]
three_tail_pags = [str(t-1) for t in three_tail_pags]

print(len(three_tail_pags) - len(Counter(pagids)))
set(three_tail_pags) - set(pagids)


pagids = [f.split('-')[1] for f in files if f.endswith('-g')]
sorted([(k,v) for k,v in Counter(pagids).items()], key=lambda x: x[1])

"""



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

def run_pval_agg_iod(true_pag, true_labels, dfs, client_labels, alpha, procedure):
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
        g_pag_labels = [list([str(a) for a in x[1]]) for x in result['G_PAG_Label_List'].items()]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list([str(a) for a in x[1]]) for x in result['Gi_PAG_Label_List'].items()]

        found_correct_pag = bool(result['found_correct_pag'][0])
        g_pag_shd = [x[1][0].item() for x in result['G_PAG_SHD'].items()]
        g_pag_for = [x[1][0].item() for x in result['G_PAG_FOR'].items()]
        g_pag_fdr = [x[1][0].item() for x in result['G_PAG_FDR'].items()]

    return g_pag_list, g_pag_labels, gi_pag_list, gi_pag_labels, {
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
        g_pag_labels = [list([str(a) for a in x[1]]) for x in result['G_PAG_Label_List'].items()]
        g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list([str(a) for a in x[1]]) for x in result['Gi_PAG_Label_List'].items()]
        gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

        #print(true_labels, labels, g_pag_labels)
        found_correct_pag = bool(result['found_correct_pag'][0])
        #print(found_correct_pag)
        g_pag_shd = [x[1][0].item() for x in result['G_PAG_SHD'].items()]
        g_pag_for = [x[1][0].item() for x in result['G_PAG_FOR'].items()]
        g_pag_fdr = [x[1][0].item() for x in result['G_PAG_FDR'].items()]

    return g_pag_list, g_pag_labels, gi_pag_list, gi_pag_labels, {
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

def server_results_to_dataframe(server, labels, results):
    likelihood_ratio_tests = server.get_likelihood_ratio_tests()

    columns = ('ord', 'X', 'Y', 'S', 'pvalue')
    rows = []

    for test in likelihood_ratio_tests:
        s_labels_string = ','.join(sorted([str(labels.index(l)+1) for l in test.conditioning_set]))
        rows.append((len(test.conditioning_set), labels.index(test.v0)+1, labels.index(test.v1)+1, s_labels_string, test.p_val))

    df = pd.DataFrame(data=rows, columns=columns)
    return pl.from_pandas(df)

def test_dataset(true_pag, true_labels, dfs, df_msep=None):
    meta_dfs, meta_labels = zip(*[mxm_ci_test(d) for d in dfs])
    iod_result_fisher = run_pval_agg_iod(true_pag, true_labels, meta_dfs, meta_labels, ALPHA, PROCEDURE)

    server = fedci.Server(
        {
            str(i):fedci.Client(d) for i,d in enumerate(dfs, start=1)
        }
    )

    fedci_results = server.run()
    fedci_all_labels = sorted(list(server.schema.keys()))
    client_labels = {id: sorted(list(schema.keys())) for id, schema in server.client_schemas.items()}
    fedci_df = server_results_to_dataframe(server, fedci_all_labels, fedci_results)

    iod_result_fedci = run_riod(
        true_pag,
        true_labels,
        fedci_df.to_pandas(),
        fedci_all_labels,
        {str(i):sorted(d.columns) for i,d in enumerate(dfs, start=1)},
        ALPHA,
        PROCEDURE
    )

    if df_msep is not None:
        _fedci_df = fedci_df
        label_mapping = {str(i):l for i,l in enumerate(true_labels, start=1)}
        _fedci_df = _fedci_df.with_columns(
            pl.col('X').cast(pl.Utf8).replace(label_mapping),
            pl.col('Y').cast(pl.Utf8).replace(label_mapping),
            pl.col('S').str.split(',').list.eval(pl.element().replace(label_mapping)).list.sort().list.join(','),
        )
        test_df = _fedci_df.select('ord', 'X', 'Y', 'S').join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)

        reverse_label_mapping = {l:str(i) for i,l in enumerate(true_labels, start=1)}
        test_df = test_df.with_columns(
            pl.col('X').replace(reverse_label_mapping).cast(pl.Int64),
            pl.col('Y').replace(reverse_label_mapping).cast(pl.Int64),
            pl.col('S').str.split(',').list.eval(pl.element().replace(reverse_label_mapping)).list.sort().list.join(','),
        )

        test_df = test_df.select(
            'ord', 'X', 'Y', 'S',
            pvalue=pl.when(pl.col('MSep')).then(pl.lit(0.999)).otherwise(pl.lit(0.001)).cast(pl.Float64)
        )

        test_df = test_df.sort('ord', 'X', 'Y', 'S')
        #pl.Config.set_tbl_rows(50)
        #print(test_df)
        #print(fedci_df.sort('ord', 'X', 'Y', 'S'))

        #iod_result_oracle = run_pval_agg_iod(true_pag, true_labels, [test_df.to_pandas()], [true_labels], ALPHA, PROCEDURE)
        #iod_result_oracle = run_riod(true_pag, true_labels, test_df.to_pandas(), true_labels, {'1': true_labels}, ALPHA, PROCEDURE)
        iod_result_oracle = run_riod(true_pag, true_labels, test_df.to_pandas(), true_labels, {str(i):sorted(d.columns) for i,d in enumerate(dfs, start=1)}, ALPHA, PROCEDURE)

        test_df = _fedci_df.join(df_msep, on=['ord', 'X', 'Y', 'S'], how='left', coalesce=True)
        test_df = test_df.with_columns(indep=pl.col('pvalue')>ALPHA)
        test_df = test_df.with_columns(faithful=pl.col('MSep')==pl.col('indep')).filter(~pl.col('faithful'))
        if len(test_df) > 0:
            print(test_df)


    return iod_result_fisher, iod_result_fedci, None if df_msep is None else iod_result_oracle


files = os.listdir(DATA_DIR)
print(len(files))
files = set([f.rpartition('-p')[0] for f in files])
files = sorted(files)
files = [DATA_DIR + '/' + f for f in files]

files = [f for f in files if f.endswith('-n')]


print(f'Found {len(files)} datasets')

pagids = [f.split('-')[1] for f in files]
from collections import Counter
print('Diff PAGs:', len(Counter(pagids)))

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

for file in files[147:]:

    print('Running for', file)

    dfs1 = []
    for _file in glob.glob(f'{file}-p1-*.parquet'):
        dfs1.append(pl.read_parquet(_file))
    dfs2 = []
    for _file in glob.glob(f'{file}-p2-*.parquet'):
        dfs2.append(pl.read_parquet(_file))

    p1_labels = dfs1[0].columns
    p2_labels = dfs2[0].columns
    intersect_labels = sorted(list(set(p1_labels)&set(p2_labels)))


    pag_id = int(file.split('-')[1])
    num_samples = int(file.split('-')[2])
    true_pag = truePAGs_map[pag_id]
    true_labels = ['A', 'B', 'C', 'D', 'E']
    not_shared_labels = sorted(list(set(true_labels) - set(intersect_labels)))

    df_msep = pl.read_parquet('./experiments/pag_msep/pag-{}.parquet'.format(pag_id))
    df_msep = df_msep.with_columns(pl.col('S').list.join(','))
    df_msep = df_msep.with_columns(
        ord=pl.when(
            pl.col('S').str.len_chars() == 0).then(
                pl.lit(0)
            ).otherwise(
                pl.col('S').str.count_matches(',') + 1
            )
    )

    test_results_fisher, test_results_fedci, test_results_oracle = test_dataset(true_pag, true_labels, dfs1+dfs2, df_msep)


    if test_results_oracle is None:
        oracle_pags, oracle_labels = [true_pag], [true_labels]
    else:
        oracle_pags, oracle_labels = test_results_oracle[0], test_results_oracle[1]

    #print(test_results_oracle[1])
    fedci_pag_list, fedci_pag_labels, fedci_pag_i_list, fedci_pag_i_labels, fedci_stats = test_results_fedci
    fisher_pag_list, fisher_pag_labels, fisher_pag_i_list, fisher_pag_i_labels, fisher_stats = test_results_fisher

    fedci_has_correct_pag = False
    fedci_shds = []
    for pred_pag, _labels in zip(fedci_pag_list, fedci_pag_labels):
        min_shd = 99
        for oracle_pag, _oracle_labels in zip(oracle_pags, oracle_labels):
            perm = [_oracle_labels.index(col) for col in _labels]
            pred_pag = np.array(pred_pag)[:, perm][perm, :]
            shd = np.sum(pred_pag != np.array(oracle_pag)).item()
            #print('Diff', shd)
            if shd < min_shd:
                min_shd = shd
        fedci_shds.append(min_shd)

    fisher_has_correct_pag = False
    fisher_shds = []
    for pred_pag, _labels in zip(fisher_pag_list, fisher_pag_labels):
        min_shd = 99
        for oracle_pag, _oracle_labels in zip(oracle_pags, oracle_labels):
            perm = [_oracle_labels.index(col) for col in _labels]
            #perm = [true_labels.index(col) for col in _labels]
            pred_pag = np.array(pred_pag)[:, perm][perm, :]
            shd = np.sum(pred_pag != np.array(oracle_pag)).item()
            if shd < min_shd:
                min_shd = shd
        fisher_shds.append(min_shd)

    result = {
        'file_id': file,
        'faithful': file.endswith('-g'),
        'num_samples': num_samples,
        'pag_id': pag_id,
        'num_clients': len(dfs1) + len(dfs2),
        'fedci_shd': fedci_shds,
        'fisher_shd': fisher_shds,
        'oracle_num_predictions': len(oracle_pags) if test_results_oracle is not None else -1,
        'fedci_num_predictions': len(fedci_pag_list),
        'fisher_num_predictions': len(fisher_pag_list),
    }

    #print(result)

    with open(LOGFILE, 'a') as f:
        result_str = json.dumps(result)
        f.write(result_str + '\n')
