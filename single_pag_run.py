import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd

import fedci

from collections import OrderedDict
import itertools
import os

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

import graphviz
arrow_type_lookup = {
        1: 'odot',
        2: 'normal',
        3: 'none'
    }

# supress R log
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

ro.r['source']('./single_pag_analysis.r')
run_fci_f = ro.globalenv['run_fci']
iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']
aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
run_ci_test_f = ro.globalenv['run_ci_test']
get_data_f = ro.globalenv['get_data_for_single_pag']

ALPHA = 0.05
NUM_SAMPLES = [15_000,20_000,30_000,50_000]
SPLITS = [[1,1], [2,1], [1,2], [3,1], [1,3]]
SEEDS = [x+200_204 for x in range(100_000)]

DF_MSEP = pl.read_parquet(
    'experiments/pag_msep/pag-slides.parquet'
).with_columns(
    pl.col('S').list.join(',')
).with_columns(
    ord=pl.when(
        pl.col('S').str.len_chars() == 0).then(
            pl.lit(0)
        ).otherwise(
            pl.col('S').str.count_matches(',') + 1
        )
)

graph_dir = 'experiments/graphs'
data_dir = 'experiments/datasets/data_slides'
#target_id = '1747098690714-slides-4000-g'

#target_id = '1746980535602-slides-8000-g'
#target_id = '1747095310771-slides-4000-g'
#target_id = '1747098690714-slides-4000-g'
#target_id = '1747157391970-slides-4000-g'
target_id = '1747234664372-slides-4000-g'


pvalue_results_dir = 'experiments/simulation/slides'
target_ids = [target_id]

partition_1_labels = ['A', 'C', 'D', 'E']
partition_2_labels = ['A', 'B', 'C', 'E']
intersection_labels = sorted(list(set(partition_1_labels)&set(partition_2_labels)))


def get_data(num_samples, seed):
    dat = get_data_f(num_samples, 'continuous', 0.2, seed)
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(dat[0])
    df = pl.from_pandas(df)
    return df

def run_pval_agg_iod(dfs, client_labels, alpha=0.05, procedure='original'):

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        label_list = [ro.StrVector(v) for v in client_labels]

        result = aggregate_ci_results_f(label_list, r_dfs, alpha, procedure)

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

def run_riod(df, labels, client_labels, alpha=0.05, procedure='original'):
    # let index start with 1
    df.index += 1

    label_list = [ro.StrVector(v) for v in client_labels.values()]
    users = list(client_labels.keys())

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        suff_stat = [
            ('citestResults', ro.conversion.get_conversion().py2rpy(df)),
            ('all_labels', ro.StrVector(labels)),
        ]
        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)

        result = iod_on_ci_data_f(label_list, suff_stat, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
        g_pag_labels = [list([str(a) for a in x[1]]) for x in result['G_PAG_Label_List'].items()]
        g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
        gi_pag_labels = [list([str(a) for a in x[1]]) for x in result['Gi_PAG_Label_List'].items()]
        gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

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

def mxm_ci_test(df):
    df = df.with_columns(cs.string().cast(pl.Categorical()))
    df = df.to_pandas()
    with (ro.default_converter + pandas2ri.converter).context():
        df_r = ro.conversion.get_conversion().py2rpy(df)
        result = run_ci_test_f(df_r, 999, "./examples/", 'dummy')
        df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
        labels = list(result['labels'])
    return df_pvals, labels

def server_results_to_dataframe(server, labels, results):
    likelihood_ratio_tests = server.get_likelihood_ratio_tests()

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
    return pl.from_pandas(df)

def test_dataset(dfs):
    # STEP 1: FCI FOR BOTH DFs

    fci_results = []
    for df in dfs:
        with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
            fci_result_df = run_fci_f(df.to_pandas(), ro.StrVector(sorted(df.columns)))
        fci_results.append(fci_result_df)

    # STEP 2: IOD WITH META-ANALYSIS
    # -> Load results df and just get pvalues from there
    #pvalue_df_meta = pvalue_df.select('ord', 'X', 'Y', 'S', pvalue=pl.col('pvalue_fisher'))
    meta_dfs, meta_labels = zip(*[mxm_ci_test(d) for d in dfs])
    iod_result_fisher = run_pval_agg_iod(meta_dfs, meta_labels)

    # STEP 3: IOD WITH FEDCI
    # -> Load results df and just get pvalues from there
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
        fedci_df.to_pandas(),
        fedci_all_labels,
        {str(i):sorted(d.columns) for i,d in enumerate(dfs, start=1)}
    )

    return (fci_results, [sorted(d.columns) for d in dfs]), iod_result_fisher, iod_result_fedci


def data2graph(data, labels):
    graph = graphviz.Digraph(format='png')
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            arrhead = data[i][j]
            arrtail = data[j][i]
            if data[i][j] == 1:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
            elif data[i][j] == 2:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
            elif data[i][j] == 3:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])

    return graph

def save_graph(graph, identifier, filename):
    png_bytes = graph.pipe(format='png')
    if not os.path.exists(f'{graph_dir}/{identifier}'):
        os.makedirs(f'{graph_dir}/{identifier}')
    # Optionally, save manually without the .gv
    with open(f'{graph_dir}/{identifier}/{filename}.png', 'wb') as f:
        f.write(png_bytes)

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
    is_faithful = faithful_df.select(faithful_count=(pl.col('indep') == pl.col('MSep')))['faithful_count'].sum() == len(faithful_df)
    return is_faithful, result_df

def split_data(df, splits):
    dfs = []

    is_faithful, overlap_df = test_faithfulness(df, DF_MSEP)
    if not is_faithful:
        return dfs

    for split in splits:
        _dfs = []
        for i in range(1,len(split)+1):
            from_idx = int(sum(split[:i-1])/sum(split) * len(df))
            to_idx = int(sum(split[:i])/sum(split) * len(df))
            #print(i, len(df), from_idx, to_idx)
            df_i = df[from_idx:to_idx]
            if i % 2 == 0:
                df_i = df_i.select(partition_1_labels)
            else:
                df_i = df_i.select(partition_2_labels)
            _dfs.append(df_i)

        is_faithful1 = test_faithfulness(pl.concat(_dfs[0::2]), DF_MSEP, overlap_df)
        is_faithful2 = test_faithfulness(pl.concat(_dfs[1::2]), DF_MSEP, overlap_df)
        if not is_faithful1 or not is_faithful2:
            continue
        dfs.append((split,_dfs))
    return dfs

for seed in SEEDS:
    for num_samples in NUM_SAMPLES:
        df = get_data(num_samples, seed)
        dfs = split_data(df, SPLITS)

        if len(dfs) == 0:
            print(f'No faithful data for {num_samples} samples and seed {seed}')
            continue

        print(f'Found faithful data for {num_samples} samples and seed {seed}: {len(dfs)}')
        with open('./faithful_finds.csv', 'a') as f:
            f.write(f'{num_samples},{seed},{SPLITS}')

        for split, _dfs in dfs:
            identifier = f'{num_samples}-{seed}-{"_".join([str(s) for s in split])}'

            result_fci, result_fisher, result_fedci = test_dataset(_dfs)

            if len(result_fisher[0]) != 1 or len(result_fedci[0]) != 1:
                print(f'... Fisher got {len(result_fisher[0])} results, Fedci got {len(result_fedci[0])} results. Skipping...')
                continue

            if np.array_equal(np.array(result_fisher[0][0]), np.array(result_fedci[0][0])):
                print('... Fisher and Fedci got same result. Skipping...')
                continue

            print('!!! Found differing PAGs')

            for i, (_split, (fci_adj_mat, fci_labels)) in enumerate(zip(split,result_fci), start=1):
                g_fci = data2graph(fci_adj_mat, fci_labels)
                save_graph(g_fci, identifier, f'fci-{str(_split)}-{i}')

            for i, (_adj_mat, _labels) in enumerate(zip(result_fisher[0], result_fisher[1]), start=1):
                g_fisher = data2graph(_adj_mat, _labels)
                save_graph(g_fisher, identifier, f'fisher-{i}')

            for i, (_adj_mat, _labels) in enumerate(zip(result_fisher[2], result_fisher[3]), start=1):
                fci_fisher_updated = (_adj_mat, _labels)
                g_fci = data2graph(*fci_fisher_updated)
                save_graph(g_fci, identifier, f'fci-{i}-fisher-updated')

            for i, (_adj_mat, _labels) in enumerate(zip(result_fedci[0], result_fedci[1]), start=1):
                g_fedci = data2graph(_adj_mat, _labels)
                save_graph(g_fedci, identifier, f'fedci-{i}')

            for i, (_adj_mat, _labels) in enumerate(zip(result_fedci[2], result_fedci[3]), start=1):
                fci_fedci_updated = (_adj_mat, _labels)
                g_fci = data2graph(*fci_fedci_updated)
                save_graph(g_fci, identifier, f'fci-{i}-fedci-updated')
