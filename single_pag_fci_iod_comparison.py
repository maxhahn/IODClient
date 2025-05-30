import polars as pl
import polars.selectors as cs
import numpy as np

from collections import OrderedDict

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

# supress R log
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

ro.r['source']('./single_pag_analysis.r')
run_fci_f = ro.globalenv['run_fci']
iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']
aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
run_ci_test_f = ro.globalenv['run_ci_test']

def run_pval_agg_iod(dfs, client_labels, alpha=0.05, procedure='original'):

    #ro.r['source']('./aggregation.r')
    #aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        #r_dfs = ro.ListVector(r_dfs)
        label_list = [ro.StrVector(v) for v in client_labels]

        #true_pag_np = np.array(true_pag)
        #r_matrix = ro.r.matrix(ro.FloatVector(true_pag_np.flatten()), nrow=len(true_labels), ncol=len(true_labels))
        #colnames = ro.StrVector(true_labels)

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

        #true_pag_np = np.array(true_pag)
        #r_matrix = ro.r.matrix(ro.FloatVector(true_pag_np.flatten()), nrow=len(true_labels), ncol=len(true_labels))
        #colnames = ro.StrVector(true_labels)

        result = iod_on_ci_data_f(label_list, suff_stat, alpha, procedure)

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


data_dir = 'experiments/datasets/data_slides'
#target_id = '1747098690714-slides-4000-g'

#target_id = '1746980535602-slides-8000-g'
#target_id = '1747095310771-slides-4000-g'
#target_id = '1747098690714-slides-4000-g'
#target_id = '1747157391970-slides-4000-g'
target_id = '1747234664372-slides-4000-g'


pvalue_results_dir = 'experiments/simulation/slides'

def test_dataset(file_id):

    df1 = pl.read_parquet(f'{data_dir}/{target_id}-p1.parquet')
    df2 = pl.read_parquet(f'{data_dir}/{target_id}-p2.parquet')

    #df1.write_csv('test1.csv')
    #df2.write_csv('test2.csv')

    # STEP 1: FCI FOR BOTH DFs

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        fci_result_df1 = run_fci_f(df1.to_pandas(), ro.StrVector(sorted(df1.columns)))
        fci_result_df2 = run_fci_f(df2.to_pandas(), ro.StrVector(sorted(df2.columns)))

    # LOAD PVALUE DF
    pvalue_df = pl.read_parquet(f'{pvalue_results_dir}/{target_id}-*(1_1)*.parquet')
    #pvalue_df.write_csv('pvals.csv')

    # STEP 2: IOD WITH META-ANALYSIS
    # -> Load results df and just get pvalues from there
    #pvalue_df_meta = pvalue_df.select('ord', 'X', 'Y', 'S', pvalue=pl.col('pvalue_fisher'))
    meta_dfs, meta_labels = zip(*[mxm_ci_test(d) for d in [df1, df2]])
    iod_result_fisher = run_pval_agg_iod(meta_dfs, meta_labels)

    # STEP 3: IOD WITH FEDCI
    # -> Load results df and just get pvalues from there
    pvalue_df_fedci = pvalue_df.select('ord', 'X', 'Y', 'S', pvalue=pl.col('pvalue_fedci'))
    fedci_labels = sorted(list(set(df1.columns)|set(df2.columns)))
    label_mapping = {l:str(i) for i,l in enumerate(fedci_labels, start=1)}
    pvalue_df_fedci = pvalue_df_fedci.with_columns(
        pl.col('X').cast(pl.Utf8).replace(label_mapping),
        pl.col('Y').cast(pl.Utf8).replace(label_mapping),
        pl.col('S').str.split(',').list.eval(pl.element().replace(label_mapping)).list.sort().list.join(','),
    )

    iod_result_fedci = run_riod(
        pvalue_df_fedci.to_pandas(),
        fedci_labels,
        {'1':sorted(df1.columns), '2':sorted(df2.columns)}
    )

    return (fci_result_df1, sorted(df1.columns)), (fci_result_df2, sorted(df2.columns)), iod_result_fisher, iod_result_fedci

import graphviz
arrow_type_lookup = {
        1: 'odot',
        2: 'normal',
        3: 'none'
    }
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
# g.pipe(format='png')

graph_dir = 'experiments/graphs'
def save_graph(graph, filename):
    png_bytes = graph.pipe(format='png')
    # Optionally, save manually without the .gv
    with open(f'{graph_dir}/{filename}.png', 'wb') as f:
        f.write(png_bytes)


target_ids = [target_id]

for target_id in target_ids:
    print('Testing', target_id)
    result_fci1, result_fci2, result_fisher, result_fedci = test_dataset(target_id)


    if len(result_fisher[0]) != 1 or len(result_fedci[0]) != 1:
        print(f'Fisher got {len(result_fisher[0])} results, Fedci got {len(result_fedci[0])} results. Skipping...')
        continue

    g_fci1 = data2graph(result_fci1[0], result_fci1[1])
    save_graph(g_fci1, f'fci1')
    g_fci2 = data2graph(result_fci2[0], result_fci2[1])
    save_graph(g_fci2, f'fci2')

    for i,(r,l) in enumerate(zip(result_fisher[0], result_fisher[1])):
        g_fisher = data2graph(r, l)
        save_graph(g_fisher, f'fisher-{i}')

    fci1_fisher_updated = (result_fisher[2][0], result_fisher[3][0])
    fci2_fisher_updated = (result_fisher[2][1], result_fisher[3][1])

    g_fci1 = data2graph(fci1_fisher_updated[0], fci1_fisher_updated[1])
    save_graph(g_fci1, f'fci1-fisher-updated')
    g_fci2 = data2graph(fci2_fisher_updated[0], fci2_fisher_updated[1])
    save_graph(g_fci2, f'fci2-fisher-updated')

    for i,(r,l) in enumerate(zip(result_fedci[0], result_fedci[1])):
        g_fedci = data2graph(r, l)
        save_graph(g_fedci, f'fedci-{i}')

    fci1_fedci_updated = (result_fedci[2][0], result_fedci[3][0])
    fci2_fedci_updated = (result_fedci[2][1], result_fedci[3][1])

    g_fci1 = data2graph(fci1_fedci_updated[0], fci1_fedci_updated[1])
    print(fci1_fedci_updated[0], g_fci1)
    save_graph(g_fci1, f'fci1-fedci-updated')
    g_fci2 = data2graph(fci2_fedci_updated[0], fci2_fedci_updated[1])
    save_graph(g_fci2, f'fci2-fedci-updated')
