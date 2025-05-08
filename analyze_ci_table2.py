import polars as pl
import polars.selectors as cs

import os

dir = 'experiments/simulation/results5/'

#all_faithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/f2')]
#all_unfaithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/uf2')]

files = os.listdir(dir)

files_by_type = {}
for f in files:
    faithfulness_type = f.split('-')[-2]
    if faithfulness_type not in files_by_type:
        files_by_type[faithfulness_type] = []
    files_by_type[faithfulness_type].append(dir+f)
#files_faithful = [dir+f for f in files if '-g-' in f]
#files_unfaithful = [dir+f for f in files if '-g-' not in f]

#print(len(files_faithful), len(files_unfaithful))

dfs = []
for t,f in files_by_type.items():
    df = pl.read_parquet(f).with_columns(faithfulness=pl.lit(t), filename=pl.lit(f))
    dfs.append(df)

dfs = []
for f in files:
    #if '1745511692009-7-50000-0.25-g-' not in f:
    #    continue
    df = pl.read_parquet(dir+'/'+f).with_columns(filename=pl.lit(f))
    dfs.append(df)

#dfs = []
#if len(files_faithful) > 0:
#    df_faithful = pl.read_parquet(files_faithful).with_columns(faithful=True)
#    dfs.append(df_faithful)
# if len(files_unfaithful) > 0:
#     df_unfaithful = pl.read_parquet(files_unfaithful).with_columns(faithful=False)
#     dfs.append(df_unfaithful)

df = pl.concat(dfs)

three_tail_pags = [2, 16, 18, 19, 20, 23, 29, 31, 37, 42, 44, 53, 57, 58, 62, 64, 66, 69, 70, 72, 73, 74, 75, 79, 81, 82, 83, 84, 93, 98]
three_tail_pags = [t-1 for t in three_tail_pags]

df = df.with_columns(
    faithfulness=pl.col('filename').str.split('-').list.get(-2),
    num_samples=pl.col('filename').str.split('-').list.get(2).cast(pl.Int32),
    pag_id=pl.col('filename').str.split('-').list.get(1).cast(pl.Int32)
)

print('Num unique pags used', df['pag_id'].n_unique())

faithfulness_filter = None#'g'
#faithfulness_filter = 'g'
#faithfulness_filter = 'l'
#faithfulness_filter = 'gl'
#faithfulness_filter = 'n'


if faithfulness_filter is None:
    faithfulness_filter= 'all'
else:
    df = df.filter(pl.col('faithfulness') == faithfulness_filter)

#overlap = df.select(x=pl.col('X')+pl.col('Y')+pl.col('S').str.replace_all(',', ''))['x'].to_list()
#overlap = [set(list(v)) for v in overlap]
#overlap = set.intersection(*overlap)


df = df.with_columns(
    pvalue_diff_fedci_pooled=pl.col('pvalue_fedci')-pl.col('pvalue_pooled'),
    pvalue_diff_fisher_pooled=pl.col('pvalue_fisher')-pl.col('pvalue_pooled')
)

#print(df.columns)
pl.Config.set_tbl_rows(50)
#print(df.select('X', 'Y', 'S', 'filename', cs.contains('pvalue')).sort('pvalue_diff_fedci_pooled'))
#print(df.select('X', 'Y', 'S', 'filename', cs.contains('pvalue')).sort('diff_pvalue')[0].to_dict())

#df = pl.read_parquet(dir)

df = df.with_columns(
    correct_fisher=pl.col('MSep')==pl.col('indep_fisher'),
    correct_fedci=pl.col('MSep')==pl.col('indep_fedci'),
    correct_pooled=pl.col('MSep')==pl.col('indep_pooled'),
    correct_as_pooled_fisher=pl.col('indep_pooled')==pl.col('indep_fisher'),
    correct_as_pooled_fedci=pl.col('indep_pooled')==pl.col('indep_fedci'),
)

import hvplot
import hvplot.polars
import holoviews as hv
import matplotlib.pyplot as plt

plt.rcParams.update({
    "svg.fonttype": "none"
})

hvplot.extension('matplotlib')


_df = df.group_by(['num_samples', 'MSep', 'faithfulness']).agg((cs.contains('correct_')-cs.contains('pooled')).mean(), pl.len())
print(_df.sort(['num_samples', 'MSep', 'faithfulness']))
