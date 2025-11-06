import polars as pl
import polars.selectors as cs

import os

#dir = 'experiments/simulation/results7'
#dir = 'experiments/simulation/slides'
dir = 'experiments/simulation/single_data_ratio' # USE SINGLE DATA TO PLOT DIFF TO REAL P VALUE

#all_faithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/f2')]
#all_unfaithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/uf2')]

files = os.listdir(dir)

# files_by_type = {}
# for f in files:
#     print(f)
#     faithfulness_type = f.split('-')[-2]
#     if faithfulness_type not in files_by_type:
#         files_by_type[faithfulness_type] = []
#     files_by_type[faithfulness_type].append(dir+f)
#files_faithful = [dir+f for f in files if '-g-' in f]
#files_unfaithful = [dir+f for f in files if '-g-' not in f]

#print(len(files_faithful), len(files_unfaithful))

# dfs = []
# for t,f in files_by_type.items():
#     df = pl.read_parquet(f).with_columns(faithfulness=pl.lit(t), filename=pl.lit(f))
#     dfs.append(df)

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
    faithfulness=pl.col('filename').str.split('-').list.get(-3),
    num_samples=pl.col('filename').str.split('-').list.get(2).cast(pl.Int32),
    split_sizes=pl.col('filename').str.split('-(').list.get(1).str.split(')-').list.get(0).str.replace_all('[\(\)]', '').str.split('_').cast(pl.List(pl.Int32)),
    pag_id=pl.col('filename').str.split('-').list.get(1)#.cast(pl.Int32)
)
df = df.with_columns(num_splits=pl.col('split_sizes').list.len())

df = df.with_columns(
    split_ratio=pl.format('{}:{}', pl.col('split_sizes').list.get(0), pl.col('split_sizes').list.get(1))
)

print('Num unique pags used', df['pag_id'].n_unique())

faithfulness_filter = None#'g'
#faithfulness_filter = 'g'
#faithfulness_filter = 'l'
#faithfulness_filter = 'gl'
#faithfulness_filter = 'n'

print(df.group_by('num_splits','num_samples').len())


#df = df.filter(~(pl.col('X')+pl.col('Y')+pl.col('S')).str.contains_any(['B', 'D']))

#df = df.filter(pl.col('num_samples') == 4000)
#df = df.filter(pl.col('num_splits') == 2)
#df = df.filter(pl.col('split_sizes').list.max() == 4)

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

__df = df.filter((pl.col('MSep'))&(pl.col('indep_fedci')!=pl.col('indep_fisher')))
print(__df.select('MSep','filename'))


#print(df.filter(~pl.col('correct_fedci') & pl.col('correct_fisher')).select('filename', 'X','Y','S',cs.contains('pvalue')))

print(df.select(cs.starts_with('correct_')).mean())
print(df.group_by('faithfulness').agg(cs.starts_with('correct_').mean()).with_columns(diff=pl.col('correct_fedci')-pl.col('correct_fisher')).sort('faithfulness'))
print(df.group_by('num_splits', 'MSep').agg(cs.starts_with('correct_').mean(), pl.len()).sort('num_splits', 'MSep'))
#print(df.group_by('ord', 'X', 'Y', 'S').agg(pl.col('MSep').first(), cs.starts_with('correct_').sum(), pl.len()).sort('ord', 'X', 'Y', 'S'))
#print(df.group_by('ord', 'X', 'Y', 'S').agg(pl.col('MSep').first(), cs.starts_with('correct_').sum() / pl.len(), pl.len()).sort('ord', 'X', 'Y', 'S'))

print(df.group_by('ord', 'X', 'Y', 'S', 'MSep').agg(pl.col('correct_fedci', 'correct_fisher').mean(), pl.len()).sort('ord', 'X', 'Y', 'S'))
# TODO: Visualizations of pvalues:
# - scatter plot
# - corr plot?
# - difference between fisher/fedci to pooled as boxplot
# Do visualizations for MSep = True / MSep = False (or use indep_pooled)
#
print(df.select(pl.col('pvalue_diff_fisher_pooled', 'pvalue_diff_fedci_pooled').abs().min().name.suffix('_MIN'), pl.col('pvalue_diff_fisher_pooled', 'pvalue_diff_fedci_pooled').abs().max().name.suffix('_MAX')))

import hvplot
import hvplot.polars
import holoviews as hv
import matplotlib.pyplot as plt

plt.rcParams.update({
    "svg.fonttype": "none"
})

hvplot.extension('matplotlib')

_df = df

_df = _df.with_columns(
    max_pvalue_diff=pl.max_horizontal(pl.col('pvalue_diff_fisher_pooled').abs(), pl.col('pvalue_diff_fedci_pooled').abs())
)
_df = _df.with_columns(
    max_pvalue_diff_per_test=pl.max('max_pvalue_diff').over('num_samples', 'num_splits', 'ord', 'X', 'Y', 'S')
)
_df = _df.with_columns(
    adjusted_pvalue_diff_fisher=pl.col('pvalue_diff_fisher_pooled')/pl.col('max_pvalue_diff_per_test'),
    adjusted_pvalue_diff_fedci=pl.col('pvalue_diff_fedci_pooled')/pl.col('max_pvalue_diff_per_test')
)


_df = _df.with_columns(
    logratio_fedci=pl.col('pvalue_fedci').log(10) - pl.col('pvalue_pooled').log(10),
    logratio_fisher=pl.col('pvalue_fisher').log(10) - pl.col('pvalue_pooled').log(10)
)


#print(_df.select(pl.col('max_pvalue_diff').max(),pl.col('pvalue_diff_fisher_pooled', 'pvalue_diff_fedci_pooled').min().name.suffix('_MIN'), pl.col('pvalue_diff_fisher_pooled', 'pvalue_diff_fedci_pooled').max().name.suffix('_MAX')))
#print(_df.select(pl.col('adjusted_pvalue_fisher', 'adjusted_pvalue_fedci').max().name.suffix('_MAX')))
#print(_df.filter(pl.col('adjusted_pvalue_fisher')>1).select('adjusted_pvalue_fisher',pl.col('pvalue_diff_fisher_pooled', 'pvalue_diff_fedci_pooled')))

if False:
    _df = _df.rename({
        'adjusted_pvalue_diff_fedci': 'Federated',
        'adjusted_pvalue_diff_fisher': 'Meta-Analysis',
    })
else:
    _df = _df.rename({
        'logratio_fedci': 'Federated',
        'logratio_fisher': 'Meta-Analysis',
    })


_df = _df.unpivot(
    on=['Federated', 'Meta-Analysis'],
    index=['num_samples', 'num_splits', 'split_sizes', 'split_ratio'],
    value_name='p-value Difference',
    variable_name='Method'
)

_df = _df.with_columns(pl.col('Method').replace_strict({
    'Federated': 'F',
    'Meta-Analysis': 'MA'
}))

_df = _df.sort('Method', 'num_samples', 'num_splits')

"""

for split_ratio in df['split_ratio'].unique().to_list():

    __df = _df.filter(pl.col('split_ratio')==split_ratio)

    plot = __df.hvplot.box(
        #y='p-value Difference',# 'Meta-Analysis'],
        y='p-value Difference',
        by=['Method', 'num_samples'],
        #y='Meta-Analysis',# 'Meta-Analysis'],
        #by=['test_id', 'Method'],
        #ylabel='Normalized Difference in p-value',
        ylabel='Log-ratio of p-values',
        xlabel='Method, # Samples',
        #ylim=(-1,1),
        showfliers=False
    )

    _render =  hv.render(plot, backend='matplotlib')
    _render.savefig(f'images/pval_diff/pval_adjusted_diff_to_pooled_box_by_samples-{split_ratio}parts-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)

    plot = __df.hvplot.box(
        #y='p-value Difference',# 'Meta-Analysis'],
        #y=['Federated', 'Meta-Analysis'],
        y='p-value Difference',
        by=['Method', 'num_samples'],
        #y='Meta-Analysis',# 'Meta-Analysis'],
        #by=['test_id', 'Method'],
        #ylabel='Normalized Difference in p-value',
        ylabel='Log-ratio of p-values',
        xlabel='Method, # Samples',
        #ylim=(-1,1),
        #showfliers=False
    )

    _render =  hv.render(plot, backend='matplotlib')
    _render.savefig(f'images/pval_diff/pval_adjusted_diff_to_pooled_box_by_samples-fliers-{split_ratio}parts-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)
"""

__df = _df.filter(pl.col('num_splits')==10)

plot = __df.hvplot.box(
    #y='p-value Difference',# 'Meta-Analysis'],
    y='p-value Difference',
    by=['Method', 'split_ratio'],
    #y='Meta-Analysis',# 'Meta-Analysis'],
    #by=['test_id', 'Method'],
    #ylabel='Normalized Difference in p-value',
    ylabel='Log-ratio of p-values',
    xlabel='Method, # Samples',
    #ylim=(-1,1),
    showfliers=False
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/pval_diff/pval_adjusted_diff_to_pooled_box_by_ratio-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)

plot = __df.hvplot.box(
    #y='p-value Difference',# 'Meta-Analysis'],
    #y=['Federated', 'Meta-Analysis'],
    y='p-value Difference',
    by=['Method', 'split_ratio'],
    #y='Meta-Analysis',# 'Meta-Analysis'],
    #by=['test_id', 'Method'],
    #ylabel='Normalized Difference in p-value',
    ylabel='Log-ratio of p-values',
    xlabel='Method, # Samples',
    #ylim=(-1,1),
    #showfliers=False
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/pval_diff/pval_adjusted_diff_to_pooled_box_by_ratio-fliers-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)
