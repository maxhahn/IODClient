import polars as pl
import polars.selectors as cs

import os

dir = 'experiments/simulation/results3/'

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
    if '1745511692009-7-50000-0.25-g-' not in f:
        continue
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

df = df.with_columns(faithfulness=pl.col('filename').str.split('-').list.get(-2))

print(df.columns)
pl.Config.set_tbl_rows(50)
print(df.select('X', 'Y', 'S', cs.contains('pvalue')))

#df = pl.read_parquet(dir)

df = df.with_columns(
    correct_fisher=pl.col('MSep')==pl.col('indep_fisher'),
    correct_fedci=pl.col('MSep')==pl.col('indep_fedci'),
    correct_pooled=pl.col('MSep')==pl.col('indep_pooled'),
)

print(df.filter(~pl.col('correct_fedci') & pl.col('correct_fisher')).select('filename', 'X','Y','S',cs.contains('pvalue')))

print(df.select(cs.starts_with('correct_')).mean())
print(df.group_by('faithfulness').agg(cs.starts_with('correct_').mean()).with_columns(diff=pl.col('correct_fedci')-pl.col('correct_fisher')).sort('faithfulness'))
print(df.group_by('faithfulness', 'MSep').agg(cs.starts_with('correct_').mean(), pl.len()).with_columns(diff=pl.col('correct_fedci')-pl.col('correct_fisher')).sort('faithfulness', 'MSep'))
#print(df.group_by('ord', 'X', 'Y', 'S').agg(pl.col('MSep').first(), cs.starts_with('correct_').sum(), pl.len()).sort('ord', 'X', 'Y', 'S'))
#print(df.group_by('ord', 'X', 'Y', 'S').agg(pl.col('MSep').first(), cs.starts_with('correct_').sum() / pl.len(), pl.len()).sort('ord', 'X', 'Y', 'S'))

# TODO: Visualizations of pvalues:
# - scatter plot
# - corr plot?
# - difference between fisher/fedci to pooled as boxplot
# Do visualizations for MSep = True / MSep = False (or use indep_pooled)

import hvplot
import hvplot.polars
import holoviews as hv

hvplot.extension('matplotlib')

_df = df.filter(pl.col('MSep'))

plot = _df.hvplot.scatter(
    x='pvalue_pooled',
    y=['pvalue_fedci', 'pvalue_fisher'],
    alpha=0.6,
    ylim=(-0.01,1.01),
    xlim=(-0.01,1.01),
    width=400,
    height=400,
    #by='Method',
    legend='bottom_right',
    #backend='matplotlib',
    #s=5000,
    xlabel=r'Baseline p-value',  # LaTeX-escaped #
    ylabel=r'Predicted p-value',
    marker=['v', '^'],
    #linestyle=['dashed', 'dotted']
    #title=f'{"Client" if i == 1 else "Clients"}'
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/ci_table/scatter-indep.svg', format='svg', bbox_inches='tight', dpi=300)

_df = df.filter(~pl.col('MSep'))

plot = _df.hvplot.scatter(
    x='pvalue_pooled',
    y=['pvalue_fedci', 'pvalue_fisher'],
    alpha=0.6,
    ylim=(-0.001,0.1),
    xlim=(-0.001,0.1),
    width=400,
    height=400,
    #by='Method',
    legend='bottom_right',
    #backend='matplotlib',
    #s=5000,
    xlabel=r'Baseline p-value',  # LaTeX-escaped #
    ylabel=r'Predicted p-value',
    marker=['v', '^'],
    #linestyle=['dashed', 'dotted']
    #title=f'{"Client" if i == 1 else "Clients"}'
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/ci_table/scatter-dep.svg', format='svg', bbox_inches='tight', dpi=300)



# def get_correlation(df, identifiers, colx, coly):
#     _df = df

#     _df = _df.group_by(
#         identifiers
#     ).agg(
#         p_value_correlation=pl.corr(colx, coly)
#     )

#     _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))

#     return _df

# identifiers = ['ord', 'X', 'Y', 'S']

# df_fed = get_correlation(df, identifiers, 'pvalue_fedci', 'pvalue_pooled').rename({'p_value_correlation': 'Federated'})
# df_fisher = get_correlation(df, identifiers, 'pvalue_fisher', 'pvalue_pooled').rename({'p_value_correlation': 'Meta-Analysis'})

# _df = df.select(identifiers).unique().join(
#     df_fed, on=identifiers, how='left'
# ).join(
#     df_fisher, on=identifiers, how='left'
# )

# _df = _df.with_columns(diff=pl.col('Federated')-pl.col('Meta-Analysis')).sort('diff')
# print(_df)
