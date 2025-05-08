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

df = df.filter(pl.col('num_samples') == 4000)

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

#print(df.filter(~pl.col('correct_fedci') & pl.col('correct_fisher')).select('filename', 'X','Y','S',cs.contains('pvalue')))

print(df.select(cs.starts_with('correct_')).mean())
print(df.group_by('faithfulness').agg(cs.starts_with('correct_').mean()).with_columns(diff=pl.col('correct_fedci')-pl.col('correct_fisher')).sort('faithfulness'))
print(df.group_by('faithfulness', 'MSep').agg(cs.starts_with('correct_').mean(), pl.len()).sort('faithfulness', 'MSep'))
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
import matplotlib.pyplot as plt

plt.rcParams.update({
    "svg.fonttype": "none"
})

hvplot.extension('matplotlib')

print('=== Now plotting scatter of pvalues')

_df = df.filter(pl.col('MSep'))
_df = _df.sample(min(len(_df), 200))

_df = _df.rename({
    'pvalue_fedci': 'Federated',
    'pvalue_fisher': 'Meta-Analysis',
})

plot = _df.hvplot.scatter(
    x='pvalue_pooled',
    y=['Federated', 'Meta-Analysis'],
    alpha=0.8,
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
_render.savefig(f'images/ci_table/scatter-indep-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)

_df = df.filter(~pl.col('MSep'))
_df = _df.sample(min(len(_df), 2000))

_df = _df.rename({
    'pvalue_fedci': 'Federated',
    'pvalue_fisher': 'Meta-Analysis',
})

plot = _df.hvplot.scatter(
    x='pvalue_pooled',
    y=['Federated', 'Meta-Analysis'],
    alpha=0.8,
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
_render.savefig(f'images/ci_table/scatter-dep-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)


_df = df
_df = _df.sample(min(len(_df), 2000))

_df = _df.rename({
    'pvalue_fedci': 'Federated',
    'pvalue_fisher': 'Meta-Analysis',
})

plot = _df.hvplot.scatter(
    x='pvalue_pooled',
    y=['Federated', 'Meta-Analysis'],
    alpha=0.8,
    ylim=(-0.001,0.1),
    xlim=(-0.001,0.1),
    width=400,
    height=400,
    #by='Method',
    legend='bottom_right',
    #backend='matplotlib',
    s=4000,
    xlabel=r'Baseline p-value',  # LaTeX-escaped #
    ylabel=r'Predicted p-value',
    marker=['v', '^'],
    #linestyle=['dashed', 'dotted']
    #title=f'{"Client" if i == 1 else "Clients"}'
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/ci_table/scatter-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)




print('=== Now calculating correlation of pvalues')

def get_correlation(df, identifiers, colx, coly):
    _df = df

    if len(identifiers) == 0:
        _df = _df.with_columns(p_value_correlation=pl.corr(colx, coly))
    else:
        _df = _df.group_by(
            identifiers
        ).agg(
            p_value_correlation=pl.corr(colx, coly)
        )

    _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None).fill_null(pl.lit(1)))

    return _df

#identifiers = ['ord', 'X', 'Y', 'S']
identifiers = []
identifiers = ['faithfulness', 'MSep']

df_fed = get_correlation(df, identifiers, 'pvalue_fedci', 'pvalue_pooled').rename({'p_value_correlation': 'Federated'})
df_fisher = get_correlation(df, identifiers, 'pvalue_fisher', 'pvalue_pooled').rename({'p_value_correlation': 'Meta-Analysis'})

if len(identifiers) == 0:
    _df = pl.concat([df_fed[0].select('Federated'), df_fisher[0].select('Meta-Analysis')], how='horizontal')
else:
    _df = df.select(identifiers).unique().join(
        df_fed, on=identifiers, how='left'
    ).join(
        df_fisher, on=identifiers, how='left'
    )
print('Showing correlation to baseline')
print(_df.with_columns(diff=pl.col('Federated')-pl.col('Meta-Analysis')).sort('diff'))

# TODO: visualize?
#
#

###
# Boxplot of p value diff
###

print('=== Now plotting boxplots of pvalue difference to pooled baseline')

_df = df

# p-value approx = 0 appears to be way too easy. Make sure pvalues on pooled data are at least slightly away from 0 for plot
l1 = len(_df)
_df = _df.filter(pl.col('pvalue_pooled') > 1e-8)
print(f'All data: Removed {100-(len(_df)/l1)*100:.3f}% ({l1-len(_df)} samples) of data because of proximity to 0')

_df = _df.rename({
    'pvalue_diff_fedci_pooled': 'Federated',
    'pvalue_diff_fisher_pooled': 'Meta-Analysis',
})

plot = _df.hvplot.box(
    y=['Federated', 'Meta-Analysis'],
    ylabel=r'p-value Difference',
    xlabel='Method',
    #ylim=(-0.000005,0.000005)
    showfliers=False
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/ci_table/pval_diff_box-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)


_df = df.filter(~pl.col('MSep'))

# Normalize (not really working)
# _df = _df.with_columns(max_diff=pl.max_horizontal('pvalue_diff_fedci_pooled', 'pvalue_diff_fisher_pooled'))
# max_diff = _df['max_diff'].max()
# _df = _df.with_columns(
#     pvalue_diff_fedci_pooled=pl.col('pvalue_diff_fedci_pooled') / max_diff,
#     pvalue_diff_fisher_pooled=pl.col('pvalue_diff_fisher_pooled') / max_diff,
# )

#_df = _df.filter((pl.col('pvalue_diff_fedci_pooled') != 0) & (pl.col('pvalue_diff_fisher_pooled') != 0))

# p-value approx = 0 appears to be way too easy. Make sure pvalues on pooled data are at least slightly away from 0 for plot
l1 = len(_df)
_df = _df.filter(pl.col('pvalue_pooled') > 1e-8)
print(f'Not MSep plot: Removed {100-(len(_df)/l1)*100:.3f}% ({l1-len(_df)} samples) of data because of proximity to 0')

_df = _df.rename({
    'pvalue_diff_fedci_pooled': 'Federated',
    'pvalue_diff_fisher_pooled': 'Meta-Analysis',
})

plot = _df.hvplot.box(
    y=['Federated', 'Meta-Analysis'],
    ylabel=r'p-value Difference',
    xlabel='Method',
    #ylim=(-0.000005,0.000005)
    showfliers=False
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/ci_table/pval_diff_box_dep-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)



# MSep Box
_df = df.filter(pl.col('MSep'))

_df = _df.rename({
    'pvalue_diff_fedci_pooled': 'Federated',
    'pvalue_diff_fisher_pooled': 'Meta-Analysis',
})

plot = _df.hvplot.box(
    y=['Federated', 'Meta-Analysis'],
    ylabel=r'p-value Difference',
    xlabel='Method',
    showfliers=False
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/ci_table/pval_diff_box_indep-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)


###
# Decision agreements
###

print('=== Now calculating % of decision agreements')

def get_accuracy(df, identifiers, colx, coly, alpha):
    df = df.with_columns(
        tp=(pl.col(colx) < alpha) & (pl.col(coly) < alpha),
        tn=(pl.col(colx) > alpha) & (pl.col(coly) > alpha),
        fp=(pl.col(colx) < alpha) & (pl.col(coly) > alpha),
        fn=(pl.col(colx) > alpha) & (pl.col(coly) < alpha),
    )
    if len(identifiers) == 0:
        df = df.select((pl.col('tp')+pl.col('tn')).mean().alias('accuracy'))
    else:
        df = df.group_by(identifiers).agg((pl.col('tp')+pl.col('tn')).mean().alias('accuracy'))
    return df

alpha = 0.05

#identifiers = ['ord', 'X', 'Y', 'S']
identifiers = []
identifiers = ['faithfulness', 'MSep']


df_fed = get_accuracy(df, identifiers, 'pvalue_fedci', 'pvalue_pooled', alpha).rename({'accuracy': 'Federated'})
df_fisher = get_accuracy(df, identifiers, 'pvalue_fisher', 'pvalue_pooled', alpha).rename({'accuracy': 'Meta-Analysis'})

if len(identifiers) == 0:
    _df = pl.concat([df_fed, df_fisher], how='horizontal')
else:
    _df = df.select(identifiers).unique().join(
        df_fed, on=identifiers, how='left'
    ).join(
        df_fisher, on=identifiers, how='left'
    )
print('Showing % of decision agreements')
print(_df.with_columns(diff=pl.col('Federated')- pl.col('Meta-Analysis')).sort('diff'))


if faithfulness_filter == 'all':

    _df = _df.rename({
      'Federated': 'F',
      'Meta-Analysis': 'MA'
    })

    _df = _df.with_columns(pl.col('faithfulness').replace_strict({
        'g': 'global only',
        'gl': 'faithful',
        'l': 'local only',
        'n': 'unfaithful'
    }))

    __df = _df.filter(pl.col('MSep'))
    plot = __df.sort('faithfulness').hvplot.bar(
        x='faithfulness',
        y=['F', 'MA'],
        xlabel='Performance Of Method Under Different Faithfulness Conditions',
        #rot=30
    )
    _render =  hv.render(plot, backend='matplotlib')
    _render.savefig(f'images/ci_accuracy/bar-indep-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)

    __df = _df.filter(~pl.col('MSep'))
    plot = __df.sort('faithfulness').hvplot.bar(
        x='faithfulness',
        y=['F', 'MA'],
        xlabel='Performance Of Method Under Different Faithfulness Conditions',
        #rot=30
    )
    _render =  hv.render(plot, backend='matplotlib')
    _render.savefig(f'images/ci_accuracy/bar-dep-{faithfulness_filter}.svg', format='svg', bbox_inches='tight', dpi=300)
