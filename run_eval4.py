import polars as pl
import polars.selectors as cs
import hvplot
import hvplot.polars
import holoviews as hv

hvplot.extension('matplotlib')

"""
sudo apt update
sudo apt install texlive-full
"""
# latex font update
"""
sudo updmap-sys --setoption pdftexDownloadBase14 true
sudo updmap-sys
"""

# sudo apt update
# sudo apt-get install texlive

#path =  './experiments/fed-v-fisher/*.ndjson'
path =  './experiments/fed-v-fisher*/*.ndjson'
#path =  './experiments/fed-v-fisher-final/*.ndjson'
df = pl.read_ndjson(path)

#df.write_ndjson('./experiments/fed-v-fisher-final/results1.ndjson')


print(df.columns)
#df = df.with_columns(max_split_percentage=pl.col('split_percentages').list.max())

df = df.select('name', 'num_clients', 'num_samples', cs.ends_with('_p_values'))

df = df.with_columns(
    experiment_type=pl.col('name').str.slice(0,3),
    conditioning_type=pl.col('name').str.slice(4)
)

experiment_types = ['M-O', 'C-B', 'B-O', 'C-O', 'C-M']
type_idx = -1
if type_idx >= 0:
    current_experiment_type = experiment_types[type_idx]
    df = df.filter(pl.col('experiment_type') == current_experiment_type)
else:
    current_experiment_type = ''

df = df.filter(pl.col('num_samples') < 4000)
#df = df.filter(pl.col('num_client') <= 5)
df = df.filter(~(pl.col('name').str.contains('\(')))

print(df.group_by('experiment_type', 'conditioning_type').agg(pl.len()).sort('len'))
#df = df.sort('experiment_type', 'conditioning_type')
df = df.explode('federated_p_values', 'fisher_p_values', 'baseline_p_values')


num_samples = 2000
plot = None
_df = df.filter(pl.col('num_samples') == num_samples)
for i in [1,3,5,7]:

    _plot = _df.filter(pl.col('num_clients') == i).hvplot.scatter(
        x='baseline_p_values',
        y=['federated_p_values', 'fisher_p_values'],
        alpha=0.3,
        ylim=(-0.01,1.01),
        width=400,
        height=400,
        #by='Method',
        #legend='bottom_right',
        #backend='matplotlib',
        #s=20,
        xlabel=r'Baseline p-value',  # LaTeX-escaped #
        ylabel=r'Predicted p-value',
        marker=['v', '^'],
        #linestyle=['dashed', 'dotted']
        #title=f'{"Client" if i == 1 else "Clients"}'
    )

    # Apply different line styles
    #_plot = _plot.opts(line_dash=['solid', 'dashed'])

    _render =  hv.render(_plot, backend='matplotlib')
    #_render.savefig(f'images/correlation-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
    _render.savefig(f'images/scatter-c{i}-samples{num_samples}.svg', format='svg', bbox_inches='tight', dpi=300)

    # if plot is None:
    #     plot = _plot
    # else:
    #     plot = plot + _plot

def get_correlation(df, identifiers, colx, coly):
    _df = df

    df_correlation_fix = df.with_columns(correct=(pl.col(colx) - pl.col(coly)).round(8) == 0)
    df_correlation_fix = df_correlation_fix.group_by(identifiers).agg(pl.all('correct'))
    df_correlation_fix = df_correlation_fix.filter(pl.col('correct')).drop('correct')
    df_correlation_fix = df_correlation_fix.with_columns(correlation_fix=pl.lit(1.0))

    df_correlation_fix2 = df.with_columns(
        pl.n_unique(colx, coly).over(identifiers).name.suffix('_nunique')
    )
    df_correlation_fix2 = df_correlation_fix2.filter((pl.col(f'{colx}_nunique') == 1) | (pl.col(f'{coly}_nunique') == 1))
    df_correlation_fix2 = df_correlation_fix2.drop(f'{colx}_nunique', f'{coly}_nunique')
    df_correlation_fix2 = df_correlation_fix2.group_by(identifiers).agg(
        mean_correctness=(pl.col(colx)==pl.col(coly)).mean(),
        mean_difference_p_value=(pl.col(colx).mean()-pl.col(coly).mean()).abs()
    )
    df_correlation_fix2 = df_correlation_fix2.with_columns(
        correlation_fix=((pl.col('mean_difference_p_value') < 1e-4) & (pl.col('mean_correctness') > 0.9)).cast(pl.Float64)
    )
    df_correlation_fix2 = df_correlation_fix2.drop('mean_difference_p_value', 'mean_correctness')

    _df = _df.group_by(
        identifiers
    ).agg(
        p_value_correlation=pl.corr(colx, coly)
    )

    _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))
    _df = _df.join(df_correlation_fix, on=identifiers, how='left')
    _df = _df.with_columns(pl.coalesce(['p_value_correlation', 'correlation_fix'])).drop('correlation_fix')

    #dfx = _df.filter(pl.col('p_value_correlation').is_null())

    _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))
    _df = _df.join(df_correlation_fix2, on=identifiers, how='left')
    _df = _df.with_columns(pl.coalesce(['p_value_correlation', 'correlation_fix'])).drop('correlation_fix')

    _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))

    #print(_df.join(dfx, on=['name', 'num_clients', 'num_samples'], how='semi'))
    assert _df['p_value_correlation'].null_count() == 0, 'NaN in correlations'

    return _df

identifiers = [
    #'name',
    'num_clients',
    'num_samples',
    #'experiment_type',
    #'conditioning_type'
]
df_fed = get_correlation(df, identifiers, 'federated_p_values', 'baseline_p_values').rename({'p_value_correlation': 'Federated'})
df_fisher = get_correlation(df, identifiers, 'fisher_p_values', 'baseline_p_values').rename({'p_value_correlation': 'Meta-Analysis'})


_df = df.select(identifiers).unique().join(
    df_fed, on=identifiers, how='left'
).join(
    df_fisher, on=identifiers, how='left'
)

df_unpivot = _df.unpivot(
    on=['Federated', 'Meta-Analysis'],
    index=identifiers,
    value_name='correlation',
    variable_name='Method'
)

df_unpivot = df_unpivot.rename({'num_samples': '# Samples', 'correlation': 'Correlation'})


import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,  # Use LaTeX for text rendering
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],  # Matches LaTeX default
# })

plt.rcParams.update({
    #"pgf.texsystem": "pdflatex",
    #"text.usetex": True,  # Use LaTeX for all text rendering
    #"font.family": "serif",
    #"pgf.rcfonts": False,
    "svg.fonttype": "none"
})


# Set up matplotlib to use LaTeX-style text rendering
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "text.latex.preamble": r"\usepackage{amsmath}"
# })

print(df_unpivot.head())

plot = None
for i in [1,3,5,7]:
    _plot = df_unpivot.filter(pl.col('num_clients') == i).sort(
        'Method', '# Samples'
    ).hvplot.line(
        x='# Samples',
        y='Correlation',
        alpha=0.85,
        ylim=(-0.01,1.01),
        width=400,
        height=400,
        by='Method',
        legend='bottom_right',
        #backend='matplotlib',
        xlabel=r'# Samples',  # LaTeX-escaped #
        ylabel=r'Correlation',
        linestyle=['dashed', 'dotted']
        #title=f'{"Client" if i == 1 else "Clients"}'
    )

    # Apply different line styles
    #_plot = _plot.opts(line_dash=['solid', 'dashed'])

    _render =  hv.render(_plot, backend='matplotlib')
    #_render.savefig(f'images/correlation-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
    _render.savefig(f'images/correlation-c{i}{current_experiment_type}.svg', format='svg', bbox_inches='tight', dpi=300)

    if plot is None:
        plot = _plot
    else:
        plot = plot + _plot


# --------------------------------

def get_accuracy(df, identifiers, colx, coly, alpha):
    df = df.with_columns(
        tp=(pl.col(colx) < alpha) & (pl.col(coly) < alpha),
        tn=(pl.col(colx) > alpha) & (pl.col(coly) > alpha),
        fp=(pl.col(colx) < alpha) & (pl.col(coly) > alpha),
        fn=(pl.col(colx) > alpha) & (pl.col(coly) < alpha),
    )

    df = df.group_by(identifiers).agg((pl.col('tp')+pl.col('tn')).mean().alias('accuracy'))
    return df

alpha = 0.05

identifiers = [
    #'name',
    'num_clients',
    'num_samples',
    #'experiment_type',
    #'conditioning_type'
]
df_fed = get_accuracy(df, identifiers, 'federated_p_values', 'baseline_p_values', alpha).rename({'accuracy': 'Federated'})
df_fisher = get_accuracy(df, identifiers, 'fisher_p_values', 'baseline_p_values', alpha).rename({'accuracy': 'Meta-Analysis'})


_df = df.select(identifiers).unique().join(
    df_fed, on=identifiers, how='left'
).join(
    df_fisher, on=identifiers, how='left'
)

df_unpivot = _df.unpivot(
    on=['Federated', 'Meta-Analysis'],
    index=identifiers,
    value_name='accuracy',
    variable_name='Method'
)

df_unpivot = df_unpivot.rename({'num_samples': '# Samples', 'accuracy': 'Accuracy'})

for i in [1,3,5,7]:
    _plot = df_unpivot.filter(pl.col('num_clients') == i).sort(
        'Method', '# Samples'
    ).hvplot.line(
        x='# Samples',
        y='Accuracy',
        alpha=0.85,
        ylim=(-0.01,1.01),
        width=400,
        height=400,
        by='Method',
        legend='bottom_right',
        #backend='matplotlib',
        xlabel=r'# Samples',  # LaTeX-escaped #
        ylabel=r'Accuracy',
        linestyle=['dashed', 'dotted']
        #title=f'{"Client" if i == 1 else "Clients"}'
    )

    # Apply different line styles
    #_plot = _plot.opts(line_dash=['solid', 'dashed'])

    #plt.rcParams['svg.fonttype'] = 'none'

    _render =  hv.render(_plot, backend='matplotlib')
    #_render.savefig(f'images/accuracy-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
    #_render.savefig(f'images/accuracy-c{i}.png', format='png', bbox_inches='tight', dpi=300)
    _render.savefig(f'images/accuracy-c{i}{current_experiment_type}.svg', format='svg', bbox_inches='tight', dpi=300)
