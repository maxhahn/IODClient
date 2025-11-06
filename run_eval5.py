import polars as pl
import polars.selectors as cs
import hvplot
import hvplot.polars
import holoviews as hv

import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.latex.preamble": r"\usepackage{amsmath}"
})

hvplot.extension('matplotlib')

path = './experiments/fed-v-fisher*/*.ndjson'
df = pl.read_ndjson(path)

print(df.columns)

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
df = df.filter(~(pl.col('name').str.contains('\(')))

print(df.group_by('experiment_type', 'conditioning_type').agg(pl.len()).sort('len'))
df = df.explode('federated_p_values', 'fisher_p_values', 'baseline_p_values')

num_samples = 2000
plot = None
_df = df.filter(pl.col('num_samples') == num_samples)
_df = _df.rename({
    'federated_p_values': 'Federated',
    'fisher_p_values': 'Meta-Analysis'
})
for i in [1,3,5,7]:
    __df = _df.filter(pl.col('num_clients') == i)
    __df = __df.sample(min(1_000, len(__df)))

    _plot = __df.hvplot.scatter(
        x='baseline_p_values',
        y=['Federated', 'Meta-Analysis'],
        alpha=0.6,
        ylim=(-0.01,1.01),
        xlim=(-0.01,1.01),
        width=400,
        height=400,
        legend='bottom_right',
        s=5000,
        xlabel=r'Baseline p-value',
        ylabel=r'Predicted p-value',
        marker=['v', '^'],
    )

    _render = hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/pval_comp/scatter-c{i}-samples{num_samples}.pdf', format='pdf', bbox_inches='tight', dpi=300)

    _plot = __df.hvplot.scatter(
        x='baseline_p_values',
        y=['Federated', 'Meta-Analysis'],
        alpha=0.6,
        ylim=(-0.01,0.1),
        xlim=(-0.01,0.1),
        width=400,
        height=400,
        legend='bottom_right',
        s=5000,
        xlabel=r'Baseline p-value',
        ylabel=r'Predicted p-value',
        marker=['v', '^'],
    )

    _render = hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/pval_comp/scatter-c{i}-samples{num_samples}-small.pdf', format='pdf', bbox_inches='tight', dpi=300)

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

    _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))
    _df = _df.join(df_correlation_fix2, on=identifiers, how='left')
    _df = _df.with_columns(pl.coalesce(['p_value_correlation', 'correlation_fix'])).drop('correlation_fix')

    _df = _df.with_columns(pl.col('p_value_correlation').fill_nan(None))
    assert _df['p_value_correlation'].null_count() == 0, 'NaN in correlations'

    return _df

identifiers = ['num_clients', 'num_samples']
df_fed = get_correlation(df, identifiers, 'federated_p_values', 'baseline_p_values').rename({'p_value_correlation': 'Federated'})
df_fisher = get_correlation(df, identifiers, 'fisher_p_values', 'baseline_p_values').rename({'p_value_correlation': 'Meta-Analysis'})

_df = df.select(identifiers).unique().join(df_fed, on=identifiers, how='left').join(df_fisher, on=identifiers, how='left')

df_unpivot = _df.unpivot(
    on=['Federated', 'Meta-Analysis'],
    index=identifiers,
    value_name='correlation',
    variable_name='Method'
)

df_unpivot = df_unpivot.rename({'num_samples': '# Samples', 'correlation': 'Correlation'})

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
        xlabel=r'# Samples',
        ylabel=r'Correlation',
        linestyle=['dashed', 'dotted']
    )

    _render = hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/pval_comp/correlation-c{i}{current_experiment_type}.pdf', format='pdf', bbox_inches='tight', dpi=300)

    if plot is None:
        plot = _plot
    else:
        plot = plot + _plot

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

df_fed = get_accuracy(df, identifiers, 'federated_p_values', 'baseline_p_values', alpha).rename({'accuracy': 'Federated'})
df_fisher = get_accuracy(df, identifiers, 'fisher_p_values', 'baseline_p_values', alpha).rename({'accuracy': 'Meta-Analysis'})

_df = df.select(identifiers).unique().join(df_fed, on=identifiers, how='left').join(df_fisher, on=identifiers, how='left')

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
        xlabel=r'# Samples',
        ylabel=r'Accuracy',
        linestyle=['dashed', 'dotted']
    )

    _render = hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/pval_comp/accuracy-c{i}{current_experiment_type}.pdf', format='pdf', bbox_inches='tight', dpi=300)
