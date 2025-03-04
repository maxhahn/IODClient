import polars as pl
import polars.selectors as cs
import hvplot
import hvplot.polars
import holoviews as hv

hvplot.extension('matplotlib')

# sudo apt update
# sudo apt-get install texlive

path =  './experiments/fed-v-fisher/*.ndjson'
df = pl.read_ndjson(path)

df = df.select('name', 'num_clients', 'num_samples', cs.ends_with('_p_values'))

df = df.with_columns(
    experiment_type=pl.col('name').str.slice(0,3),
    conditioning_type=pl.col('name').str.slice(4)
)

print(df.group_by('experiment_type', 'conditioning_type').agg(pl.len()).sort('len'))
#df = df.sort('experiment_type', 'conditioning_type')
df = df.explode('federated_p_values', 'fisher_p_values', 'baseline_p_values')


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

df_unpivot = df_unpivot.rename({'num_samples': '\# Samples', 'correlation': 'Correlation'})


import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,  # Use LaTeX for text rendering
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],  # Matches LaTeX default
# })

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,  # Use LaTeX for all text rendering
    "font.family": "serif",
    "pgf.rcfonts": False
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
for i in [1,3,5]:
    _plot = df_unpivot.filter(pl.col('num_clients') == i).sort(
        'Method', '\# Samples'
    ).hvplot.line(
        x='\# Samples',
        y='Correlation',
        alpha=0.85,
        ylim=(-0.01,1.01),
        width=400,
        height=400,
        by='Method',
        legend='bottom_right',
        #backend='matplotlib',
        xlabel=r'\# Samples',  # LaTeX-escaped #
        ylabel=r'Correlation',
        linestyle=['solid', 'dashed']
        #title=f'{"Client" if i == 1 else "Clients"}'
    )

    # Apply different line styles
    #_plot = _plot.opts(line_dash=['solid', 'dashed'])

    _render =  hv.render(_plot, backend='matplotlib')
    _render.savefig(f'images/test-{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)

    if plot is None:
        plot = _plot
    else:
        plot = plot + _plot

#plot = plot.opts(ylim=(-0.01,1.01), legend='bottom_left',)

# Convert HoloViews object to matplotlib figure
mpl_fig = hv.render(plot, backend='matplotlib')

# Adjust font sizes and styling
for ax, num_c in zip(mpl_fig.axes, [1,3,5]):
    ax.tick_params(labelsize=10)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    ax.set_title('{} Clients'.format(num_c), fontsize=14)
    #if ax.get_legend():
    #    ax.legend(fontsize=10)

# Save as SVG with LaTeX-rendered text
#mpl_fig.savefig('images/test.svg', format='svg', bbox_inches='tight', dpi=300)
mpl_fig.savefig('images/test.pgf', format='pgf', bbox_inches='tight', dpi=300)
plt.close(mpl_fig)  # Clean up


# latex font update
"""
sudo updmap-sys --setoption pdftexDownloadBase14 true
sudo updmap-sys
"""
