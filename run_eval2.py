import polars as pl
import polars.selectors as cs
import hvplot
import hvplot.polars

# wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
# tar -xvf geckodriver-v0.35.0-linux64.tar.gz
# mv geckodriver /usr/local/bin
# sudo apt update && sudo apt install firefox

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
df_fisher = get_correlation(df, identifiers, 'fisher_p_values', 'baseline_p_values').rename({'p_value_correlation': 'Meta Analysis'})


_df = df.select(identifiers).unique().join(
    df_fed, on=identifiers, how='left'
).join(
    df_fisher, on=identifiers, how='left'
)

df_unpivot = _df.unpivot(
    on=['Federated', 'Meta Analysis'],
    index=identifiers,
    value_name='correlation',
    variable_name='Method'
)

df_unpivot = df_unpivot.rename({'num_samples': '# Samples', 'correlation': 'Correlation'})

#hv.extension('matplotlib')
# plot = df_unpivot.sort(
#     'technique', 'num_clients', 'num_samples'
# ).hvplot.line(
#     x='num_samples',
#     y='correlation',
#     alpha=0.6,
#     #groupby=['num_clients', 'experiment_type', 'conditioning_type'],
#     ylim=(-0.01,1.01),
#     #width=400,
#     #height=400,
#     #subplots=True,
#     by='technique',
#     col='num_clients',
#     #row='conditioning_type',
#     #legend='bottom'
# )


plot = None
for i in [1,3,5]:
    _plot = df_unpivot.filter(pl.col('num_clients') == i).sort(
        'Method', '# Samples'
    ).hvplot.line(
        x='# Samples',
        y='Correlation',
        alpha=0.6,
        #groupby=['num_clients', 'experiment_type', 'conditioning_type'],
        ylim=(-0.01,1.01),
        width=400,
        height=400,
        #subplots=True,
        by='Method',
        #col='num_clients',
        #row='conditioning_type',
        legend='bottom_right',# False if i!=5 else 'bottom',
        #legend='bottom'
    )
    #_plot = _plot.opts(legend_offset=(200,0))
    if plot is None:
        plot = _plot
    else:
        plot = plot + _plot

hvplot.save(plot, 'images/test.html')


import holoviews as hv
import matplotlib.pyplot as plt

# Convert HoloViews object to matplotlib figure
mpl_fig = hv.render(plot, backend='matplotlib')

# Save as SVG
mpl_fig.savefig('images/test.svg', format='svg', bbox_inches='tight', dpi=300)
plt.close(mpl_fig)  # Clean up
