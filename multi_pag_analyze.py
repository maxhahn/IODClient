import polars as pl
import polars.selectors as cs

target_folder = 'shd'

df = pl.read_ndjson('./mixed_pag_results_non_faithful.json')

df = df.with_columns(
    found_correct_fedci=pl.col('fedci_shd').list.contains(0),
    found_correct_fisher=pl.col('fisher_shd').list.contains(0),
    num_pags_fedci=pl.col('fedci_shd').list.len(),
    num_pags_fisher=pl.col('fisher_shd').list.len(),
    sum_shd_fedci=pl.col('fedci_shd').list.sum(),
    sum_shd_fisher=pl.col('fisher_shd').list.sum(),
)

print(df)

print('Found correct')
print(df.select(cs.starts_with('found_correct_')).mean())

print('Avg Num Predictions')
print(df.select(cs.starts_with('num_pags_')).mean())

print('No Predictions')
print(df.select(cs.starts_with('num_pags_')==0).mean())

print('Avg. SHD')
print(df.select(cs.starts_with('num_pags') | cs.starts_with('sum')).sum().select(
    fedci=pl.col('sum_shd_fedci')/pl.col('num_pags_fedci'),
    fisher=pl.col('sum_shd_fisher')/pl.col('num_pags_fisher'),
))


import hvplot
import hvplot.polars
import holoviews as hv
import matplotlib.pyplot as plt

plt.rcParams.update({
    "svg.fonttype": "none"
})

hvplot.extension('matplotlib')


df_exp = df.unpivot(
    on=['fedci_shd', 'fisher_shd'],
    #index='test_id',
    value_name='SHD',
    variable_name='Method'
)

df_exp = df_exp.explode('SHD')

plot = df_exp.hvplot.hist(
    y='SHD',
    by='Method',
    alpha=0.5
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/{target_folder}/hist-comparison.svg', format='svg', bbox_inches='tight', dpi=300)


_df = df.filter(pl.col('num_pags_fisher')==0)

df_exp = _df.unpivot(
    on=['fedci_shd', 'fisher_shd'],
    #index='test_id',
    value_name='SHD',
    variable_name='Method'
)

df_exp = df_exp.explode('SHD')

plot = df_exp.hvplot.hist(
    y='SHD',
    #by='Method',
    alpha=0.5
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/{target_folder}/hist-where-fisher-failed.svg', format='svg', bbox_inches='tight', dpi=300)


_df = df.filter(pl.col('num_pags_fisher')>0)

df_exp = _df.unpivot(
    on=['fedci_shd', 'fisher_shd'],
    #index='test_id',
    value_name='SHD',
    variable_name='Method'
)

df_exp = df_exp.explode('SHD')

plot = df_exp.hvplot.hist(
    y='SHD',
    by='Method',
    alpha=0.5
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/{target_folder}/hist-where-both-predicted.svg', format='svg', bbox_inches='tight', dpi=300)


_df = df.with_columns(
    fedci_min_shd=pl.col('fedci_shd').list.min(),
    fisher_min_shd=pl.col('fisher_shd').list.min(),
)

_df = _df.with_columns(pl.col('fedci_min_shd', 'fisher_min_shd').fill_null(pl.lit(20)))

_df = _df.unpivot(
    on=['fedci_min_shd', 'fisher_min_shd'],
    #index='test_id',
    value_name='MIN_SHD',
    variable_name='Method'
)

plot = _df.hvplot.hist(
    y='MIN_SHD',
    by='Method',
    alpha=0.5
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/{target_folder}/hist-min-shd.svg', format='svg', bbox_inches='tight', dpi=300)


plot = _df.hvplot.box(
    y='MIN_SHD',
    by='Method'
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/{target_folder}/box-min-shd.svg', format='svg', bbox_inches='tight', dpi=300)


_df = df.filter(pl.col('num_pags_fisher')>0)

_df = _df.with_columns(
    fedci_min_shd=pl.col('fedci_shd').list.min(),
    fisher_min_shd=pl.col('fisher_shd').list.min(),
)

#_df = _df.with_columns(pl.col('fedci_min_shd', 'fisher_min_shd').fill_null(pl.lit(20)))

_df = _df.unpivot(
    on=['fedci_min_shd', 'fisher_min_shd'],
    #index='test_id',
    value_name='MIN_SHD',
    variable_name='Method'
)

plot = _df.hvplot.hist(
    y='MIN_SHD',
    by='Method',
    alpha=0.5
)

_render =  hv.render(plot, backend='matplotlib')
_render.savefig(f'images/{target_folder}/hist-min-shd-where-both-predicted.svg', format='svg', bbox_inches='tight', dpi=300)
