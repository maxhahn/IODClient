import polars as pl
import polars.selectors as cs

pl.Config.set_tbl_rows(100)

# TODO: get containsTheTrueGraph in temp.r and return it

json_file = "experiments/simulation/pvalagg_vs_fedci_new/*.ndjson"
#schema = pl.read_ndjson("experiments/simulation/pvalagg_vs_fedci/1738847995495-0-300-3.ndjson").schema
df = pl.read_ndjson(json_file)#, schema=schema)
print(df['metrics_fedci'][0])
print(len(df))

# TODO: get number of predictions for each sample

#print(df)
#df = pl.scan_ndjson(json_file).with_columns(pl.col('metrics_fedci').fill_null(pl.struct())).collect()
grouping_keys = ['num_samples', 'num_clients', 'alpha']

df = df.drop('split_percentiles')


df = df.with_columns(has_prediction_fedci=pl.col('metrics_fedci').struct.field('SHD').list.len() > 0)
df = df.with_columns(has_prediction_pvalagg=pl.col('metrics_pvalagg').struct.field('SHD').list.len() > 0)

dfx = df
dfx = dfx.with_columns(
    found_correct_pag_fedci=pl.col('metrics_fedci').struct.field('found_correct'),
    found_correct_pag_pvalagg=pl.col('metrics_pvalagg').struct.field('found_correct')
)

print(dfx.select('has_prediction_fedci', 'has_prediction_pvalagg').mean())
print(dfx.group_by('has_prediction_fedci', 'has_prediction_pvalagg').len())
print(dfx.select('found_correct_pag_fedci', 'found_correct_pag_pvalagg').mean())

# only where data exists
#df = df.filter(pl.col('has_prediction_fedci') & pl.col('has_prediction_pvalagg'))

df = df.with_columns(
    pl.col('metrics_fedci').struct.unnest().name.prefix('fedci_'),
    pl.col('metrics_pvalagg').struct.unnest().name.prefix('pvalagg_'),
).drop('metrics_fedci', 'metrics_pvalagg')

df = df.drop((cs.starts_with('fedci_') | cs.starts_with('pvalagg_')) - (cs.contains('_MEAN_') | cs.contains('_MIN_') | cs.contains('_MAX_')))

print(df.head())

asd
df = df.with_row_index()


df = df.unpivot(
    on=cs.starts_with('metric_'),
    index=['index', 'alpha', 'num_samples', 'num_clients', 'has_prediction_fedci', 'has_prediction_pvalagg'],
    variable_name='metric'
)
df = df.with_columns(
    name=pl.col('metric').str.split('_').list.get(0),
    type=pl.col('metric').str.split('_').list.get(1),
    metric=pl.col('metric').str.split('_').list.get(2)
)
#print(df.head())


#df = df.join(df_fedci, on=['index'], how='left')
#df = df.join(df_pvalagg, on=['index'], how='left')

#df = df.with_columns()
#print(df)


import hvplot.polars
import hvplot

#print(df.select(cs.starts_with('metric_') & cs.ends_with('_mean') & cs.contains('F1_Score')).describe())

#metric_FDR
#metric_FOR
#metric_SHD
# plot = df.sort(
#     'num_clients', 'num_samples'
# ).hvplot.box(
#     by='name',
#     y='metric_SHD_mean',
#     #alpha=0.7,
#     #ylim=(-0.1,1.1),
#     #xlim=(-0.1,1.1),
#     height=400,
#     width=400,
#     row='num_clients',
#     col='num_samples',
#     #groupby=['num_clients', 'num_samples'],
#     subplots=True,
#     #widget_location='bottom'
#     )

# plot = df.sort(
#     'num_clients', 'num_samples'
# ).hvplot.scatter(
#     x='metric_fedci_SHD_min',
#     y='metric_pvalagg_SHD_min',
#     #alpha=0.7,
#     #ylim=(-0.1,1.1),
#     #xlim=(-0.1,1.1),
#     height=400,
#     width=400,
#     row='num_clients',
#     col='num_samples',
#     #groupby=['num_clients', 'num_samples'],
#     subplots=True,
#     #widget_location='bottom'
#     )
#

plot = df.sort(
    'num_clients', 'num_samples', 'name'
).hvplot.box(
    by='name',
    #y=['metric_fedci_SHD_mean', 'metric_pvalagg_SHD_mean'],
    y='value',
    #alpha=0.7,
    #ylim=(-0.1,1.1),
    #xlim=(-0.1,1.1),
    height=800,
    width=800,
    row='num_clients',
    col='num_samples',
    groupby=['metric', 'type'],
    subplots=True,
    #widget_location='bottom'
    )

hvplot.save(plot, 'images/test.html')


plot = df.sort(
    'num_clients', 'num_samples', 'name'
).hvplot.box(
    by=['name', 'has_prediction_pvalagg'],
    #y=['metric_fedci_SHD_mean', 'metric_pvalagg_SHD_mean'],
    y='value',
    #alpha=0.7,
    #ylim=(-0.1,1.1),
    #xlim=(-0.1,1.1),
    height=800,
    width=800,
    row='num_clients',
    col='num_samples',
    groupby=['metric', 'type'],
    subplots=True,
    #widget_location='bottom'
    )

hvplot.save(plot, 'images/test2.html')

plot = df.sort(
    'num_clients', 'num_samples', 'name'
).hvplot.box(
    by=['name', 'has_prediction_fedci'],
    #y=['metric_fedci_SHD_mean', 'metric_pvalagg_SHD_mean'],
    y='value',
    #alpha=0.7,
    #ylim=(-0.1,1.1),
    #xlim=(-0.1,1.1),
    height=800,
    width=800,
    row='num_clients',
    col='num_samples',
    groupby=['metric', 'type'],
    subplots=True,
    #widget_location='bottom'
    )

hvplot.save(plot, 'images/test3.html')
