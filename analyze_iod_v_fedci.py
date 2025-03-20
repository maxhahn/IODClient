import polars as pl
import polars.selectors as cs

pl.Config.set_tbl_rows(100)

# TODO: get containsTheTrueGraph in temp.r and return it

json_file = "experiments/simulation/iod_comparison/*.ndjson"
schema = pl.read_ndjson("experiments/simulation/iod_comparison/1742481776605-0-10000-2.ndjson").schema
df = pl.read_ndjson(json_file, schema=schema)


#df = df.filter(pl.col('num_samples') < 50_000)

print(len(df))



# TODO: get number of predictions for each sample

#print(df)
#df = pl.scan_ndjson(json_file).with_columns(pl.col('metrics_fedci').fill_null(pl.struct())).collect()
grouping_keys = ['num_clients', 'num_samples']

#df = df.drop('split_percentiles')

df = df.with_columns(max_split_percentile=pl.col('split_percentiles').list.max())
df = df.with_columns(max_split_percentile_bucket=pl.col('max_split_percentile').qcut(10))


df = df.with_columns(num_prediction_fedci=pl.col('metrics_fedci').struct.field('SHD').list.len())
df = df.with_columns(num_prediction_fedci_ot=pl.col('metrics_fedci').struct.field('SHD').list.len())
df = df.with_columns(num_prediction_fisher=pl.col('metrics_fisher').struct.field('SHD').list.len())
df = df.with_columns(num_prediction_fisher_ot=pl.col('metrics_fisher_ot').struct.field('SHD').list.len())

df = df.with_columns(has_prediction_fedci=pl.col('num_prediction_fedci') > 0)
df = df.with_columns(has_prediction_fedci_ot=pl.col('num_prediction_fedci_ot') > 0)
df = df.with_columns(has_prediction_fisher=pl.col('num_prediction_fisher') > 0)
df = df.with_columns(has_prediction_fisher_ot=pl.col('num_prediction_fisher_ot') > 0)

dfx = df
dfx = dfx.with_columns(
    found_correct_pag_fedci=pl.col('metrics_fedci').struct.field('found_correct'),
    found_correct_pag_fedci_ot=pl.col('metrics_fedci_ot').struct.field('found_correct'),
    found_correct_pag_fisher=pl.col('metrics_fisher').struct.field('found_correct'),
    found_correct_pag_fisher_ot=pl.col('metrics_fisher_ot').struct.field('found_correct')
)

print(dfx.select(cs.starts_with('has_prediction_')).mean())
print(dfx.group_by('num_clients', 'num_samples').agg(cs.starts_with('has_prediction_').mean()).sort('num_clients', 'num_samples'))
#print(dfx.group_by(cs.starts_with('has_prediction_')).len())
print(dfx.select(cs.starts_with('found_correct_')).mean())

# only where data exists
#df = df.filter(pl.col('has_prediction_fedci') & pl.col('has_prediction_pvalagg'))

df = df.with_columns(
    pl.col('metrics_fedci').struct.unnest().name.prefix('fedci_'),
    pl.col('metrics_fedci_ot').struct.unnest().name.prefix('fedci_ot'),
    pl.col('metrics_fisher').struct.unnest().name.prefix('fisher'),
    pl.col('metrics_fisher_ot').struct.unnest().name.prefix('fisher_ot'),
).drop('metrics_fedci', 'metrics_fedci_ot', 'metrics_fisher', 'metrics_fisher_ot')

df = df.drop((cs.starts_with('fedci_') | cs.starts_with('iod_')) - (cs.contains('_MEAN_') | cs.contains('_MIN_') | cs.contains('_MAX_')))

#print(df.head())

dfx = df.group_by('num_clients', 'num_samples', 'max_split_percentile_bucket').agg(cs.ends_with('MIN_SHD').mean())
print(dfx.sort('num_clients', 'num_samples', 'max_split_percentile_bucket'))

dfx = df.group_by('num_clients', 'num_samples').agg(cs.ends_with('MIN_SHD').mean())
print(dfx.sort('num_clients', 'num_samples'))

#dfx = df.group_by('num_samples', 'num_clients').agg(cs.ends_with('MIN_SHD').mean())
#print(dfx.sort('num_samples', 'num_clients'))

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

"""
Plot Ideas - each also per approach:
 - correct_pag_hits per sample size
 - number_of_predictions per sample size ?
 - SHD per sample size

 - How good is the SHD per algorithm
 - How good is the SHD per algorithm when others did not predict anything
 - How good is the SHD when all have a prediction
"""


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
