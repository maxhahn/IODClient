import polars as pl
import polars.selectors as cs

pl.Config.set_tbl_rows(100)

# TODO: get containsTheTrueGraph in temp.r and return it

#json_file = "experiments/simulation/iod_comparison/*.ndjson"
json_file = "experiments/simulation/iod_comparison_final/*.ndjson"
#json_file = "experiments/simulation/iod_comparison_single_pag/*.ndjson"
#json_file = "experiments/simulation/iod_comparison_inbalance/*.ndjson"
#json_file = "experiments/simulation/iod_comparison_find_correct/*.ndjson"
#json_file = "experiments/simulation/iod_comparison_inbalance2/*.ndjson"
#json_file = "experiments/simulation/results5a/*-g-*.ndjson"
#json_file = "experiments/simulation/single_dataa/*-g-*.ndjson"
#json_file = "experiments/simulation/slidesa/*-g-*.ndjson"
json_file = "experiments/simulation/results7a/*-g-*.ndjson"

metric = 'MIN_SHD'
#metric = 'MIN_FDR'
#metric = 'MIN_FDR'

schema = pl.read_ndjson("experiments/simulation/iod_comparison_final/*.ndjson").schema
df = pl.read_ndjson(json_file, schema=schema)

print(df.columns)

#df = df.filter(pl.col('num_samples') < 50_000)

print(len(df))
print(df.group_by('num_clients', 'num_samples', 'pag_id').len().sort('num_clients', 'num_samples', 'pag_id'))
print(df.group_by('num_clients', 'num_samples', 'pag_id').len()['len'].value_counts())

df = df.with_columns(max_split_percentile=pl.col('split_percentiles').list.max())
#df = df.with_columns(max_split_percentile_bucket=pl.col('max_split_percentile').qcut(5))
df = df.with_columns(max_split_percentile_bucket=pl.col('max_split_percentile').cut([i/10 for i in range(11)]))

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
print(dfx.group_by(cs.starts_with('has_prediction_')).len().sort(cs.starts_with('has_prediction_')))
print(dfx.select(cs.starts_with('found_correct_')).mean())

# only where data exists
#df = df.filter(pl.col('has_prediction_fedci') & pl.col('has_prediction_pvalagg'))

df = df.with_columns(
    pl.col('metrics_fedci').struct.unnest().name.prefix('fedci_'),
    pl.col('metrics_fedci_ot').struct.unnest().name.prefix('fedci_ot_'),
    pl.col('metrics_fisher').struct.unnest().name.prefix('fisher_'),
    pl.col('metrics_fisher_ot').struct.unnest().name.prefix('fisher_ot_'),
).drop('metrics_fedci', 'metrics_fedci_ot', 'metrics_fisher', 'metrics_fisher_ot')

df = df.drop((cs.starts_with('fedci_') | cs.starts_with('fisher_')) - (cs.contains('_MEAN_') | cs.contains('_MIN_') | cs.contains('_MAX_')))

#print(df.head())

df = df.with_columns(cs.ends_with('_SHD')/20) # normalize to values between 0 and 1 (divide by number of edges of full graph with 5 nodes)

dfx = df.group_by('num_clients', 'num_samples', 'max_split_percentile_bucket').agg(cs.ends_with(metric).mean())
print(dfx.sort('num_clients', 'num_samples', 'max_split_percentile_bucket'))

dfx = df.group_by('num_clients', 'num_samples').agg(cs.ends_with(metric).mean())
print(dfx.sort('num_clients', 'num_samples'))

#dfx = df.group_by('num_samples', 'num_clients').agg(cs.ends_with('MIN_SHD').mean())
#print(dfx.sort('num_samples', 'num_clients'))

dfx = df.filter(pl.all_horizontal(cs.starts_with('has_prediction_')))
dfx = dfx.group_by('num_clients', 'num_samples').agg(cs.ends_with(metric).mean())
print(dfx.sort('num_clients', 'num_samples'))


###
# PLOTS
###

import matplotlib.pyplot as plt
import hvplot.polars
import hvplot as hv
import seaborn as sns

hv.extension('matplotlib')

# --- plot fedci v fisher

def plot_fed_v_fisher_lines(df):

    for i in [2,4,8]:
        df_plot = df
        df_plot = df_plot.filter(pl.all_horizontal(cs.starts_with('has_prediction_')))
        df_plot = df_plot.filter(pl.col('num_clients')==i)
        if len(df_plot) == 0:
            continue
        df_plot = df_plot.group_by('num_samples').agg(pl.all().mean())

        df_plot = df_plot.unpivot(
            on=[f'fedci_{metric}', f'fisher_{metric}'],
            index=['num_clients', 'num_samples'],
            value_name=metric
        )

        df_plot = df_plot.with_columns(pl.col('variable').replace({
            f'fedci_{metric}': 'fedci',
            f'fisher_{metric}': 'fisher',
            f'fedci_ot_{metric}': 'fedci OT',
            f'fisher_ot_{metric}': 'fisher OT'
        }))

        plot = df_plot.sort(
            'num_clients', 'num_samples'
        ).hvplot.line(
            x='num_samples',
            y=metric,
            by='variable',
            #alpha=0.7,
            #ylim=(-0.1,1.1),
            #xlim=(-0.1,1.1),
            #height=400,
            #width=400,
            linestyle=['dashed', 'dotted']
            )

        _render = hv.render(plot, backend='matplotlib')
        _render.savefig(f'images/fedci_v_fisher/{metric}_c{i}.svg', format='svg', bbox_inches='tight', dpi=300)

    for i in [2,4,8]:
        df_plot = df
        df_plot = df_plot.filter(pl.all_horizontal(cs.starts_with('has_prediction_')))
        df_plot = df_plot.filter(pl.col('num_clients')==i)
        if len(df_plot) == 0:
            continue
        df_plot = df_plot.group_by('num_samples').agg(pl.all().mean())


        df_plot = df_plot.unpivot(
            on=[f'fedci_{metric}', f'fisher_{metric}', f'fedci_ot_{metric}', f'fisher_ot_{metric}'],
            index=['num_clients', 'num_samples'],
            value_name=metric
        )

        df_plot = df_plot.with_columns(pl.col('variable').replace({
            f'fedci_{metric}': 'fedci',
            f'fisher_{metric}': 'fisher',
            f'fedci_ot_{metric}': 'fedci OT',
            f'fisher_ot_{metric}': 'fisher OT'
        }))

        plot = df_plot.sort(
            'num_clients', 'num_samples'
        ).hvplot.line(
            x='num_samples',
            y=metric,
            by='variable',
            alpha=0.8,
            #ylim=(-0.1,1.1),
            #xlim=(-0.1,1.1),
            #height=400,
            #width=400,
            linestyle=['solid', 'dashed', 'dotted', 'dashdot']
            )

        _render = hv.render(plot, backend='matplotlib')
        _render.savefig(f'images/fedci_v_fisher/with_ot_{metric}_c{i}.svg', format='svg', bbox_inches='tight', dpi=300)





#plot_fed_v_fisher_lines(df)




# ---

import numpy as np
import holoviews as hv
import hvplot.pandas  # Ensure hvplot is available
import polars as pl
import re

from bokeh.io import export_svgs

#hv.extension('bokeh')
hv.extension('matplotlib')

# Function to extract numeric midpoints
def extract_midpoint(category_str):
    match = re.findall(r"[-+]?\d*\.\d+|\d+", category_str)  # Extract numbers
    if len(match) == 2:  # If it's a range like "(0.8, 0.9]"
        return (float(match[0]) + float(match[1])) / 2  # Compute midpoint
    return None  # Fallback case (shouldn't happen if categories are formatted correctly)


def plot_imbalance_over_split_percentages(df, ncs=[2,4,8], nss=[500,1000,2500,5000]):
    # Convert categorical intervals to numeric midpoints
    df_plot = df
    df_plot = df_plot.group_by('num_clients', 'num_samples', 'max_split_percentile_bucket').agg(
        cs.ends_with(metric).mean()
    )

    df_plot = df_plot.unpivot(
        on=[f'fedci_{metric}', f'fisher_{metric}'],
        index=['num_clients', 'num_samples', 'max_split_percentile_bucket']
    )

    # Convert category to numerical midpoints
    df_plot = df_plot.with_columns(
        pl.col('max_split_percentile_bucket').map_elements(extract_midpoint).alias('split_midpoint')
    )

    for nc in ncs:
        for ns in nss:
            _df_plot = df_plot.filter(
                (pl.col('num_clients') == nc) & (pl.col('num_samples') == ns)
            )
            if len(_df_plot) == 0:
                continue

            scatter = _df_plot.sort('num_clients', 'num_samples', 'split_midpoint').hvplot.scatter(
                x='split_midpoint',
                y='value',
                by='variable',
                s=10000,  # Reduced marker size for clarity
                alpha=0.9,
                marker=['v', '^'],
            )

            # Compute and overlay best-fit lines
            lines = []
            for var in sorted(_df_plot['variable'].unique()):
                sub_df = _df_plot.filter(pl.col('variable') == var)
                x = sub_df['split_midpoint'].to_numpy()
                y = sub_df['value'].to_numpy()

                if len(x) > 1:  # Ensure we have enough points to fit a line
                    coeffs = np.polyfit(x, y, deg=1)  # Linear fit
                    fit_x = np.linspace(x.min(), x.max(), 100)
                    fit_y = np.polyval(coeffs, fit_x)
                    lines.append(hv.Curve((fit_x, fit_y), label=f"{var} Fit").opts(alpha=0.6, show_legend=False))

            plot = scatter * hv.Overlay(lines)  # Combine scatter and best-fit lines
            #plot = plot

            hv.save(plot, f'images/metric_per_split_percentage/per_bucket_c{nc}_s{ns}.svg', fmt="svg")
            #_render = hv.render(plot, backend='bokeh')
            #_render.output_backend = "svg"
            #export_svgs(_render, filename = f'images/metric_per_split_percentage/c{nc}_s{ns}.svg')
            #_render.savefig(f'images/metric_per_split_percentage/c{nc}_s{ns}.svg', format='svg', bbox_inches='tight', dpi=300)
            #hv.save(plot, f'images/metric_per_split_percentage/c{nc}_s{ns}.html')





#plot_imbalance_over_split_percentages(df)


def plot_imbalance_over_split_percentages_no_bucket(df, ncs=[2,4,8], nss=[500,1000,2500,5000]):
    # Convert categorical intervals to numeric midpoints
    df_plot = df.sort(f'fisher_{metric}')
    df_plot = df_plot.group_by('num_clients', 'num_samples', 'max_split_percentile').agg(
        cs.ends_with(metric).mean()
    )

    df_plot = df_plot.unpivot(
        on=[f'fedci_{metric}', f'fisher_{metric}'],
        index=['num_clients', 'num_samples', 'max_split_percentile']
    )

    # Convert category to numerical midpoints
    # df_plot = df_plot.with_columns(
    #     pl.col('max_split_percentile').map_elements(extract_midpoint).alias('split_midpoint')
    # )

    for nc in ncs:
        for ns in nss:
            _df_plot = df_plot.filter(
                (pl.col('num_clients') == nc) & (pl.col('num_samples') == ns)
            )
            if len(_df_plot) == 0:
                continue


            value_variable = metric.split('_')[-1]

            _df_plot = _df_plot.rename({
                'num_clients': '# Clients',
                'num_samples': '# Samples',
                'max_split_percentile': 'Percentage of Largest Split',
                'value': value_variable,
            })
            _df_plot = _df_plot.with_columns(variable=pl.col('variable').str.reverse().str.splitn('_',3).struct.rename_fields(['a','b','c']).struct.field('c').str.reverse())

            scatter = _df_plot.sort('# Clients', '# Samples', 'Percentage of Largest Split').hvplot.scatter(
                x='Percentage of Largest Split',
                y=value_variable,
                by='variable',
                s=10000,  # Reduced marker size for clarity
                alpha=0.9,
                marker=['v', '^'],
            )

            # Compute and overlay best-fit lines
            lines = []
            for var in sorted(_df_plot['variable'].unique()):
                sub_df = _df_plot.filter(pl.col('variable') == var)
                x = sub_df['Percentage of Largest Split'].to_numpy()
                y = sub_df[value_variable].to_numpy()

                if len(x) > 1:  # Ensure we have enough points to fit a line
                    coeffs = np.polyfit(x, y, deg=1)  # Linear fit
                    fit_x = np.linspace(x.min(), x.max(), 100)
                    fit_y = np.polyval(coeffs, fit_x)
                    lines.append(hv.Curve((fit_x, fit_y), label=f"{var} Fit").opts(alpha=0.6, show_legend=False))

            plot = scatter * hv.Overlay(lines)  # Combine scatter and best-fit lines
            #plot = plot

            hv.save(plot, f'images/metric_per_split_percentage/c{nc}_s{ns}.svg', fmt="svg")
            #_render = hv.render(plot, backend='bokeh')
            #_render.output_backend = "svg"
            #export_svgs(_render, filename = f'images/metric_per_split_percentage/c{nc}_s{ns}.svg')
            #_render.savefig(f'images/metric_per_split_percentage/c{nc}_s{ns}.svg', format='svg', bbox_inches='tight', dpi=300)
            #hv.save(plot, f'images/metric_per_split_percentage/c{nc}_s{ns}.html')

#plot_imbalance_over_split_percentages_no_bucket(df)





def plot_imbalance_over_split_percentages_no_bucket_boxplot(df):
    # Convert categorical intervals to numeric midpoints
    #df_plot = df.sort(f'fisher_{metric}')
    #df_plot = df_plot.group_by('num_clients', 'num_samples', 'max_split_percentile').agg(
    #    cs.ends_with(metric).mean()
    #)

    df_plot = df.unpivot(
        on=[f'fedci_{metric}', f'fisher_{metric}'],
        index=['num_clients', 'num_samples', 'max_split_percentile']
    )

    for nc in [2,4,8]:
        for ns in [500,1000,2500,5000]:
            _df_plot = df_plot.filter(
                (pl.col('num_clients') == nc) & (pl.col('num_samples') == ns)
            )
            if len(_df_plot) == 0:
                continue


            value_variable = metric.split('_')[-1]

            _df_plot = _df_plot.rename({
                'num_clients': '# Clients',
                'num_samples': '# Samples',
                'max_split_percentile': 'Percentage of Largest Split',
                'value': value_variable,
            })
            _df_plot = _df_plot.with_columns(variable=pl.col('variable').str.reverse().str.splitn('_',3).struct.rename_fields(['a','b','c']).struct.field('c').str.reverse())

            #_df_plot = _df_plot.drop_nulls()

            _df_plot = _df_plot.with_columns(pl.col('Percentage of Largest Split').cast(pl.Utf8))

            print(_df_plot.head())

            plot = _df_plot.sort(
                '# Clients', '# Samples', 'variable', 'Percentage of Largest Split'
            ).hvplot.box(
                #x='Percentage of Largest Split',
                y=value_variable,
                by=['Percentage of Largest Split', 'variable'],
                #by='variable',
                #s=10000,  # Reduced marker size for clarity
                #alpha=0.9,
                #marker=['v', '^'],
                height=400,
                width=800,
            )


            hv.save(plot, f'images/metric_per_split_percentage/c{nc}_s{ns}.svg', fmt="svg")
            #_render = hv.render(plot, backend='bokeh')
            #_render.output_backend = "svg"
            #export_svgs(_render, filename = f'images/metric_per_split_percentage/c{nc}_s{ns}.svg')
            #_render.savefig(f'images/metric_per_split_percentage/c{nc}_s{ns}.svg', format='svg', bbox_inches='tight', dpi=300)
            #hv.save(plot, f'images/metric_per_split_percentage/c{nc}_s{ns}.html')

plot_imbalance_over_split_percentages_no_bucket_boxplot(df)










def plot_scatter_fedci_v_fisher(df):
    df_plot = df.filter(pl.all_horizontal(cs.starts_with('has_prediction_')))
    # df_plot = df_plot.filter(
    #     (pl.col('num_clients') == 4) & (pl.col('num_samples') == 5000)
    # )

    # df_plot = df_plot.filter(
    #     pl.col('num_samples') == 5000
    # )

    df_plot = df_plot.group_by('num_clients', f'fisher_{metric}').agg(pl.col(f'fedci_{metric}').mean())
    # hvplot.bivariate
    plot = df_plot.hvplot.scatter(
        x=f'fisher_{metric}',
        y=f'fedci_{metric}',
        #by='num_clients',
        ylim=(-0.1,1.1),
        xlim=(-0.1,1.1),
        height=400,
        width=400,
    )
    fit_x = np.linspace(-0.1, 1.1, 100)
    plot = plot * hv.Curve((fit_x, fit_x), label=f"Eq").opts(alpha=0.6, show_legend=False)
    hv.save(plot, f'images/test5.html')

#plot_scatter_fedci_v_fisher(df)



# BOXPLOT DIFF IN SHD

df_plot = df.filter(pl.all_horizontal(cs.starts_with('has_prediction_')))
df_plot = df_plot.with_columns(diff=pl.col(f'fisher_{metric}') - pl.col(f'fedci_{metric}'))

df_plot = df_plot.filter(pl.col('diff') != 0)
print(len(df_plot))

df_plot = df_plot.with_columns(pl.col('num_clients', 'num_samples').cast(pl.Utf8))

plot = df_plot.sort(
    'num_samples', 'num_clients'
).hvplot.box(
    y='diff',
    by=['num_samples','num_clients'],
    height=400,
    width=400,
)
hv.save(plot, f'images/test6.html')





df_plot = df

#df_plot = df_plot.group_by('num_clients', 'num_samples').agg(cs.starts_with('has_prediction_').mean())
df_plot = df_plot.group_by('num_clients').agg(cs.starts_with('has_prediction_').mean())


df_plot = df_plot.unpivot(
    on=cs.starts_with('has_prediction'),
    index=['num_clients']#, 'num_samples']
)


#df_plot = df_plot.with_columns(pl.col('num_clients', 'num_samples').cast(pl.Utf8))
df_plot = df_plot.with_columns(pl.col('num_clients').cast(pl.Utf8))

df_plot = df_plot.sort('num_clients')

plot = df_plot.hvplot.bar(
    y='value',
    x='variable',
    by='num_clients'
)
hv.save(plot, f'images/test7.png')

# plot = df_plot.hvplot.bar(
#     y='value',
#     x='variable',
#     by='num_samples'
# )
# hv.save(plot, f'images/test8.png')

"""
Plot Ideas - each also per approach:
 - correct_pag_hits per sample size
 - number_of_predictions per sample size ?
 - SHD per sample size

 - How good is the SHD per algorithm
 - How good is the SHD per algorithm when others did not predict anything
 - How good is the SHD when all have a prediction
"""
