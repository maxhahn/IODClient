import holoviews as hv
from holoviews import opts
import hvplot
import hvplot.polars

hvplot.extension('matplotlib')
hv.extension('matplotlib')

import polars as pl
import polars.selectors as cs



import matplotlib.pyplot as plt


plt.rcParams.update({"svg.fonttype": "none"})

pl.Config.set_tbl_rows(40)


image_folder = "images/fixed_effect_images"

folder = "experiments/fixed_effect_data/SLIDES"
#folder = "experiments/fixed_effect_data/SLIDES_MIXED"


alpha = 0.05
df_base = pl.read_parquet(f"{folder}/*.parquet")

graph = None  # "SLIDES"
num_samples = None
partitions = None
x_type = None
y_type = None

if graph is not None:
    df_base = df_base.filter(pl.col("graph") == graph)
if num_samples is not None:
    df_base = df_base.filter(pl.col("num_samples") == num_samples)
if partitions is not None:
    df_base = df_base.filter(pl.col("partitions") == partitions)
if x_type is not None:
    df_base = df_base.filter(pl.col("x_type") == x_type)
if y_type is not None:
    df_base = df_base.filter(pl.col("y_type") == y_type)

if len(df_base) == 0:
    raise Exception("Removed all samples")

# print(
#     df_base.group_by(
#         "partitions", "num_samples", pl.col("fisher_pvalue").is_null()
#     ).len()
# )


def show_null_counts_in_pvalues(df_base):
    print("== Nulls in predictions")
    df = df_base
    print(df.select(cs.contains("pvalue").null_count() / pl.len()))


def show_correlation_to_msep(df_base):
    print("== Correlation to MSep")
    df = df_base
    df = df.filter(~pl.col("fisher_pvalue").is_nan())
    print(df.select("MSep", cs.contains("pvalue")).corr())

    def get_correlation(df, identifiers, colx, coly):
        _df = df

        if len(identifiers) == 0:
            _df = _df.with_columns(p_value_correlation=pl.corr(colx, coly))
        else:
            _df = _df.group_by(identifiers).agg(p_value_correlation=pl.corr(colx, coly))

        _df = _df.with_columns(
            pl.col("p_value_correlation").fill_nan(None).fill_null(pl.lit(1))
        )

        return _df

    identifiers = ["partitions", "num_samples"]

    df_pooled = get_correlation(df, identifiers, "pooled_pvalue", "MSep").rename(
        {"p_value_correlation": "Pooled"}
    )
    df_fisher = get_correlation(df, identifiers, "fisher_pvalue", "MSep").rename(
        {"p_value_correlation": "Fisher"}
    )
    df_fed = get_correlation(df, identifiers, "fedci_pvalue", "MSep").rename(
        {"p_value_correlation": "FedCI"}
    )
    if len(identifiers) == 0:
        _df = pl.concat([df_fisher[0].select('Pooled'), df_fisher[0].select('Fisher'), df_fed[0].select('FedCI')], how='horizontal')
    else:
        _df = df.select(identifiers).unique().join(
            df_pooled, on=identifiers, how='left'
        ).join(
            df_fisher, on=identifiers, how='left'
        ).join(
            df_fed, on=identifiers, how='left'
        )

    df_unpivot = _df.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="correlation",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "correlation": "Correlation"}
    )

    for i in df['partitions'].unique().to_list():
        _plot = (
            df_unpivot.filter(pl.col("partitions") == i)
            .sort("Method", "\# Samples")
            .hvplot.line(
                x="\# Samples",
                y="Correlation",
                alpha=1,
                ylim=(-0.01, 1.01),
                width=400,
                height=400,
                by="Method",
                legend="top_left",
                xlabel=r"\# Samples",  # LaTeX-escaped #
                ylabel=r"Correlation",
                linestyle=["solid", "dashed", "dotted"],
                # title=f'{"Client" if i == 1 else "Clients"}'
            )
        )

        def adjust_axis_spacing(plot, element):
            ax = plot.handles["axis"]
            # Distance between axis line and tick labels
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)
            # Distance between tick labels and axis labels
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10

        _plot = _plot.opts(
            legend_opts={
                # "title": "Correctness",
                # "ncol": 2,  # two columns
                # "loc": "upper center",  # position above plot
                "bbox_to_anchor": (0.01, 1.0),  # fine-tune vertical placement
                "borderpad": 1.2,
                "labelspacing": 1.2,
                "frameon": False,  # usually looks cleaner for LaTeX
            },
            # --- Axis and label padding ---
            hooks=[adjust_axis_spacing],
        )
        _plot = _plot.opts(
            opts.Curve(
                color=hv.Cycle(['#1f77b4', '#d62728', '#2ca02c']),  # Blue, Red, Green
                linestyle=hv.Cycle(['solid', 'dashed', 'dotted']),
                alpha=1,
                backend='matplotlib'
            ),
        )

        # # Apply different line styles
        # # _plot = _plot.opts(line_dash=['solid', 'dashed'])

        _render = hv.render(_plot, backend="matplotlib")
        # _render.savefig(f'images/correlation-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
        _render.savefig(
            f"{image_folder}/slides_corr/correlation-c{i}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )

def show_bad_fisher_predictions(df_base):
    df = df_base
    print(
        df.filter(
            pl.col("fisher_pvalue").is_null() | pl.col("fisher_pvalue").is_nan()
        ).select("num_samples", "partitions", cs.contains("pvalue"))
    )


def show_msep_agreement(df_base):
    print("== MSep Agreement")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)
    df = df.with_columns(cs.contains("pvalue") == pl.col("MSep"))
    print("Dependent!")
    df_dep = df.filter(~pl.col("MSep"))
    print(
        df_dep.group_by("pooled_pvalue", "fisher_pvalue", "fedci_pvalue")
        .len()
        .sort("pooled_pvalue", "fisher_pvalue", "fedci_pvalue")
    )

    identifiers = ["partitions", "num_samples"]

    df_dep = df_dep.group_by(identifiers).agg(cs.contains('pvalue').mean(), pl.len())

    df_dep = df_dep.rename({
        'pooled_pvalue': 'Pooled',
        'fisher_pvalue': 'Fisher',
        'fedci_pvalue': 'FedCI',
    })

    df_unpivot = df_dep.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="accuracy",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "accuracy": "Decision Agreements"}
    )

    for i in df['partitions'].unique().to_list():
        _plot = (
            df_unpivot.filter(pl.col("partitions") == i)
            .sort("Method", "\# Samples")
            .hvplot.line(
                x="\# Samples",
                y="Decision Agreements",
                alpha=1,
                ylim=(-0.01, 1.01),
                width=400,
                height=400,
                by="Method",
                legend="top_left",
                xlabel=r"\# Samples",  # LaTeX-escaped #
                ylabel=r"Decision Agreements",
                linestyle=["solid", "dashed", "dotted"],
                # title=f'{"Client" if i == 1 else "Clients"}'
            )
        )

        def adjust_axis_spacing(plot, element):
            ax = plot.handles["axis"]
            # Distance between axis line and tick labels
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)
            # Distance between tick labels and axis labels
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10

        _plot = _plot.opts(
            legend_opts={
                # "title": "Correctness",
                # "ncol": 2,  # two columns
                # "loc": "upper center",  # position above plot
                "bbox_to_anchor": (0.01, 0.45),  # fine-tune vertical placement
                "borderpad": 1.2,
                "labelspacing": 1.2,
                "frameon": False,  # usually looks cleaner for LaTeX
            },
            # --- Axis and label padding ---
            hooks=[adjust_axis_spacing],
        )
        _plot = _plot.opts(
            opts.Curve(
                color=hv.Cycle(['#1f77b4', '#d62728', '#2ca02c']),  # Blue, Red, Green
                linestyle=hv.Cycle(['solid', 'dashed', 'dotted']),
                alpha=1,
                backend='matplotlib'
            ),
        )

        # # Apply different line styles
        # # _plot = _plot.opts(line_dash=['solid', 'dashed'])

        _render = hv.render(_plot, backend="matplotlib")
        # _render.savefig(f'images/correlation-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
        _render.savefig(
            f"{image_folder}/slides_acc/accuracy-dep-c{i}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )
    print("Independent!")
    df_indep = df.filter(pl.col("MSep"))
    print(
        df_indep.group_by("pooled_pvalue", "fisher_pvalue", "fedci_pvalue")
        .len()
        .sort("pooled_pvalue", "fisher_pvalue", "fedci_pvalue")
    )

    identifiers = ["partitions", "num_samples"]

    df_indep = df_indep.group_by(identifiers).agg(cs.contains('pvalue').mean(), pl.len())

    df_indep = df_indep.rename({
        'pooled_pvalue': 'Pooled',
        'fisher_pvalue': 'Fisher',
        'fedci_pvalue': 'FedCI',
    })

    df_unpivot = df_indep.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="accuracy",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "accuracy": "Decision Agreements"}
    )

    for i in df['partitions'].unique().to_list():
        _plot = (
            df_unpivot.filter(pl.col("partitions") == i)
            .sort("Method", "\# Samples")
            .hvplot.line(
                x="\# Samples",
                y="Decision Agreements",
                alpha=1,
                ylim=(-0.01, 1.01),
                width=400,
                height=400,
                by="Method",
                legend="top_left",
                xlabel=r"\# Samples",  # LaTeX-escaped #
                ylabel=r"Decision Agreements",
                linestyle=["solid", "dashed", "dotted"],
                # title=f'{"Client" if i == 1 else "Clients"}'
            )
        )

        def adjust_axis_spacing(plot, element):
            ax = plot.handles["axis"]
            # Distance between axis line and tick labels
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)
            # Distance between tick labels and axis labels
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10

        _plot = _plot.opts(
            legend_opts={
                # "title": "Correctness",
                # "ncol": 2,  # two columns
                # "loc": "upper center",  # position above plot
                "bbox_to_anchor": (0.01, 0.45),  # fine-tune vertical placement
                "borderpad": 1.2,
                "labelspacing": 1.2,
                "frameon": False,  # usually looks cleaner for LaTeX
            },
            # --- Axis and label padding ---
            hooks=[adjust_axis_spacing],
        )
        _plot = _plot.opts(
            opts.Curve(
                color=hv.Cycle(['#1f77b4', '#d62728', '#2ca02c']),  # Blue, Red, Green
                linestyle=hv.Cycle(['solid', 'dashed', 'dotted']),
                alpha=1,
                backend='matplotlib'
            ),
        )

        # # Apply different line styles
        # # _plot = _plot.opts(line_dash=['solid', 'dashed'])

        _render = hv.render(_plot, backend="matplotlib")
        # _render.savefig(f'images/correlation-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
        _render.savefig(
            f"{image_folder}/slides_acc/accuracy-indep-c{i}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )

    #print('Weighted!')

    df_weighted = df_dep.join(df_indep, on=identifiers, suffix='_indep')
    assert len(df_weighted) == len(df_dep) and len(df_dep) == len(df_indep)

    df_weighted=df_weighted.with_columns(
        total_len=pl.col('len')+pl.col('len_indep')
    )
    df_weighted=df_weighted.with_columns(
        Pooled=(pl.col('Pooled')*pl.col('len')+pl.col('Pooled_indep')*pl.col('len_indep'))/pl.col('total_len'),
        Fisher=(pl.col('Fisher')*pl.col('len')+pl.col('Fisher_indep')*pl.col('len_indep'))/pl.col('total_len'),
        FedCI=(pl.col('FedCI')*pl.col('len')+pl.col('FedCI_indep')*pl.col('len_indep'))/pl.col('total_len'),
    )
    df_weighted = df_weighted.drop(cs.ends_with('_indep')).drop('total_len')

    df_unpivot = df_weighted.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="accuracy",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "accuracy": "Decision Agreements"}
    )

    for i in df['partitions'].unique().to_list():
        _plot = (
            df_unpivot.filter(pl.col("partitions") == i)
            .sort("Method", "\# Samples")
            .hvplot.line(
                x="\# Samples",
                y="Decision Agreements",
                alpha=1,
                ylim=(-0.01, 1.01),
                width=400,
                height=400,
                by="Method",
                legend="top_left",
                xlabel=r"\# Samples",  # LaTeX-escaped #
                ylabel=r"Decision Agreements",
                linestyle=["solid", "dashed", "dotted"],
                # title=f'{"Client" if i == 1 else "Clients"}'
            )
        )

        def adjust_axis_spacing(plot, element):
            ax = plot.handles["axis"]
            # Distance between axis line and tick labels
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)
            # Distance between tick labels and axis labels
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10

        _plot = _plot.opts(
            legend_opts={
                # "title": "Correctness",
                # "ncol": 2,  # two columns
                # "loc": "upper center",  # position above plot
                "bbox_to_anchor": (0.01, 0.45),  # fine-tune vertical placement
                "borderpad": 1.2,
                "labelspacing": 1.2,
                "frameon": False,  # usually looks cleaner for LaTeX
            },
            # --- Axis and label padding ---
            hooks=[adjust_axis_spacing],
        )
        _plot = _plot.opts(
            opts.Curve(
                color=hv.Cycle(['#1f77b4', '#d62728', '#2ca02c']),  # Blue, Red, Green
                linestyle=hv.Cycle(['solid', 'dashed', 'dotted']),
                alpha=1,
                backend='matplotlib'
            ),
        )

        # # Apply different line styles
        # # _plot = _plot.opts(line_dash=['solid', 'dashed'])

        _render = hv.render(_plot, backend="matplotlib")
        # _render.savefig(f'images/correlation-c{i}.pgf', format='pgf', bbox_inches='tight', dpi=300)
        _render.savefig(
            f"{image_folder}/slides_acc/accuracy-weighted-c{i}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )

    df = df_base.with_columns(cs.contains("pvalue") >= alpha)
    df = df.with_columns(cs.contains("pvalue") == pl.col("MSep"))
    for num_p in df['partitions'].unique().to_list():
        for num_s in df['num_samples'].unique().to_list():
            _df = df.filter((pl.col('num_samples')==num_s)&(pl.col('partitions')==num_p))
            _df = _df.group_by('MSep', cs.contains('pvalue')).len().sort('MSep', cs.contains('pvalue'))
            print(f"Agreement Table for {num_s} samples over {num_p} partitions")
            print(_df)#.filter(pl.col('fisher_pvalue')!=pl.col('fedci_pvalue')))


def show_difference_to_msep(df_base):
    print("== MSep Difference")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)

    print("Pooled")
    print(df.group_by("MSep", "pooled_pvalue").len().sort("MSep", "pooled_pvalue"))
    print("Fisher")
    print(df.group_by("MSep", "fisher_pvalue").len().sort("MSep", "fisher_pvalue"))
    print("FedCI")
    print(df.group_by("MSep", "fedci_pvalue").len().sort("MSep", "fedci_pvalue"))


def show_correct_or_incorrect(df_base):
    print("== CORRECT OR INCORRECT")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)
    df = df.with_columns(cs.contains("pvalue") == pl.col("MSep"))

    print("Pooled")
    print(df.group_by("pooled_pvalue").len().sort("pooled_pvalue"))
    print("Fisher")
    print(df.group_by("fisher_pvalue").len().sort("fisher_pvalue"))
    print("FedCI")
    print(df.group_by("fedci_pvalue").len().sort("fedci_pvalue"))


def show_msep_versus_prediction_by_partition(df_base):
    print("== MSep Difference By Num Partitions")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)

    # df = df.filter(pl.col("MSep"))

    print("Pooled")
    print(
        df.group_by("partitions", "MSep", "pooled_pvalue")
        .len()
        .sort("partitions", "MSep", "pooled_pvalue")
    )
    print("Fisher")
    print(
        df.group_by("partitions", "MSep", "fisher_pvalue")
        .len()
        .sort("partitions", "MSep", "fisher_pvalue")
    )
    print("FedCI")
    print(
        df.group_by("partitions", "MSep", "fedci_pvalue")
        .len()
        .sort("partitions", "MSep", "fedci_pvalue")
    )


def show_incorrect_in_perc_by_partition(df_base):
    print("== MSep Difference By Num Partitions With Percent only errors")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)

    df = df.with_columns(num_tests_per_partition=pl.len().over("partitions"))

    print("Pooled")
    print(
        df.filter(pl.col("MSep") != pl.col("pooled_pvalue"))
        .group_by("partitions", "MSep", "pooled_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("MSep", "pooled_pvalue", "partitions")
    )
    print("Fisher")
    print(
        df.filter(pl.col("MSep") != pl.col("fisher_pvalue"))
        .group_by("partitions", "MSep", "fisher_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("MSep", "fisher_pvalue", "partitions")
    )
    print("FedCI")
    print(
        df.filter(pl.col("MSep") != pl.col("fedci_pvalue"))
        .group_by("partitions", "MSep", "fedci_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("MSep", "fedci_pvalue", "partitions")
    )


def show_incorrect_in_perc_based_on_indep_by_partition(df_base):
    print("== MSep Difference By Num Partitions With Percent of Dep/Indep only errors")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)

    df = df.with_columns(num_tests_per_partition=pl.len().over("MSep", "partitions"))

    print("Pooled")
    print(
        df.filter(pl.col("MSep") != pl.col("pooled_pvalue"))
        .group_by("partitions", "MSep", "pooled_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("MSep", "pooled_pvalue", "partitions")
    )
    print("Fisher")
    print(
        df.filter(pl.col("MSep") != pl.col("fisher_pvalue"))
        .group_by("partitions", "MSep", "fisher_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("MSep", "fisher_pvalue", "partitions")
    )
    print("FedCI")
    print(
        df.filter(pl.col("MSep") != pl.col("fedci_pvalue"))
        .group_by("partitions", "MSep", "fedci_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("MSep", "fedci_pvalue", "partitions")
    )


def show_incorrect_in_perc_based_on_indep_by_partition_with_ord(df_base):
    print("== MSep Difference By Num Partitions With Percent of Dep/Indep only errors")
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)

    df = df.with_columns(num_tests_per_partition=pl.len().over("MSep", "partitions"))

    print("Pooled")
    print(
        df.filter(pl.col("MSep") != pl.col("pooled_pvalue"))
        .group_by("ord", "partitions", "MSep", "pooled_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("ord", "MSep", "pooled_pvalue", "partitions")
    )
    print("Fisher")
    print(
        df.filter(pl.col("MSep") != pl.col("fisher_pvalue"))
        .group_by("ord", "partitions", "MSep", "fisher_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("ord", "MSep", "fisher_pvalue", "partitions")
    )
    print("FedCI")
    print(
        df.filter(pl.col("MSep") != pl.col("fedci_pvalue"))
        .group_by("ord", "partitions", "MSep", "fedci_pvalue")
        .agg(perc=pl.len() / pl.first("num_tests_per_partition"))
        .sort("ord", "MSep", "fedci_pvalue", "partitions")
    )


def show_correctness_on_bad_fisher_predictions(df_base):
    df = df_base
    df = df.filter(pl.col("fisher_pvalue").is_null() | pl.col("fisher_pvalue").is_nan())

    show_incorrect_in_perc_based_on_indep_by_partition(df)


show_null_counts_in_pvalues(df_base)
show_correlation_to_msep(df_base)
show_msep_agreement(df_base)
# show_difference_to_msep(df_base)
# show_correct_or_incorrect(df_base)
show_msep_versus_prediction_by_partition(df_base)
# show_incorrect_in_perc_by_partition(df_base)
show_incorrect_in_perc_based_on_indep_by_partition(df_base)
# show_incorrect_in_perc_based_on_indep_by_partition_with_ord(df_base)
# show_correctness_on_bad_fisher_predictions(df_base)


"""

Agreement Table for 1000 samples over 4 partitions
shape: (8, 5)
┌───────┬───────────────┬───────────────┬──────────────┬─────┐
│ MSep  ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len │
│ ---   ┆ ---           ┆ ---           ┆ ---          ┆ --- │
│ bool  ┆ bool          ┆ bool          ┆ bool         ┆ u32 │
╞═══════╪═══════════════╪═══════════════╪══════════════╪═════╡
│ false ┆ false         ┆ false         ┆ true         ┆ 94  │ <-
│ false ┆ false         ┆ true          ┆ false        ┆ 27  │
│ false ┆ true          ┆ false         ┆ true         ┆ 219 │ <-
│ false ┆ true          ┆ true          ┆ false        ┆ 42  │
│ true  ┆ false         ┆ false         ┆ true         ┆ 13  │
│ true  ┆ false         ┆ true          ┆ false        ┆ 17  │
│ true  ┆ true          ┆ false         ┆ true         ┆ 15  │
│ true  ┆ true          ┆ true          ┆ false        ┆ 6   │
└───────┴───────────────┴───────────────┴──────────────┴─────┘
Agreement Table for 1500 samples over 4 partitions
shape: (8, 5)
┌───────┬───────────────┬───────────────┬──────────────┬─────┐
│ MSep  ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len │
│ ---   ┆ ---           ┆ ---           ┆ ---          ┆ --- │
│ bool  ┆ bool          ┆ bool          ┆ bool         ┆ u32 │
╞═══════╪═══════════════╪═══════════════╪══════════════╪═════╡
│ false ┆ false         ┆ false         ┆ true         ┆ 71  │ <-
│ false ┆ false         ┆ true          ┆ false        ┆ 18  │
│ false ┆ true          ┆ false         ┆ true         ┆ 240 │ <-
│ false ┆ true          ┆ true          ┆ false        ┆ 28  │
│ true  ┆ false         ┆ false         ┆ true         ┆ 23  │
│ true  ┆ false         ┆ true          ┆ false        ┆ 16  │
│ true  ┆ true          ┆ false         ┆ true         ┆ 10  │
│ true  ┆ true          ┆ true          ┆ false        ┆ 9   │
└───────┴───────────────┴───────────────┴──────────────┴─────┘
Agreement Table for 2000 samples over 4 partitions
shape: (8, 5)
┌───────┬───────────────┬───────────────┬──────────────┬─────┐
│ MSep  ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len │
│ ---   ┆ ---           ┆ ---           ┆ ---          ┆ --- │
│ bool  ┆ bool          ┆ bool          ┆ bool         ┆ u32 │
╞═══════╪═══════════════╪═══════════════╪══════════════╪═════╡
│ false ┆ false         ┆ false         ┆ true         ┆ 66  │ <-
│ false ┆ false         ┆ true          ┆ false        ┆ 12  │
│ false ┆ true          ┆ false         ┆ true         ┆ 191 │ <-
│ false ┆ true          ┆ true          ┆ false        ┆ 50  │
│ true  ┆ false         ┆ false         ┆ true         ┆ 27  │
│ true  ┆ false         ┆ true          ┆ false        ┆ 16  │
│ true  ┆ true          ┆ false         ┆ true         ┆ 11  │
│ true  ┆ true          ┆ true          ┆ false        ┆ 11  │
└───────┴───────────────┴───────────────┴──────────────┴─────┘
Agreement Table for 2500 samples over 4 partitions
shape: (8, 5)
┌───────┬───────────────┬───────────────┬──────────────┬─────┐
│ MSep  ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len │
│ ---   ┆ ---           ┆ ---           ┆ ---          ┆ --- │
│ bool  ┆ bool          ┆ bool          ┆ bool         ┆ u32 │
╞═══════╪═══════════════╪═══════════════╪══════════════╪═════╡
│ false ┆ false         ┆ false         ┆ true         ┆ 50  │ <-
│ false ┆ false         ┆ true          ┆ false        ┆ 11  │
│ false ┆ true          ┆ false         ┆ true         ┆ 178 │ <-
│ false ┆ true          ┆ true          ┆ false        ┆ 39  │
│ true  ┆ false         ┆ false         ┆ true         ┆ 16  │
│ true  ┆ false         ┆ true          ┆ false        ┆ 13  │
│ true  ┆ true          ┆ false         ┆ true         ┆ 10  │
│ true  ┆ true          ┆ true          ┆ false        ┆ 7   │
└───────┴───────────────┴───────────────┴──────────────┴─────┘
Agreement Table for 3000 samples over 4 partitions
shape: (8, 5)
┌───────┬───────────────┬───────────────┬──────────────┬─────┐
│ MSep  ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len │
│ ---   ┆ ---           ┆ ---           ┆ ---          ┆ --- │
│ bool  ┆ bool          ┆ bool          ┆ bool         ┆ u32 │
╞═══════╪═══════════════╪═══════════════╪══════════════╪═════╡
│ false ┆ false         ┆ false         ┆ true         ┆ 46  │ <-
│ false ┆ false         ┆ true          ┆ false        ┆ 9   │
│ false ┆ true          ┆ false         ┆ true         ┆ 190 │ <-
│ false ┆ true          ┆ true          ┆ false        ┆ 36  │
│ true  ┆ false         ┆ false         ┆ true         ┆ 15  │ <-
│ true  ┆ false         ┆ true          ┆ false        ┆ 32  │
│ true  ┆ true          ┆ false         ┆ true         ┆ 7   │ <-
│ true  ┆ true          ┆ true          ┆ false        ┆ 10  │
└───────┴───────────────┴───────────────┴──────────────┴─────┘


=== FULL TABLE
Agreement Table for 3000 samples over 4 partitions
shape: (16, 5)
┌───────┬───────────────┬───────────────┬──────────────┬──────┐
│ MSep  ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len  │
│ ---   ┆ ---           ┆ ---           ┆ ---          ┆ ---  │
│ bool  ┆ bool          ┆ bool          ┆ bool         ┆ u32  │
╞═══════╪═══════════════╪═══════════════╪══════════════╪══════╡
│ false ┆ false         ┆ false         ┆ false        ┆ 193  │
│ false ┆ false         ┆ false         ┆ true         ┆ 46   │
│ false ┆ false         ┆ true          ┆ false        ┆ 9    │
│ false ┆ false         ┆ true          ┆ true         ┆ 310  │
│ false ┆ true          ┆ false         ┆ false        ┆ 670  │
│ false ┆ true          ┆ false         ┆ true         ┆ 190  │
│ false ┆ true          ┆ true          ┆ false        ┆ 36   │
│ false ┆ true          ┆ true          ┆ true         ┆ 5746 │
│ true  ┆ false         ┆ false         ┆ false        ┆ 10   │
│ true  ┆ false         ┆ false         ┆ true         ┆ 15   │
│ true  ┆ false         ┆ true          ┆ false        ┆ 32   │
│ true  ┆ false         ┆ true          ┆ true         ┆ 536  │
│ true  ┆ true          ┆ false         ┆ false        ┆ 7    │
│ true  ┆ true          ┆ false         ┆ true         ┆ 7    │
│ true  ┆ true          ┆ true          ┆ false        ┆ 10   │
│ true  ┆ true          ┆ true          ┆ true         ┆ 183  │
└───────┴───────────────┴───────────────┴──────────────┴──────┘

"""
