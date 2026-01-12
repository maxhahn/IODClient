import holoviews as hv
import hvplot
import hvplot.polars
from holoviews import opts

hvplot.extension("matplotlib")
hv.extension("matplotlib")

import glob

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs

plt.rcParams.update({"svg.fonttype": "none"})

pl.Config.set_tbl_rows(40)

# df.group_by('pag_id', 'seed', 'num_samples', 'partitions').len().filter(pl.col('len')!=42)

image_folder = "images/fixed_effect_images"

# folder = "experiments/fixed_effect_data/sim"
folder = "experiments/fixed_effect_data/sim"
# folder = "experiments/fixed_effect_data/SLIDES_MIXED"

alpha = 0.05

dfs = []
for file in glob.glob(f"{folder}/*.parquet"):
    df = pl.read_parquet(file)
    dfs.append(df)
#df_base = pl.read_parquet(f"{folder}/*.parquet")

df_base = pl.concat(dfs, how='vertical_relaxed')

####
# FILTER OUT ALL IMPOSSIBLE TESTS (are auto filtered in updated code, but this supports old files)
df_base = df_base.with_columns(
    non_global_vars=pl.col("vars1").list.set_symmetric_difference(pl.col("vars2"))
)
df_base = df_base.filter(
    (
        pl.col("S").str.replace_all(
            ",",
            "",
        )
        + pl.col("X")
        + pl.col("Y")
    )
    .str.split("")
    .list.set_intersection(pl.col("non_global_vars"))
    .list.len()
    < 2
)
####

print("== Perc of nulls in pooled")
print(df_base.select(cs.contains("pvalue").null_count() / len(df_base)))


df_sample_errors = (
    df_base.filter(
        pl.col("pooled_pvalue")
        .is_null()
        .any()
        .over("pag_id", "seed", "num_samples", "partitions")
    )
    .select("pag_id", "seed", "num_samples", "partitions")
    .unique()
)
print("== SAMPLE ERRORS ==")
print(df_sample_errors)

# df_base = df_base.filter(pl.col("pooled_pvalue").is_not_null())

df_base = df_base.join(
    df_sample_errors, on=["pag_id", "seed", "num_samples", "partitions"], how="anti"
)

df_base = df_base.with_columns(
    max_norm=pl.max_horizontal(
        [
            pl.col("norm_X_unres").list.max(),
            pl.col("norm_X_res").list.max(),
            pl.col("norm_Y_unres").list.max(),
            pl.col("norm_Y_res").list.max(),
        ]
    )
)

# df_base = df_base.filter(pl.col("max_norm") < 1000)


# df_base = df_base.filter(pl.col("fedci_pvalue").is_null())
# df_base = df_base.with_columns(
#     filename=pl.format(
#         "experiments/fixed_effect_data/sim2/{}-id{}-s{}-c{}.parquet",
#         pl.col("seed"),
#         pl.col("pag_id"),
#         pl.col("num_samples"),
#         pl.col("partitions"),
#     )
# )
# import os

# print(df_base.select("filename", "pag_id", "seed", "partitions", "num_samples"))
# for row in df_base.unique(subset=["filename"]).rows(named=True):
#     fname = row["filename"]
#     os.remove(fname)
# asd

# df_base = df_base.filter(pl.col("seed") != 10011)

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

    # print(df.filter(pl.col("fisher_pvalue").is_null()).select(cs.contains("pvalue")))


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
        _df = pl.concat(
            [
                df_fisher[0].select("Pooled"),
                df_fisher[0].select("Fisher"),
                df_fed[0].select("FedCI"),
            ],
            how="horizontal",
        )
    else:
        _df = (
            df.select(identifiers)
            .unique()
            .join(df_pooled, on=identifiers, how="left")
            .join(df_fisher, on=identifiers, how="left")
            .join(df_fed, on=identifiers, how="left")
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

    for i in df["partitions"].unique().to_list():
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
                color=hv.Cycle(["#1f77b4", "#d62728", "#2ca02c"]),  # Blue, Red, Green
                linestyle=hv.Cycle(["solid", "dashed", "dotted"]),
                alpha=1,
                backend="matplotlib",
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

    df_dep = df_dep.group_by(identifiers).agg(cs.contains("pvalue").mean(), pl.len())

    df_dep = df_dep.rename(
        {
            "pooled_pvalue": "Pooled",
            "fisher_pvalue": "Fisher",
            "fedci_pvalue": "FedCI",
        }
    )

    df_unpivot = df_dep.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="accuracy",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "accuracy": "Decision Agreements"}
    )

    for i in df["partitions"].unique().to_list():
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
                color=hv.Cycle(["#1f77b4", "#d62728", "#2ca02c"]),  # Blue, Red, Green
                linestyle=hv.Cycle(["solid", "dashed", "dotted"]),
                alpha=1,
                backend="matplotlib",
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

    df_indep = df_indep.group_by(identifiers).agg(
        cs.contains("pvalue").mean(), pl.len()
    )

    df_indep = df_indep.rename(
        {
            "pooled_pvalue": "Pooled",
            "fisher_pvalue": "Fisher",
            "fedci_pvalue": "FedCI",
        }
    )

    df_unpivot = df_indep.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="accuracy",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "accuracy": "Decision Agreements"}
    )

    for i in df["partitions"].unique().to_list():
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
                color=hv.Cycle(["#1f77b4", "#d62728", "#2ca02c"]),  # Blue, Red, Green
                linestyle=hv.Cycle(["solid", "dashed", "dotted"]),
                alpha=1,
                backend="matplotlib",
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

    # print('Weighted!')

    df_weighted = df_dep.join(df_indep, on=identifiers, suffix="_indep")
    assert len(df_weighted) == len(df_dep) and len(df_dep) == len(df_indep)

    df_weighted = df_weighted.with_columns(
        total_len=pl.col("len") + pl.col("len_indep")
    )
    df_weighted = df_weighted.with_columns(
        Pooled=(
            pl.col("Pooled") * pl.col("len")
            + pl.col("Pooled_indep") * pl.col("len_indep")
        )
        / pl.col("total_len"),
        Fisher=(
            pl.col("Fisher") * pl.col("len")
            + pl.col("Fisher_indep") * pl.col("len_indep")
        )
        / pl.col("total_len"),
        FedCI=(
            pl.col("FedCI") * pl.col("len")
            + pl.col("FedCI_indep") * pl.col("len_indep")
        )
        / pl.col("total_len"),
    )
    df_weighted = df_weighted.drop(cs.ends_with("_indep")).drop("total_len")

    df_unpivot = df_weighted.unpivot(
        on=["Pooled", "Fisher", "FedCI"],
        index=identifiers,
        value_name="accuracy",
        variable_name="Method",
    )

    df_unpivot = df_unpivot.rename(
        {"num_samples": "\# Samples", "accuracy": "Decision Agreements"}
    )

    for i in df["partitions"].unique().to_list():
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
                color=hv.Cycle(["#1f77b4", "#d62728", "#2ca02c"]),  # Blue, Red, Green
                linestyle=hv.Cycle(["solid", "dashed", "dotted"]),
                alpha=1,
                backend="matplotlib",
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
    for num_p in df["partitions"].unique().to_list():
        for num_s in df["num_samples"].unique().to_list():
            _df = df.filter(
                (pl.col("num_samples") == num_s) & (pl.col("partitions") == num_p)
            )
            _df = (
                _df.group_by("MSep", cs.contains("pvalue"))
                .len()
                .sort("MSep", cs.contains("pvalue"))
            )
            print(f"Agreement Table for {num_s} samples over {num_p} partitions")
            print(_df)  # .filter(pl.col('fisher_pvalue')!=pl.col('fedci_pvalue')))


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


def show_fisher_v_fedci_disagreement(df_base):
    print("== Show Number of Disagreements between Fisher and FedCI")
    df = df_base
    df = df_base.with_columns(cs.contains("pvalue") >= alpha)
    df = df.with_columns(cs.contains("pvalue") == pl.col("MSep"))
    df = df.filter(pl.col("fisher_pvalue") != pl.col("fedci_pvalue")).drop_nulls()

    print(
        df.group_by("partitions", "num_samples", "fisher_pvalue", "fedci_pvalue")
        .len()
        .sort("partitions", "num_samples", "fisher_pvalue", "fedci_pvalue")
    )


def show_fedci_pooled_diff(df_base, log=True):
    print("== Largest fedci diff to pooled")
    df = df_base

    if log:
        df = df.with_columns(
            pl.col("fedci_pvalue", "pooled_pvalue")
            .name.suffix("_log")
            .clip(pl.lit(1e-15), None)
            .log()
        )
    df = df.with_columns(
        diff=pl.col("fedci_pvalue") - pl.col("pooled_pvalue"),
        diff_log=pl.col("fedci_pvalue_log") - pl.col("pooled_pvalue_log"),
    )
    df = df.with_columns(
        X_type=pl.when(pl.col("X") == "A")
        .then(pl.col("var_types").struct.field("A"))
        .when(pl.col("X") == "B")
        .then(pl.col("var_types").struct.field("B"))
        .when(pl.col("X") == "C")
        .then(pl.col("var_types").struct.field("C"))
        .when(pl.col("X") == "D")
        .then(pl.col("var_types").struct.field("D"))
        .otherwise(pl.col("var_types").struct.field("E")),
        Y_type=pl.when(pl.col("Y") == "A")
        .then(pl.col("var_types").struct.field("A"))
        .when(pl.col("Y") == "B")
        .then(pl.col("var_types").struct.field("B"))
        .when(pl.col("Y") == "C")
        .then(pl.col("var_types").struct.field("C"))
        .when(pl.col("Y") == "D")
        .then(pl.col("var_types").struct.field("D"))
        .otherwise(pl.col("var_types").struct.field("E")),
    )
    pl.Config.set_tbl_cols(13)
    print("-- With Norms")
    print(
        df.select(
            "X",
            "Y",
            "S",
            "X_type",
            "Y_type",
            "norm_X_res",
            "norm_X_unres",
            "norm_Y_res",
            "norm_Y_unres",
            "pooled_pvalue",
            "fedci_pvalue",
            "diff_log",
        ).sort("diff_log")
    )
    print("-- Which test")
    print(
        df.select(
            "X",
            "Y",
            "S",
            "pag_id",
            "seed",
            "partitions",
            "num_samples",
            "pooled_pvalue",
            "fedci_pvalue",
            "pooled_pvalue_log",
            "fedci_pvalue_log",
            "diff",
            "diff_log",
        ).sort("diff_log")
    )
    print(
        pl.concat(
            [
                df.select(
                    "X",
                    "Y",
                    "S",
                    "pag_id",
                    "seed",
                    "partitions",
                    "num_samples",
                    "pooled_pvalue",
                    "fedci_pvalue",
                    "pooled_pvalue_log",
                    "fedci_pvalue_log",
                    "diff",
                    "diff_log",
                )
                .sort("diff_log")
                .head(2),
                df.select(
                    "X",
                    "Y",
                    "S",
                    "pag_id",
                    "seed",
                    "partitions",
                    "num_samples",
                    "pooled_pvalue",
                    "fedci_pvalue",
                    "pooled_pvalue_log",
                    "fedci_pvalue_log",
                    "diff",
                    "diff_log",
                )
                .sort("diff_log")
                .tail(2),
            ]
        )
    )
    # print("-- Bad fits test")
    # print(
    #     df.filter(pl.col("fedci_bad_fit"))
    #     .select(
    #         "X",
    #         "Y",
    #         "S",
    #         "pag_id",
    #         "seed",
    #         "partitions",
    #         "num_samples",
    #         "pooled_pvalue",
    #         "fedci_pvalue",
    #         "pooled_pvalue_log",
    #         "fedci_pvalue_log",
    #         "diff",
    #         "diff_log",
    #     )
    #     .sort("diff_log")
    # )


def show_deviation_from_pooled(df_base):
    print("== Deviation from pooled test")
    df = df_base

    df = df.drop_nulls()

    df = df.with_columns(cs.contains("pvalue").clip(pl.lit(1e-15), None).log())

    df = df.with_columns(
        fisher_pvalue_diff=pl.col("fisher_pvalue") - pl.col("pooled_pvalue"),
        fedci_pvalue_diff=pl.col("fedci_pvalue") - pl.col("pooled_pvalue"),
    )

    print(
        df.select(
            pl.col("fisher_pvalue_diff", "fedci_pvalue_diff")
            .mean()
            .name.suffix("_mean"),
            pl.col("fisher_pvalue_diff", "fedci_pvalue_diff").std().name.suffix("_std"),
        )
    )

    print(
        df.select(
            pl.col("fisher_pvalue_diff", "fedci_pvalue_diff").min().name.suffix("_min"),
            pl.col("fisher_pvalue_diff", "fedci_pvalue_diff").max().name.suffix("_max"),
        )
    )

    print(
        df.group_by("MSep").agg(
            pl.col("fisher_pvalue_diff", "fedci_pvalue_diff")
            .mean()
            .name.suffix("_mean"),
            pl.col("fisher_pvalue_diff", "fedci_pvalue_diff").std().name.suffix("_std"),
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

    df = df.rename({"fisher_pvalue_diff": "Fisher", "fedci_pvalue_diff": "FedCI"})

    df = df.unpivot(
        on=["FedCI", "Fisher"],
        index=["num_samples", "partitions"],
        value_name="p-value Difference",
        variable_name="Method",
    )

    # df = df.with_columns(
    #     pl.col("Method").replace_strict({"Federated": "F", "Meta-Analysis": "MA"})
    # )

    def hierarchical_labels(plot, element):
        ax = plot.handles["axis"]
        fig = plot.handles["fig"]

        # Get current tick labels and positions
        tick_positions = ax.get_xticks()
        labels = [label.get_text() for label in ax.get_xticklabels()]

        # Parse labels to extract method and partition
        parsed = []
        for label in labels:
            if "," in label:
                method, partition = label.split(",")
                parsed.append((method.strip(), partition.strip()))
            else:
                parsed.append((label, ""))

        # Set partition numbers as main labels
        ax.set_xticklabels([p[1] for p in parsed])

        # Create method group labels
        current_method = None
        method_spans = []
        start_idx = 0

        for idx, (method, _) in enumerate(parsed):
            if method != current_method:
                if current_method is not None:
                    method_spans.append((current_method, start_idx, idx - 1))
                current_method = method
                start_idx = idx

        # Add the last method group
        if current_method is not None:
            method_spans.append((current_method, start_idx, len(parsed) - 1))

        # Add method labels below using actual tick positions
        for method, start, end in method_spans:
            center = (tick_positions[start] + tick_positions[end]) / 2
            ax.text(
                center,
                -0.15,
                method,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

        # Move the xlabel further down
        ax.xaxis.set_label_coords(0.5, -0.25)

        # Adjust bottom margin to fit all labels
        fig.subplots_adjust(bottom=0.25)

    df = df.sort("Method", "num_samples", "partitions")

    # _df = _df.filter(pl.col("num_splits") > 2)

    for nsamples in df["num_samples"].unique().to_list():
        _df = df.filter(pl.col("num_samples") == nsamples)
        # print(_df)
        plot = _df.hvplot.box(
            # y='p-value Difference',# 'Meta-Analysis'],
            y="p-value Difference",
            by=["Method", "partitions"],
            # y='Meta-Analysis',# 'Meta-Analysis'],
            # by=['test_id', 'Method'],
            # ylabel='Normalized Difference in p-value',
            ylabel="Log-ratio of p-values",
            xlabel="Method, \# Partitions",
            # ylim=(-1,1),
            showfliers=True,
        )

        # plot = plot.opts(
        #     hooks=[adjust_axis_spacing],
        # )
        # Use this hook instead of adjust_axis_spacing
        plot = plot.opts(
            hooks=[hierarchical_labels],
        )

        _render = hv.render(plot, backend="matplotlib")
        _render.savefig(
            f"{image_folder}/logratio2pooled/by-partitions-s{nsamples}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )
    for nparts in df["partitions"].unique().to_list():
        _df = df.filter(pl.col("partitions") == nparts)

        plot = _df.hvplot.box(
            # y='p-value Difference',# 'Meta-Analysis'],
            y="p-value Difference",
            by=["Method", "num_samples"],
            # y='Meta-Analysis',# 'Meta-Analysis'],
            # by=['test_id', 'Method'],
            # ylabel='Normalized Difference in p-value',
            ylabel="Log-ratio of p-values",
            xlabel="Method, # Samples",
            # ylim=(-1,1),
            showfliers=True,
        )

        plot = plot.opts(
            hooks=[hierarchical_labels],
        )

        _render = hv.render(plot, backend="matplotlib")
        _render.savefig(
            f"{image_folder}/logratio2pooled/by-samples-p{nparts}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )


def pval_diffs(df_base):
    df = df_base

    df = df.with_columns(pval_diff_to_local=pl.col('pooled_pvalue')-pl.col('fedci_pvalue'))
    print(df.sort('pval_diff_to_local').select('seed', 'pag_id', 'num_samples', 'partitions', 'X','Y','S', cs.contains('pvalue'), 'p1', 'p2'))

    df_ = df.filter((pl.col('p1')-pl.col('p2')).abs() > 0.3)
    print(len(df_))
    print(df_.select("MSep", cs.contains("pvalue")).corr())

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
# show_fisher_v_fedci_disagreement(df_base)
show_deviation_from_pooled(df_base)
show_fedci_pooled_diff(df_base)
pval_diffs(df_base)
