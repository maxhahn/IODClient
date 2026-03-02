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


def adjust_axis_spacing(plot, element):
    ax = plot.handles["axis"]
    # Distance between axis line and tick labels
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    # Distance between tick labels and axis labels
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10


plt.rcParams.update({"svg.fonttype": "none"})
plt.rcParams["axes.unicode_minus"] = False

pl.Config.set_tbl_rows(40)

NORM_SHD = True

# df.group_by('pag_id', 'seed', 'num_samples', 'partitions').len().filter(pl.col('len')!=42)

image_folder = "images/fixed_effect_images"

# folder = "experiments/fixed_effect_data/sim"
folder = "experiments/fixed_effect_data/sim"
# folder = "experiments/fixed_effect_data/SLIDES_MIXED"

alpha = 0.05

# dfs = []
# for file in glob.glob(f"{folder}/*.parquet"):
#     df = pl.read_parquet(file)
#     dfs.append(df)
# # df_base = pl.read_parquet(f"{folder}/*.parquet")

# df_base = pl.concat(dfs, how="vertical_relaxed")


df_base = pl.read_parquet("simulations/final-data.parquet")

print(df_base.select(pl.col("pag_id", "seed", "num_samples", "partitions").n_unique()))
print(df_base.group_by(pl.col("pag_id")).len().sort("len"))

print("30 runs only", len(df_base))
print(df_base.select(pl.col("pag_id", "seed", "num_samples", "partitions").n_unique()))
print(df_base.group_by("seed", "pag_id", "num_samples", "partitions").len().sort("len"))

print("== Perc of nulls in pooled")
print(df_base.select(cs.contains("pvalue").null_count() / len(df_base)))

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


df_base = df_base.rename({"fedci_shds_og_right": "fedci_local_shds_og"})

print(
    "Total number of sims",
    len(df_base.select("seed", "pag_id", "num_samples", "partitions").unique()),
)
print(df_base.group_by("seed", "pag_id", "num_samples", "partitions").len().sort("len"))


def eval_shd(df):
    def get_shd_stats(df, colname):
        if NORM_SHD:
            df = df.with_columns(pl.col(colname).list.eval(pl.element() / 20))
        df = df.with_columns(
            pl.col(colname).list.min().name.suffix("_min"),
            pl.col(colname).list.max().name.suffix("_max"),
            pl.col(colname).list.mean().name.suffix("_mean"),
            pl.col(colname).list.std().name.suffix("_std"),
            pl.col(colname).list.sum().name.suffix("_sum"),
            pl.col(colname).list.len().name.suffix("_len"),
        )
        return df

    df = get_shd_stats(df, "pooled_shds_og")
    df = get_shd_stats(df, "fisher_shds_og")
    df = get_shd_stats(df, "fedci_shds_og")
    df = get_shd_stats(df, "fedci_local_shds_og")
    # print(df.select(cs.contains("_mean").mean()))

    # sum is stupid -> benefits no prediction # maybe when 0 len is filtered out, but do you filter full row or only that entry?
    # len pretty useless
    # std pretty useless
    for metric in ["max", "min", "mean"]:
        _df = df.unpivot(
            on=cs.ends_with(f"_{metric}"),
            index=["pag_id", "seed", "num_samples", "partitions"],
            value_name="shd_stat",
            variable_name="method",
        )

        method_name_dict = {
            f"pooled_shds_og_{metric}": "Pooled",
            f"fisher_shds_og_{metric}": "Fisher",
            f"fedci_shds_og_{metric}": "fedCI",
            f"fedci_local_shds_og_{metric}": "fedCI-CA",
        }
        _df = _df.with_columns(pl.col("method").replace_strict(method_name_dict))

        plot = _df.hvplot.box(
            # y='p-value Difference',# 'Meta-Analysis'],
            y="shd_stat",
            by=["method"],
            # y='Meta-Analysis',# 'Meta-Analysis'],
            # by=['test_id', 'Method'],
            # ylabel='Normalized Difference in p-value',
            ylabel="SHD",
            xlabel="Method",
            # ylim=(-1,1),
            showfliers=True,
        )
        plot = plot.opts(
            hooks=[adjust_axis_spacing],
        )

        _render = hv.render(plot, backend="matplotlib")
        _render.savefig(
            f"{image_folder}/shd/box-{metric}.svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )

    _df = df.unpivot(
        on=["pooled_shds_og", "fisher_shds_og", "fedci_shds_og", "fedci_local_shds_og"],
        index=["pag_id", "seed", "num_samples", "partitions"],
        value_name="shd_stat",
        variable_name="method",
    )

    method_name_dict = {
        f"pooled_shds_og": "Pooled",
        f"fisher_shds_og": "Fisher",
        f"fedci_shds_og": "fedCI",
        f"fedci_local_shds_og": "fedCI-CA",
    }
    _df = _df.with_columns(pl.col("method").replace_strict(method_name_dict))

    _df = _df.explode(["shd_stat"])

    print(
        _df.group_by("method").agg(
            pl.mean("shd_stat").name.suffix("_mean"),
            pl.std("shd_stat").name.suffix("_std"),
        )
    )

    plot = _df.hvplot.box(
        # y='p-value Difference',# 'Meta-Analysis'],
        y="shd_stat",
        by=["method"],
        # y='Meta-Analysis',# 'Meta-Analysis'],
        # by=['test_id', 'Method'],
        # ylabel='Normalized Difference in p-value',
        ylabel="SHD",
        xlabel="Method",
        # ylim=(-1,1),
        showfliers=True,
    )
    plot = plot.opts(
        hooks=[adjust_axis_spacing],
    )

    _render = hv.render(plot, backend="matplotlib")
    _render.savefig(
        f"{image_folder}/shd/box-raw.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
    )


def eval_correct_iod(df):
    df = df.with_columns(
        cs.contains("shds_og").list.contains(pl.lit(0)).name.suffix("_correct")
    )

    print(df.select(cs.ends_with("_correct").mean()))


def eval_missing_iod(df):
    df = df.with_columns(
        (cs.contains("shds_og").list.len() == 0).name.suffix("_missing_pred")
    )

    print(df.select(cs.ends_with("_missing_pred").mean()))
    pl.Config.set_tbl_cols("10")
    print(
        df.group_by(cs.ends_with("_missing_pred"))
        .agg(pl.len())
        # .sort(cs.ends_with("_missing_pred"))
        .sort(
            pl.col("fisher_shds_og_missing_pred"),
            cs.ends_with("_missing_pred") - cs.by_name("fisher_shds_og_missing_pred"),
        )
    )

    print(
        df.group_by(cs.ends_with("_missing_pred"))
        .agg(cs.contains("shds_og").list.min().mean())
        .sort(
            pl.col("fisher_shds_og_missing_pred"),
            cs.ends_with("_missing_pred") - cs.by_name("fisher_shds_og_missing_pred"),
        )
    )


def eval_diff_iod(df):
    if NORM_SHD:
        df = df.with_columns(cs.contains("_shds_og").list.eval(pl.element() / 20))

    df = df.with_columns(cs.contains("_shds_og").list.min().name.suffix("_min"))
    # df = df.with_columns(cs.ends_with("_min").fill_null(1))
    df = df.with_columns(
        fedci_pooled_diff=pl.col("fedci_shds_og_min") - pl.col("pooled_shds_og_min"),
        fedci_ca_pooled_diff=pl.col("fedci_local_shds_og_min")
        - pl.col("pooled_shds_og_min"),
        fisher_pooled_diff=pl.col("fisher_shds_og_min") - pl.col("pooled_shds_og_min"),
        fisher_fedci_diff=pl.col("fisher_shds_og_min") - pl.col("fedci_shds_og_min"),
        fedci_ca_fedci_diff=pl.col("fedci_local_shds_og_min")
        - pl.col("fedci_shds_og_min"),
        fisher_fedci_ca_diff=pl.col("fisher_shds_og_min")
        - pl.col("fedci_local_shds_og_min"),
    )

    method_name_dict = {
        "fisher_fedci_diff": "Fisher-fedCI",
        "fedci_pooled_diff": "fedCI-Pooled",
        "fisher_pooled_diff": "Fisher-Pooled",
    }
    _df = df.drop(
        cs.contains("diff") - cs.by_name(list(method_name_dict.keys()))
    ).unpivot(
        on=cs.ends_with("_diff"),
        index=["pag_id", "seed", "num_samples", "partitions"],
        value_name="shd_stat",
        variable_name="method",
    )

    _df = _df.with_columns(pl.col("method").replace_strict(method_name_dict))

    plot = _df.hvplot.box(
        # y='p-value Difference',# 'Meta-Analysis'],
        y="shd_stat",
        by=["method"],
        # y='Meta-Analysis',# 'Meta-Analysis'],
        # by=['test_id', 'Method'],
        # ylabel='Normalized Difference in p-value',
        ylabel="SHD",
        xlabel="Method",
        # ylim=(-1,1),
        showfliers=True,
        # notch=True,
    )
    plot = plot.opts(
        hooks=[adjust_axis_spacing],
    )

    _render = hv.render(plot, backend="matplotlib")
    _render.savefig(
        f"{image_folder}/shd/box-diff.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
    )

    print(
        df.select(
            cs.ends_with("diff").mean(),
            cs.ends_with("diff").std().name.suffix("_std"),
        ).unpivot(
            on=cs.contains("_diff"),
            index=[],
            value_name="Value",
            variable_name="Method",
        )
    )


def eval_diff_significance(df):
    if NORM_SHD:
        df = df.with_columns(cs.contains("_shds_og").list.eval(pl.element() / 20))
    # remove rows without predictions
    print(len(df))
    df = df.filter(~pl.all_horizontal(cs.contains("_shds_og").list.len() == 0))
    print(len(df))

    df = df.with_columns(cs.contains("_shds_og").list.min().name.suffix("_min"))
    # df = (
    #     df.group_by("num_samples", "partitions")
    #     .agg(cs.contains("_shds_og_min").mean())
    #     .sort("num_samples", "partitions")
    # )

    pairings = [
        ## ("pooled", "fisher"),
        ## ("pooled", "fedci"),
        ("fisher", "pooled"),
        ("fedci", "pooled"),
        ## ("fedci", "pooled"),
        ## ("pooled", "fedci_local"),
        ## ("fedci", "fisher"),
        # ("fedci_local", "pooled"),
    ]

    df = df.with_columns(
        (pl.col(f"{a}_shds_og_min") - pl.col(f"{b}_shds_og_min")).alias(f"{a}_{b}_diff")
        for a, b in pairings
    )

    # result = df.group_by("num_samples", "partitions").agg(
    #     pl.map_groups(
    #         cs.ends_with("_diff"),
    #         t_test,
    #         return_dtype=pl.Float64,
    #         returns_scalar=True,
    #     ).name.suffix("_ttest_p"),
    #     pl.len(),
    # )

    print(
        df.group_by("num_samples", "partitions")
        .agg(cs.ends_with("_shds_og_min").mean())
        .sort("num_samples", "partitions")
    )
    print(
        df.group_by("num_samples", "partitions")
        .agg(
            [
                (
                    (pl.col(f"{a}_shds_og_min") - pl.col(f"{b}_shds_og_min")).mean()
                ).alias(f"{a}_{b}_mean")
                for a, b in pairings
            ]
        )
        .sort("num_samples", "partitions")
    )

    cohen = (
        df.group_by("num_samples", "partitions")
        .agg(
            [
                (
                    (pl.col(f"{a}_shds_og_min") - pl.col(f"{b}_shds_og_min")).mean()
                    / (pl.col(f"{a}_shds_og_min") - pl.col(f"{b}_shds_og_min")).std()
                ).alias(f"{a}_{b}_cohens_d")
                for a, b in pairings
            ]
        )
        .fill_nan(pl.lit(0.0))
    )

    cohen = (
        cohen.with_columns(
            [
                (
                    pl.col(f"{a}_{b}_cohens_d").abs() * 0
                    + (1 / 10000 + pl.col(f"{a}_{b}_cohens_d") ** 2 / 20000).sqrt()
                ).alias(f"{a}_{b}_se")
                for a, b in pairings
            ]
        )
        .with_columns(
            [
                (pl.col(f"{a}_{b}_cohens_d") - 1.96 * pl.col(f"{a}_{b}_se")).alias(
                    f"{a}_{b}_ci_lower"
                )
                for a, b in pairings
            ],
        )
        .with_columns(
            [
                (pl.col(f"{a}_{b}_cohens_d") + 1.96 * pl.col(f"{a}_{b}_se")).alias(
                    f"{a}_{b}_ci_upper"
                )
                for a, b in pairings
            ],
        )
    )

    print(
        cohen.sort("num_samples", "partitions").drop(
            cs.contains("_upper") | cs.contains("_lower")
        )
    )
    return

    # equalities = df.group_by("num_samples", "partitions").agg(
    #     [
    #         (pl.col(f"{a}_shds_og_min") == pl.col(f"{b}_shds_og_min"))
    #         .mean()
    #         .alias(f"{a}_{b}_equal")
    #         for a, b in pairings
    #     ],
    # )

    # # --- Effect sizes to accompany Wilcoxon tests ---

    # # 1) Median paired difference (robust, interpretable)
    # effect_sizes = df.group_by("num_samples", "partitions").agg(
    #     [
    #         (pl.col(f"{a}_shds_og_min") - pl.col(f"{b}_shds_og_min"))
    #         .median()
    #         .alias(f"{a}_{b}_median_diff")
    #         for a, b in pairings
    #     ]
    # )

    # # 2) Probability of superiority (common-language effect size)
    # def prob_superiority(series: pl.Series):
    #     s = pl.Series(series).explode().drop_nulls()
    #     if len(s) == 0:
    #         return None
    #     return (s < 0).mean()  # lower SHD = better

    # prob_effects = df.group_by("num_samples", "partitions").agg(
    #     [
    #         pl.map_groups(
    #             pl.col(f"{a}_shds_og_min") - pl.col(f"{b}_shds_og_min"),
    #             prob_superiority,
    #             return_dtype=pl.Float64,
    #             returns_scalar=True,
    #         ).alias(f"{a}_{b}_P_superiority")
    #         for a, b in pairings
    #     ]
    # )

    # # 3) Join with your existing Wilcoxon table
    # # final_table = (
    # #     result.join(effect_sizes, on=["num_samples", "partitions"])
    # #     .join(prob_effects, on=["num_samples", "partitions"])
    # #     .join(equalities, on=["num_samples", "partitions"])
    # # )
    # final_table = result
    # pl.Config.set_tbl_cols(20)
    # pl.Config.set_tbl_rows(120)
    # # print(final_table.sort("num_samples", "partitions"))

    # long = final_table.unpivot(
    #     index=["num_samples", "partitions"], variable_name="key", value_name="value"
    # )

    # print(long.sort("num_samples", "partitions", "key"))


# print(df_base.columns)
# eval_shd(df_base)
# eval_correct_iod(df_base)
# eval_missing_iod(df_base)
# eval_diff_iod(df_base)
print("before diff signif", len(df_base))
eval_diff_significance(df_base)
