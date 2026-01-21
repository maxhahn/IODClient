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


df_base = pl.read_parquet("simulations/*.parquet")

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


def eval_shd(df):
    def adjust_axis_spacing(plot, element):
        ax = plot.handles["axis"]
        # Distance between axis line and tick labels
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)
        # Distance between tick labels and axis labels
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10

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

    df = df.rename({"fedci_shds_og_right": "fedci_local_shds_og"})
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


# print(df_base.columns)
eval_shd(df_base)
eval_correct_iod(df_base)
eval_missing_iod(df_base)
