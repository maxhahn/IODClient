import polars as pl
import polars.selectors as cs

target_folder = "shd"

df = pl.read_ndjson("./mixed_pag_results.json")

df = df.with_columns(
    found_correct_fedci=pl.col("fedci_shd").list.contains(0),
    found_correct_fisher=pl.col("fisher_shd").list.contains(0),
    num_pags_fedci=pl.col("fedci_shd").list.len(),
    num_pags_fisher=pl.col("fisher_shd").list.len(),
    sum_shd_fedci=pl.col("fedci_shd").list.sum(),
    sum_shd_fisher=pl.col("fisher_shd").list.sum(),
    min_shd_fedci=pl.col("fedci_shd").list.min(),
    min_shd_fisher=pl.col("fisher_shd").list.min(),
    fully_faithful_fedci=pl.col("fedci_unfaithfulness").list.len() == 0,
    fully_faithful_fisher=pl.col("fisher_unfaithfulness").list.len() == 0,
)

# df2 = df.group_by("fully_faithful_fedci", "fully_faithful_fisher").agg(
#     avg_min_shd_fedci=pl.sum("sum_shd_fedci") / pl.sum("num_pags_fedci"),
#     avg_min_shd_fisher=pl.sum("sum_shd_fisher") / pl.sum("num_pags_fisher"),
#     len=pl.len(),
# )
# print(df2)

df_counts = df.group_by("fully_faithful_fedci", "fully_faithful_fisher").len()

df_fed = (
    df.select("fully_faithful_fedci", "fully_faithful_fisher", "fedci_shd")
    .explode("fedci_shd")
    .group_by("fully_faithful_fedci", "fully_faithful_fisher")
    .agg(mean=pl.mean("fedci_shd"), std=pl.std("fedci_shd"), len2=pl.len())
)
print(
    df_fed.join(
        df_counts, on=["fully_faithful_fedci", "fully_faithful_fisher"], how="inner"
    ).sort("fully_faithful_fedci", "fully_faithful_fisher")
)


df_fish = (
    df.select("fully_faithful_fedci", "fully_faithful_fisher", "fisher_shd")
    .explode("fisher_shd")
    .group_by("fully_faithful_fedci", "fully_faithful_fisher")
    .agg(mean=pl.mean("fisher_shd"), std=pl.std("fisher_shd"), len2=pl.len())
)
print(
    df_fish.join(
        df_counts, on=["fully_faithful_fedci", "fully_faithful_fisher"], how="inner"
    ).sort("fully_faithful_fedci", "fully_faithful_fisher")
)

df_fed = (
    df.select("found_correct_fedci", "found_correct_fisher", "fedci_shd")
    .with_columns(
        mean_fedci_shd=pl.col("fedci_shd").list.mean(),
        std_fedci_shd=pl.col("fedci_shd").list.std(),
    )
    .group_by("found_correct_fedci", "found_correct_fisher")
    .agg(mean=pl.mean("fedci_shd"), std=pl.std("fedci_shd"), len=pl.len())
)
print(df_fed)
asd

print(df)

print("Found correct")
print(df.select(cs.starts_with("found_correct_")).mean())

cnt = 0
for row in df.filter(
    pl.col("found_correct_fisher") & ~pl.col("found_correct_fedci")
).iter_rows(named=True):
    if cnt < 3:
        cnt += 1
        continue
    print(row["file_id"])
    print("FEDCI")
    print(row["found_correct_fedci"], row["num_pags_fedci"])
    for d in row["fedci_unfaithfulness"]:
        print(
            f"{d['X']} indep {d['Y']} given {','.join(d['S']):7} -> {str(d['predicted_independence']):5} ({d['pvalue']:.4f})"
        )
    print("FISHER")
    print(row["found_correct_fisher"], row["num_pags_fisher"])
    for d in row["fisher_unfaithfulness"]:
        print(d)
        print(
            f"{d['X']} indep {d['Y']} given {','.join(d['S']):7} -> {str(d['predicted_independence']):5} ({d['pvalue']:.4f})"
        )
    break

print("Num of correctness for fedci and fisher")
print(df.group_by(cs.starts_with("found_correct_")).len())
print("Num of correctness for fedci and fisher by faithfulness")
print(
    df.group_by("fully_faithful_fedci", cs.starts_with("found_correct_"))
    .len()
    .sort("fully_faithful_fedci", cs.starts_with("found_correct_"))
)
print("Num of correctness for fedci and fisher with at least one prediction for both")
print(
    df.filter((pl.col("num_pags_fedci") > 0) & (pl.col("num_pags_fisher") > 0))
    .group_by(cs.starts_with("found_correct_"))
    .len()
)

print("Avg Num Predictions")
print(df.select(cs.starts_with("num_pags_")).mean())
print("Avg Num Predictions with at least one prediction for both")
print(
    df.filter((pl.col("num_pags_fedci") > 0) & (pl.col("num_pags_fisher") > 0))
    .select(cs.starts_with("num_pags_"))
    .mean()
)

print("No Predictions")
print(df.select(cs.starts_with("num_pags_") == 0).mean())

print("Avg. SHD")
print(
    df.select(cs.starts_with("num_pags") | cs.starts_with("sum"))
    .sum()
    .select(
        fedci=pl.col("sum_shd_fedci") / pl.col("num_pags_fedci"),
        fisher=pl.col("sum_shd_fisher") / pl.col("num_pags_fisher"),
    )
)

print("Avg MIN SHD")
print(df.select(pl.col("min_shd_fedci", "min_shd_fisher").mean()))
print(
    df.group_by("fully_faithful_fedci").agg(
        pl.col("min_shd_fedci", "min_shd_fisher").mean()
    )
)
print(
    df.group_by(pl.col("file_id").str.contains("-g")).agg(
        pl.col("min_shd_fedci", "min_shd_fisher").mean()
    )
)

_df = df.with_columns(pl.col("min_shd_fedci", "min_shd_fisher").fill_null(20))
print(
    _df.group_by(cs.starts_with("fully_faithful_")).agg(
        pl.col("min_shd_fedci", "min_shd_fisher").mean(), pl.len()
    )
)
print(
    _df.group_by(cs.starts_with("fully_faithful_fedci")).agg(
        pl.col("min_shd_fedci", "min_shd_fisher").mean(), pl.len()
    )
)

print("Faithfulness")
print(df.group_by(cs.starts_with("fully_faithful_")).len())

print(
    df.group_by(
        pl.col("file_id").str.contains("-g"), cs.starts_with("fully_faithful_")
    ).len()
)
print(df.group_by(pl.col("file_id").str.contains("-g")).len())

import hvplot
import hvplot.polars
import holoviews as hv
import matplotlib.pyplot as plt

plt.rcParams.update({"svg.fonttype": "none"})

hvplot.extension("matplotlib")


df_exp = df.unpivot(
    on=["fedci_shd", "fisher_shd"],
    # index='test_id',
    value_name="SHD",
    variable_name="Method",
)

df_exp = df_exp.explode("SHD")

plot = df_exp.hvplot.hist(y="SHD", by="Method", alpha=0.5)

_render = hv.render(plot, backend="matplotlib")
_render.savefig(
    f"images/{target_folder}/hist-comparison.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


_df = df.filter(pl.col("num_pags_fisher") == 0)

df_exp = _df.unpivot(
    on=["fedci_shd", "fisher_shd"],
    # index='test_id',
    value_name="SHD",
    variable_name="Method",
)

df_exp = df_exp.explode("SHD")

plot = df_exp.hvplot.hist(
    y="SHD",
    # by='Method',
    alpha=0.5,
)

_render = hv.render(plot, backend="matplotlib")
_render.savefig(
    f"images/{target_folder}/hist-where-fisher-failed.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


_df = df.filter(pl.col("num_pags_fisher") > 0)

df_exp = _df.unpivot(
    on=["fedci_shd", "fisher_shd"],
    # index='test_id',
    value_name="SHD",
    variable_name="Method",
)

df_exp = df_exp.explode("SHD")

plot = df_exp.hvplot.hist(y="SHD", by="Method", alpha=0.5)

_render = hv.render(plot, backend="matplotlib")
_render.savefig(
    f"images/{target_folder}/hist-where-both-predicted.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


_df = df.with_columns(
    fedci_min_shd=pl.col("fedci_shd").list.min(),
    fisher_min_shd=pl.col("fisher_shd").list.min(),
)

_df = _df.with_columns(pl.col("fedci_min_shd", "fisher_min_shd").fill_null(pl.lit(20)))

_df = _df.unpivot(
    on=["fedci_min_shd", "fisher_min_shd"],
    # index='test_id',
    value_name="MIN_SHD",
    variable_name="Method",
)

plot = _df.hvplot.hist(y="MIN_SHD", by="Method", alpha=0.5)

_render = hv.render(plot, backend="matplotlib")
_render.savefig(
    f"images/{target_folder}/hist-min-shd.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


plot = _df.hvplot.box(y="MIN_SHD", by="Method")

_render = hv.render(plot, backend="matplotlib")
_render.savefig(
    f"images/{target_folder}/box-min-shd.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


_df = df.filter(pl.col("num_pags_fisher") > 0)

_df = _df.with_columns(
    fedci_min_shd=pl.col("fedci_shd").list.min(),
    fisher_min_shd=pl.col("fisher_shd").list.min(),
)

# _df = _df.with_columns(pl.col('fedci_min_shd', 'fisher_min_shd').fill_null(pl.lit(20)))

_df = _df.unpivot(
    on=["fedci_min_shd", "fisher_min_shd"],
    # index='test_id',
    value_name="MIN_SHD",
    variable_name="Method",
)

plot = _df.hvplot.hist(y="MIN_SHD", by="Method", alpha=0.5)

_render = hv.render(plot, backend="matplotlib")
_render.savefig(
    f"images/{target_folder}/hist-min-shd-where-both-predicted.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)
