import polars as pl
import polars.selectors as cs

import os

# dir, target_folder = 'experiments/simulation/results7', 'ci_table'
dir, target_folder = "experiments/mixed_pag_pvalues3", "mixed_pags"
# dir, target_folder = 'experiments/simulation/single_data', 'ci_table_single_dataset'
#
# dir, target_folder = 'experiments/simulation/mixed_pag2', 'ci_table_fully_random_data'


# all_faithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/f2')]
# all_unfaithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/uf2')]

files = os.listdir(dir)


# print(len(files_faithful), len(files_unfaithful))

# dfs = []
# for t,f in files_by_type.items():
#     df = pl.read_parquet(f).with_columns(faithfulness=pl.lit(t), filename=pl.lit(f))
#     dfs.append(df)

dfs = []
for f in files:
    # if '1745511692009-7-50000-0.25-g-' not in f:
    #    continue
    df = pl.read_parquet(dir + "/" + f).with_columns(filename=pl.lit(f))
    dfs.append(df)

# dfs = []
# if len(files_faithful) > 0:
#    df_faithful = pl.read_parquet(files_faithful).with_columns(faithful=True)
#    dfs.append(df_faithful)
# if len(files_unfaithful) > 0:
#     df_unfaithful = pl.read_parquet(files_unfaithful).with_columns(faithful=False)
#     dfs.append(df_unfaithful)

df = pl.concat(dfs)

df = df.drop_nulls(subset="MSep")

three_tail_pags = [
    2,
    16,
    18,
    19,
    20,
    23,
    29,
    31,
    37,
    42,
    44,
    53,
    57,
    58,
    62,
    64,
    66,
    69,
    70,
    72,
    73,
    74,
    75,
    79,
    81,
    82,
    83,
    84,
    93,
    98,
]
three_tail_pags = [t - 1 for t in three_tail_pags]

df = df.with_columns(
    num_samples=pl.col("filename").str.split("-").list.get(1).cast(pl.Int32),
    split_sizes=pl.col("filename")
    .str.split("-")
    .list.get(2)
    .str.split("_")
    .cast(pl.List(pl.Int32)),
    pag_id=pl.col("filename").str.split("-").list.get(0),  # .cast(pl.Int32)
)
df = df.with_columns(num_splits=pl.col("split_sizes").list.len())

print("Num unique pags used", df["pag_id"].n_unique())


df = df.filter(pl.col("num_samples") == 10000)
df = df.filter(pl.col("num_splits") == 4)


alpha = 0.05
df = df.with_columns(
    indep_fisher=pl.col("pvalue_fisher") > alpha,
    indep_fedci=pl.col("pvalue_fedci") > alpha,
    indep_pooled=pl.col("pvalue_pooled") > alpha,
)

# print(df.columns)
pl.Config.set_tbl_rows(50)
# print(df.select('X', 'Y', 'S', 'filename', cs.contains('pvalue')).sort('pvalue_diff_fedci_pooled'))
# print(df.select('X', 'Y', 'S', 'filename', cs.contains('pvalue')).sort('diff_pvalue')[0].to_dict())

# df = pl.read_parquet(dir)

df = df.with_columns(
    correct_fisher=pl.col("MSep") == pl.col("indep_fisher"),
    correct_fedci=pl.col("MSep") == pl.col("indep_fedci"),
    correct_pooled=pl.col("MSep") == pl.col("indep_pooled"),
    correct_as_pooled_fisher=pl.col("indep_pooled") == pl.col("indep_fisher"),
    correct_as_pooled_fedci=pl.col("indep_pooled") == pl.col("indep_fedci"),
)


df = df.with_columns(is_faithful=pl.all("correct_pooled").over("filename"))

# df = df.filter(pl.col('is_faithful'))


# print(df.filter(~pl.col('correct_fedci') & pl.col('correct_fisher')).select('filename', 'X','Y','S',cs.contains('pvalue')))

print(df.select(cs.starts_with("correct_")).mean())
print(
    df.group_by("num_splits", "MSep")
    .agg(cs.starts_with("correct_").mean(), pl.len())
    .sort("num_splits", "MSep")
)
# print(df.group_by('ord', 'X', 'Y', 'S').agg(pl.col('MSep').first(), cs.starts_with('correct_').sum(), pl.len()).sort('ord', 'X', 'Y', 'S'))
# print(df.group_by('ord', 'X', 'Y', 'S').agg(pl.col('MSep').first(), cs.starts_with('correct_').sum() / pl.len(), pl.len()).sort('ord', 'X', 'Y', 'S'))

print(
    df.group_by("ord", "X", "Y", "S", "MSep")
    .agg(pl.col("correct_fedci", "correct_fisher").mean(), pl.len())
    .sort("ord", "X", "Y", "S")
)
# TODO: Visualizations of pvalues:
# - scatter plot
# - corr plot?
# - difference between fisher/fedci to pooled as boxplot
# Do visualizations for MSep = True / MSep = False (or use indep_pooled)

print(
    df.filter(~pl.col("correct_fisher")).select(
        "filename", "X", "Y", "S", "MSep", cs.contains("pvalue") - cs.contains("pooled")
    )
)

import hvplot
import hvplot.polars
import holoviews as hv
import matplotlib.pyplot as plt

plt.rcParams.update({"svg.fonttype": "none"})

hvplot.extension("matplotlib")

diagonal = hv.Curve([(0, 0), (1, 1)]).opts(linestyle="dotted", color="black", alpha=0.5)

# _df = df
# _df1 = _df.filter(pl.col('MSep'))
# _df1 = _df1.sample(min(len(_df1), 500))

# _df2 = _df.filter(~pl.col('MSep'))
# _df2 = _df2.sample(min(len(_df2), 500))

# _df = pl.concat([_df1, _df2])

# print(df.filter(pl.col('correct_fisher').is_null()).select('MSep',cs.starts_with('pva')))

print(df.group_by(pl.col("correct_fisher", "correct_fedci")).len())

_df = df.sample(min(len(df), 3000))
# _df = df

_df = _df.rename(
    {
        "pvalue_fedci": "Federated",
        "pvalue_fisher": "Meta-Analysis",
        "pvalue_pooled": "Pooled",
    }
)

_df = _df.with_columns(
    confusion_value=pl.when(pl.col("correct_fisher") & pl.col("correct_fedci"))
    .then(pl.lit("Both Correct"))
    .when(~pl.col("correct_fisher") & pl.col("correct_fedci"))
    .then(pl.lit("MA Incorrect"))
    .when(pl.col("correct_fisher") & ~pl.col("correct_fedci"))
    .then(pl.lit("F Incorrect"))
    .otherwise(pl.lit("Both Incorrect"))
)

color_mapping = {
    "Both Correct": "#2ca02c",  #'green',
    "MA Incorrect": "#ff7f0e",  #'orange',
    "F Incorrect": "#1f77b4",  #'blue',
    "Both Incorrect": "#d62728",  #'red'
}

_df = _df.with_columns(color=pl.col("confusion_value").replace_strict(color_mapping))

_df = _df.rename({"confusion_value": "Correctness"})

magni1 = hv.Curve([(-0.01, 0.11), (0.11, 0.11)]).opts(
    linestyle="solid", color="black", alpha=0.8
)
magni2 = hv.Curve([(0.11, -0.01), (0.11, 0.11)]).opts(
    linestyle="solid", color="black", alpha=0.8
)
magni3 = hv.Curve([(-0.01, -0.01), (-0.01, 0.11)]).opts(
    linestyle="solid", color="black", alpha=0.8
)
magni4 = hv.Curve([(-0.01, -0.01), (0.11, -0.01)]).opts(
    linestyle="solid", color="black", alpha=0.8
)


plot = _df.sort("Correctness").hvplot.scatter(
    x="Federated",
    y="Meta-Analysis",
    by="Correctness",
    color="color",
    alpha=0.7,
    ylim=(-0.01, 1.01),
    xlim=(-0.01, 1.01),
    width=400,
    height=400,
    # by='Method',
    # legend='top_left',
    # backend='matplotlib',
    # s=4000,
    xlabel=r"Federated p-value",  # LaTeX-escaped #
    ylabel=r"Meta-Analysis p-value",
    marker=["+", "x", "^", "v"],
    # linestyle=['dashed', 'dotted']
    # title=f'{"Client" if i == 1 else "Clients"}'
)

# plot = plot.opts(legend_opts={'title': 'Correctness'})

# plot = plot.opts(
#     legend_opts={
#         #"title": "Correctness",
#         "borderpad": 1.5,
#         "labelspacing": 1.05,
#         "frameon": True,
#         "loc": "upper left"
#     }
# )

plot = plot.opts(
    legend_opts={
        # "title": "Correctness",
        "ncol": 2,  # two columns
        "loc": "upper center",  # position above plot
        "bbox_to_anchor": (0.5, 1.25),  # fine-tune vertical placement
        "borderpad": 1.2,
        "labelspacing": 0.9,
        "frameon": False,  # usually looks cleaner for LaTeX
    }
)

_render = hv.render(plot * magni1 * magni2 * magni3 * magni4, backend="matplotlib")
ax = _render.axes[0]
legend = ax.get_legend()
if legend:
    legend.set_title("Correctness")
_render.savefig(
    f"images/{target_folder}/scatter-fedci-v-fisher-colored.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)

__df = _df.filter((pl.col("Federated") <= 0.1) & (pl.col("Meta-Analysis") <= 0.1))

plot = __df.sort("Correctness").hvplot.scatter(
    x="Federated",
    y="Meta-Analysis",
    by="Correctness",
    color="color",
    alpha=0.7,
    ylim=(-0.001, 0.101),
    xlim=(-0.001, 0.101),
    width=400,
    height=400,
    # by='Method',
    # legend='top_right',
    # backend='matplotlib',
    # s=4000,
    xlabel=r"Federated p-value",  # LaTeX-escaped #
    ylabel=r"Meta-Analysis p-value",
    marker=["+", "x", "^", "v"],
    borderpad=100,
    # linestyle=['dashed', 'dotted']
    # title=f'{"Client" if i == 1 else "Clients"}'
)

plot = plot.opts(
    legend_opts={
        # "title": "Correctness",
        "ncol": 2,  # two columns
        "loc": "upper center",  # position above plot
        "bbox_to_anchor": (0.05, 1.25),  # fine-tune vertical placement
        "borderpad": 1.2,
        "labelspacing": 0.9,
        "frameon": False,  # usually looks cleaner for LaTeX
    }
)

_render = hv.render(plot, backend="matplotlib")
ax = _render.axes[0]
legend = ax.get_legend()
if legend:
    legend.set_title("Correctness")
    legend.set_bbox_to_anchor((0.05, 1.25))
_render.savefig(
    f"images/{target_folder}/scatter-fedci-v-fisher-colored-small.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


# plot independencies and dependencies separately

__df = _df.filter(pl.col("MSep"))

plot = __df.sort("Correctness").hvplot.scatter(
    x="Federated",
    y="Meta-Analysis",
    by="Correctness",
    color="color",
    alpha=0.7,
    ylim=(-0.01, 1.01),
    xlim=(-0.01, 1.01),
    width=400,
    height=400,
    # by='Method',
    # legend='top_right',
    # backend='matplotlib',
    # s=4000,
    xlabel=r"Federated p-value",  # LaTeX-escaped #
    ylabel=r"Meta-Analysis p-value",
    marker=["+", "x", "^", "v"],
    borderpad=100,
    # linestyle=['dashed', 'dotted']
    # title=f'{"Client" if i == 1 else "Clients"}'
)

plot = plot.opts(
    legend_opts={
        # "title": "Correctness",
        "ncol": 2,  # two columns
        "loc": "upper center",  # position above plot
        "bbox_to_anchor": (0.05, 1.25),  # fine-tune vertical placement
        "borderpad": 1.2,
        "labelspacing": 0.9,
        "frameon": False,  # usually looks cleaner for LaTeX
    }
)

_render = hv.render(plot, backend="matplotlib")
ax = _render.axes[0]
legend = ax.get_legend()
if legend:
    legend.set_title("Correctness")
    legend.set_bbox_to_anchor((0.05, 1.25))
_render.savefig(
    f"images/{target_folder}/scatter-fedci-v-fisher-independent.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)

__df = _df.filter(~pl.col("MSep"))

plot = __df.sort("Correctness").hvplot.scatter(
    x="Federated",
    y="Meta-Analysis",
    by="Correctness",
    color="color",
    alpha=0.7,
    ylim=(-0.01, 1.01),
    xlim=(-0.01, 1.01),
    width=400,
    height=400,
    # by='Method',
    # legend='top_right',
    # backend='matplotlib',
    # s=4000,
    xlabel=r"Federated p-value",  # LaTeX-escaped #
    ylabel=r"Meta-Analysis p-value",
    marker=["+", "x", "^", "v"],
    borderpad=100,
    # linestyle=['dashed', 'dotted']
    # title=f'{"Client" if i == 1 else "Clients"}'
)

plot = plot.opts(
    legend_opts={
        # "title": "Correctness",
        "ncol": 2,  # two columns
        "loc": "upper center",  # position above plot
        "bbox_to_anchor": (0.05, 1.25),  # fine-tune vertical placement
        "borderpad": 1.2,
        "labelspacing": 0.9,
        "frameon": False,  # usually looks cleaner for LaTeX
    }
)

_render = hv.render(plot, backend="matplotlib")
ax = _render.axes[0]
legend = ax.get_legend()
if legend:
    legend.set_title("Correctness")
    legend.set_bbox_to_anchor((0.05, 1.25))
_render.savefig(
    f"images/{target_folder}/scatter-fedci-v-fisher-dependent.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)

__df = __df.filter((pl.col("Federated") <= 0.1) & (pl.col("Meta-Analysis") <= 0.1))

plot = __df.sort("Correctness").hvplot.scatter(
    x="Federated",
    y="Meta-Analysis",
    by="Correctness",
    color="color",
    alpha=0.7,
    ylim=(-0.001, 0.101),
    xlim=(-0.001, 0.101),
    width=400,
    height=400,
    # by='Method',
    # legend='top_right',
    # backend='matplotlib',
    # s=4000,
    xlabel=r"Federated p-value",  # LaTeX-escaped #
    ylabel=r"Meta-Analysis p-value",
    marker=["+", "x", "^", "v"],
    borderpad=100,
    # linestyle=['dashed', 'dotted']
    # title=f'{"Client" if i == 1 else "Clients"}'
)

plot = plot.opts(
    legend_opts={
        # "title": "Correctness",
        "ncol": 2,  # two columns
        "loc": "upper center",  # position above plot
        "bbox_to_anchor": (0.05, 1.25),  # fine-tune vertical placement
        "borderpad": 1.2,
        "labelspacing": 0.9,
        "frameon": False,  # usually looks cleaner for LaTeX
    }
)

_render = hv.render(plot, backend="matplotlib")
ax = _render.axes[0]
legend = ax.get_legend()
if legend:
    legend.set_title("Correctness")
    legend.set_bbox_to_anchor((0.05, 1.25))
_render.savefig(
    f"images/{target_folder}/scatter-fedci-v-fisher-dependent-small.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)


# # vs pooled scatter

# _df = _df.with_columns(
#     Correctness=pl.when(
#         (pl.col("MSep") == pl.col("indep_fisher"))
#         & (pl.col("MSep") == pl.col("indep_pooled"))
#     )
#     .then(pl.lit("Both correct"))
#     .when(pl.col("MSep") == pl.col("indep_fisher"))
#     .then(pl.lit("Fisher correct"))
#     .when(pl.col("MSep") == pl.col("indep_pooled"))
#     .then(pl.lit("Pooled correct"))
#     .otherwise(pl.lit("Both incorrect"))
# )

# plot = _df.sort("Correctness").hvplot.scatter(
#     x="Pooled",
#     y="Meta-Analysis",
#     by="Correctness",
#     # color="color",
#     alpha=0.7,
#     ylim=(-0.01, 1.01),
#     xlim=(-0.01, 1.01),
#     width=400,
#     height=400,
#     # by='Method',
#     # legend='top_left',
#     # backend='matplotlib',
#     # s=4000,
#     xlabel=r"Pooled p-value",  # LaTeX-escaped #
#     ylabel=r"Meta-Analysis p-value",
#     # marker=["+", "x", "^", "v"],
#     # linestyle=['dashed', 'dotted']
#     # title=f'{"Client" if i == 1 else "Clients"}'
# )
# _render = hv.render(plot, backend="matplotlib")
# ax = _render.axes[0]
# legend = ax.get_legend()
# if legend:
#     legend.set_title("Correctness")
# _render.savefig(
#     f"images/{target_folder}/scatter-fisher-v-pooled-full.svg",
#     format="svg",
#     bbox_inches="tight",
#     dpi=300,
# )


# _df = _df.with_columns(
#     Correctness=pl.when(
#         (pl.col("MSep") == pl.col("indep_fedci"))
#         & (pl.col("MSep") == pl.col("indep_pooled"))
#     )
#     .then(pl.lit("Both correct"))
#     .when(pl.col("MSep") == pl.col("indep_fedci"))
#     .then(pl.lit("FedCI correct"))
#     .when(pl.col("MSep") == pl.col("indep_pooled"))
#     .then(pl.lit("Pooled correct"))
#     .otherwise(pl.lit("Both incorrect"))
# )
# plot = _df.sort("Correctness").hvplot.scatter(
#     x="Pooled",
#     y="Federated",
#     by="Correctness",
#     # color="color",
#     alpha=0.7,
#     ylim=(-0.01, 1.01),
#     xlim=(-0.01, 1.01),
#     width=400,
#     height=400,
#     # by='Method',
#     # legend='top_left',
#     # backend='matplotlib',
#     # s=4000,
#     xlabel=r"Pooled p-value",  # LaTeX-escaped #
#     ylabel=r"Federated p-value",
#     # marker=["+", "x", "^", "v"],
#     # linestyle=['dashed', 'dotted']
#     # title=f'{"Client" if i == 1 else "Clients"}'
# )
# _render = hv.render(plot, backend="matplotlib")
# ax = _render.axes[0]
# legend = ax.get_legend()
# if legend:
#     legend.set_title("Correctness")
# _render.savefig(
#     f"images/{target_folder}/scatter-fedci-v-pooled-full.svg",
#     format="svg",
#     bbox_inches="tight",
#     dpi=300,
# )


# TABLE ANALYSIS - VENN DIAGRAM

_df = df

_df = _df.with_columns(
    MSep=pl.when(pl.col("MSep"))
    .then(pl.lit("Independent"))
    .otherwise(pl.lit("Dependent"))
)

print(_df.columns)

print(
    _df.group_by("MSep", "correct_fisher", "correct_fedci", "correct_pooled")
    .len()
    .sort("MSep", "correct_fisher", "correct_fedci", "correct_pooled")
    .with_columns(
        perc=pl.col("len") / pl.sum("len").over("MSep"),
        perc2=pl.col("len") / pl.sum("len"),
    )
)
