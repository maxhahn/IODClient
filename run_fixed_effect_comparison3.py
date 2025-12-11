import itertools
import os

import graphviz
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import rpy2
import rpy2.robjects as ro
import scipy
from rpy2.robjects import numpy2ri, pandas2ri

arrow_type_lookup = {1: "odot", 2: "normal", 3: "none"}

# supress R log
import rpy2.rinterface_lib.callbacks as cb

import fedci

cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

ro.r["source"]("./ci_functions2.r")
get_data_f = ro.globalenv["get_data"]
get_data_for_single_pag_f = ro.globalenv["get_data_for_single_pag"]
run_ci_test_f = ro.globalenv["run_ci_test"]
msep_f = ro.globalenv["msep"]

# 337009
ALPHA = 0.05
COEF_THRESHOLD = 0.2  # 0.1

# DF_MSEP = (
#     pl.read_parquet("experiments/pag_msep/pag-slides.parquet")
#     .with_columns(pl.col("S").list.join(","))
#     .with_columns(
#         ord=pl.when(pl.col("S").str.len_chars() == 0)
#         .then(pl.lit(0))
#         .otherwise(pl.col("S").str.count_matches(",") + 1)
#     )
# )


from itertools import chain, combinations


def is_m_separable(pag, labels):
    raw_labels = list(labels)
    label_set = set(labels)
    cnt = 0
    result = []
    for x in label_set:
        label_wo_x = label_set - {x}
        for y in label_wo_x:
            if x > y:
                continue
            conditioning_set = chain.from_iterable(
                combinations(label_wo_x - {y}, r) for r in range(0, len(label_wo_x))
            )
            for s in conditioning_set:
                cnt += 1
                with (
                    ro.default_converter + pandas2ri.converter + numpy2ri.converter
                ).context():
                    is_msep = msep_f(pag, raw_labels, x, y, list(s))
                r = {
                    "ord": len(s),
                    "X": x,
                    "Y": y,
                    "S": ",".join(sorted(list(s))),
                    "MSep": bool(is_msep[0]),
                }
                result.append(r)
                # print(x,y,s, bool(is_msep[0]))
    df = pl.from_dicts(result).sort("ord", "X", "Y", "S")
    return df


# Slide example
graph_type = "SLIDES_MIXED"
var_types = {
    "A": "binary",
    "B": "continuous",
    "C": "nominal",
    "D": "ordinal",
    "E": "continuous",
}
var_levels = [2, 1, 4, 4, 1]
TRUE_PAG = np.array(
    [
        [0, 0, 2, 2, 0],
        [0, 0, 2, 0, 0],
        [2, 1, 0, 2, 2],
        [2, 0, 3, 0, 2],
        [0, 0, 3, 3, 0],
    ]
)

# graph_type = "SLIDES"
# var_types = {
#     "A": "continuous",
#     "B": "continuous",
#     "C": "continuous",
#     "D": "continuous",
#     "E": "continuous",
# }
# var_levels = [1, 1, 1, 1, 1]
# TRUE_PAG = np.array(
#     [
#         [0, 0, 2, 2, 0],
#         [0, 0, 2, 0, 0],
#         [2, 1, 0, 2, 2],
#         [2, 0, 3, 0, 2],
#         [0, 0, 3, 3, 0],
#     ]
# )

# Simple examples
# var_types = {
#     "X": "continuous",
#     "Y": "nominal",
#     "Z": "continuous",
# }
# var_levels = [1, 4, 1]
# # Fork
# graph_type = "CONDITIONAL_INDEPENDENCE"
# TRUE_PAG = np.array(
#     [  # X  Y  Z
#         [0, 0, 3],
#         [0, 0, 3],
#         [2, 2, 0],
#     ]
# )

# # Collider
# graph_type = "CONDITIONAL_DEPENDENCE"
# TRUE_PAG = np.array(
#     [  # X  Y  Z
#         [0, 0, 2],
#         [0, 0, 2],
#         [3, 3, 0],
#     ]
# )

# # Chain
# graph_type = "CHAIN"
# TRUE_PAG = np.array(
#     [  # X  Y  Z
#         [0, 0, 2],
#         [0, 0, 3],
#         [3, 2, 0],
#     ]
# )

# var_types = {
#     "X": "continuous",
#     "Y": "continuous",
# }
# var_levels = [1, 1]
# graph_type = "MARGINAL_INDEPENDENCE"
# TRUE_PAG = np.array(
#     [  # X  Y
#         [0, 0],
#         [0, 0],
#     ]
# )
# graph_type = "MARGINAL_DEPENDENCE"
# TRUE_PAG = np.array(
#     [  # X  Y
#         [0, 2],
#         [3, 0],
#     ]
# )

DF_MSEP = is_m_separable(TRUE_PAG, var_types.keys())

FIXED_EFFECT_PAG = np.zeros((TRUE_PAG.shape[0] + 1, TRUE_PAG.shape[1] + 1))
FIXED_EFFECT_PAG[:-1, :-1] = TRUE_PAG
FIXED_EFFECT_PAG[:-1, -1] = np.full(TRUE_PAG.shape[0], fill_value=3)
FIXED_EFFECT_PAG[-1, :-1] = np.full(TRUE_PAG.shape[0], fill_value=2)
# FIXED_EFFECT_PAG[-1, -1] = np.full(TRUE_PAG.shape[0] + 1, fill_value=2)


def data2graph(data, labels):
    graph = graphviz.Digraph(format="png")
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            arrhead = int(data[i][j])
            arrtail = int(data[j][i])
            if arrhead == 0 or arrtail == 0:
                continue
            # print(f'Drawing {labels[i]} {arrow_type_lookup[arrtail]}-{arrow_type_lookup[arrhead]} {labels[j]}')
            graph.edge(
                labels[i],
                labels[j],
                arrowtail=arrow_type_lookup[arrtail],
                arrowhead=arrow_type_lookup[arrhead],
                dir="both",
            )
    return graph


# print(FIXED_EFFECT_PAG)
# print(data2graph(FIXED_EFFECT_PAG, list(var_types.keys()) + ["CLIENT"]))
# asd


def get_data(num_samples, seed):
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        dat = get_data_f(
            FIXED_EFFECT_PAG,
            num_samples,
            list(var_types.keys()),
            var_levels,
            "mixed",
            COEF_THRESHOLD,
            seed,
        )

        df = ro.conversion.get_conversion().rpy2py(pd.DataFrame(dat["dat"]))
    df = pl.from_pandas(df)
    for var_name, var_type in var_types.items():
        if var_type == "continuous":
            df = df.with_columns(pl.col(var_name).cast(pl.Float64))
        elif var_type == "binary":
            df = df.with_columns(pl.col(var_name) == "A")
        elif var_type == "ordinal":
            repl_dict = {"A": 1, "B": 2, "C": 3, "D": 4}
            df = df.with_columns(
                pl.col(var_name).cast(pl.Utf8).replace(repl_dict).cast(pl.Int32)
            )
        elif var_type == "nominal":
            df = df.with_columns(pl.col(var_name).cast(pl.Utf8))
    return df


def mxm_ci_test(df):
    df = df.with_columns(cs.string().cast(pl.Categorical()))
    df = df.to_pandas()
    with (ro.default_converter + pandas2ri.converter).context():
        # # load local-ci script
        # ro.r['source']('./local-ci.r')
        # # load function from R script
        # run_ci_test_f = ro.globalenv['run_ci_test']
        # converting it into r object for passing into r function
        df_r = ro.conversion.get_conversion().py2rpy(df)
        # Invoking the R function and getting the result
        result = run_ci_test_f(df_r, 999, "./examples/", "dummy")
        # Converting it back to a pandas dataframe.
        df_pvals = ro.conversion.get_conversion().rpy2py(result["citestResults"])
        labels = list(result["labels"])
    return df_pvals, labels


def server_results_to_dataframe(labels, results):
    likelihood_ratio_tests = results

    columns = ("ord", "X", "Y", "S", "pvalue")
    rows = []

    lrt_ord_0 = [
        (lrt.v0, lrt.v1)
        for lrt in likelihood_ratio_tests
        if len(lrt.conditioning_set) == 0
    ]
    label_combinations = itertools.combinations(labels, 2)
    missing_base_rows = []
    for label_combination in label_combinations:
        if label_combination in lrt_ord_0:
            continue
        # print('MISSING', label_combination)
        l0, l1 = label_combination
        missing_base_rows.append((0, labels.index(l0) + 1, labels.index(l1) + 1, "", 1))
    rows += missing_base_rows

    for test in likelihood_ratio_tests:
        s_labels_string = ",".join(
            sorted([str(labels.index(l) + 1) for l in test.conditioning_set])
        )
        rows.append(
            (
                len(test.conditioning_set),
                labels.index(test.v0) + 1,
                labels.index(test.v1) + 1,
                s_labels_string,
                test.p_value,
            )
        )

    df = pd.DataFrame(data=rows, columns=columns)
    return df


def test_dataset(df):
    # STEP 1: Test with pooled data
    pooled_result_df, _ = mxm_ci_test(df.drop("CLIENT"))
    pooled_result_df = pl.from_pandas(pooled_result_df).sort("ord", "X", "Y", "S")
    # print(pooled_result_df)

    dfs = df.partition_by("CLIENT")
    dfs = [d.drop("CLIENT") for d in dfs]

    try:
        # STEP 2: Test with Meta Analysis
        meta_dfs, _ = zip(*[mxm_ci_test(d) for d in dfs])
        meta_dfs = [pl.from_pandas(d) for d in meta_dfs]

        fisher_df = pl.concat(meta_dfs)
        fisher_df = fisher_df.group_by(["ord", "X", "Y", "S"]).agg(pl.col("pvalue"))

        fisher_df = fisher_df.with_columns(
            DOFs=2 * pl.col("pvalue").list.len(),
            T=-2 * (pl.col("pvalue").list.eval(pl.element().log()).list.sum()),
        )

        fisher_df = (
            fisher_df.with_columns(
                pvalue_fisher=pl.struct(["DOFs", "T"]).map_elements(
                    lambda row: scipy.stats.chi2.sf(row["T"], row["DOFs"]),
                    return_dtype=pl.Float64,
                )
            )
            .drop("DOFs", "T", "pvalue")
            .rename({"pvalue_fisher": "pvalue"})
            .sort("ord", "X", "Y", "S")
        )
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        fisher_df = pooled_result_df.with_columns(pvalue=pl.lit(None))
    # print(fisher_df)
    # STEP 3: Test with FedCI
    server = fedci.Server([fedci.Client(str(i), d) for i, d in enumerate(dfs, start=1)])

    fedci_results = server.run()
    fedci_all_labels = sorted(list(server.schema.keys()))
    # client_labels = {
    #     id: sorted(list(schema.keys())) for id, schema in server.client_schemas.items()
    # }
    fedci_df = server_results_to_dataframe(fedci_all_labels, fedci_results)
    fedci_df = pl.from_pandas(fedci_df).sort("ord", "X", "Y", "S")
    # print(fedci_df)
    return pooled_result_df, fisher_df, fedci_df


data_dir = "experiments/fixed_effect_data"
import random

# seed = random.randint(0, 100000)
seed_start = 10000
num_runs = 100
SEEDS = range(seed_start, seed_start + num_runs)
SAMPLES = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000]
# SAMPLES = [5000, 10000]

NUM_CLIENTS = 4

var_levels.append(NUM_CLIENTS)
var_types["CLIENT"] = "nominal"

from tqdm import tqdm

for seed in tqdm(SEEDS, position=0, leave=True):
    for num_samples in tqdm(SAMPLES, position=1, leave=False):
        df = get_data(num_samples, seed)
        # print(df.group_by("CLIENT").len())
        # continue
        df_pooled, df_fisher, df_fedci = test_dataset(df)

        df_pooled = df_pooled.with_columns(pooled_pvalue=pl.col("pvalue")).drop(
            "pvalue"
        )
        df_fisher = df_fisher.with_columns(fisher_pvalue=pl.col("pvalue")).drop(
            "pvalue"
        )
        df_fedci = df_fedci.with_columns(fedci_pvalue=pl.col("pvalue")).drop("pvalue")

        df_result = df_pooled.join(df_fisher, on=["ord", "X", "Y", "S"], how="left")
        df_result = df_result.join(df_fedci, on=["ord", "X", "Y", "S"], how="left")
        mapping = {str(i): l for i, l in enumerate(var_types.keys(), start=1)}
        df_result = df_result.with_columns(
            pl.col("X").cast(pl.Utf8).replace(mapping),
            pl.col("Y").cast(pl.Utf8).replace(mapping),
            pl.col("S")
            .str.split(",")
            .list.eval(pl.element().replace(mapping))
            .list.sort()
            .list.join(","),
        )

        df_result = df_result.join(DF_MSEP, on=["ord", "X", "Y", "S"], how="left")

        df_result = df_result.with_columns(
            seed=pl.lit(seed),
            num_samples=pl.lit(num_samples),
            graph=pl.lit(graph_type),
            partitions=pl.lit(NUM_CLIENTS),
            vars=list(var_types.keys()),
            var_types=list(var_types.values()),
        )

        if not os.path.exists(f"{data_dir}/{graph_type}"):
            os.makedirs(f"{data_dir}/{graph_type}")
        df_result.write_parquet(
            f"{data_dir}/{graph_type}/{seed}-s{num_samples}-c{NUM_CLIENTS}.parquet"
        )

# df.write_parquet("simple_test.parquet")

# df1 = df.filter(pl.col("CLIENT") == "A").drop("CLIENT")
# df2 = df.filter(pl.col("CLIENT") == "B").drop("CLIENT")

# df1.write_parquet(f"{data_dir}/{seed}-c1-1.parquet")
# df2.write_parquet(f"{data_dir}/{seed}-c2-1.parquet")
