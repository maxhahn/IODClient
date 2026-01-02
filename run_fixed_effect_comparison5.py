import itertools
import os
import random
import time

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import rpy2

pl.Config.set_tbl_rows(80)
# supress R log
import rpy2.rinterface_lib.callbacks as cb
import rpy2.robjects as ro
import scipy
from rpy2.robjects import numpy2ri, pandas2ri

import fedci

cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

ro.r["source"]("./ci_functions2.r")
get_data_f = ro.globalenv["get_data"]
get_data_for_single_pag_f = ro.globalenv["get_data_for_single_pag"]
run_ci_test_f = ro.globalenv["run_ci_test"]
run_ci_test2_f = ro.globalenv["run_ci_test2"]
msep_f = ro.globalenv["msep"]
load_pags = ro.globalenv["load_pags"]
ci_mxm_test_f = ro.globalenv["ci_mxm_test"]


test_pags, label_splits = load_pags()
label_splits = [(sorted(tuple(x[0])), sorted(tuple(x[1]))) for x in label_splits]


def floatmatrix_to_nparray(r_floatmatrix):
    numpy_matrix = numpy2ri.rpy2py(r_floatmatrix)
    return numpy_matrix.astype(int)  # .tolist()


test_pags = [floatmatrix_to_nparray(pag) for pag in test_pags]


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
assert len(three_tail_pags) == 30

test_pag_mapping = {i: pag for i, pag in enumerate(test_pags) if i in three_tail_pags}
test_label_split_mapping = {
    i: label_splits
    for i, label_splits in enumerate(label_splits)
    if i in three_tail_pags
}
df_msep_mapping = {
    i: pl.read_parquet(f"experiments/pag_msep/pag-{i}.parquet")
    .with_columns(pl.col("S").list.join(","))
    .with_columns(
        ord=pl.when(pl.col("S").str.len_chars() == 0)
        .then(pl.lit(0))
        .otherwise(pl.col("S").str.count_matches(",") + 1)
    )
    for i in three_tail_pags
}

ALPHA = 0.05
COEF_THRESHOLD = 0.2  # 0.1


def get_data(pag, num_samples, var_types, var_levels, seed):
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        dat = get_data_f(
            pag,
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

    # lrt_ord_0 = [
    #     (lrt.v0, lrt.v1)
    #     for lrt in likelihood_ratio_tests
    #     if len(lrt.conditioning_set) == 0
    # ]
    # label_combinations = itertools.combinations(labels, 2)
    # missing_base_rows = []
    # for label_combination in label_combinations:
    #     if label_combination in lrt_ord_0:
    #         continue
    #     # print('MISSING', label_combination)
    #     l0, l1 = label_combination
    #     missing_base_rows.append((0, labels.index(l0) + 1, labels.index(l1) + 1, "", 1))
    # rows += missing_base_rows

    for test in likelihood_ratio_tests:
        s_labels_string = ",".join(sorted(test.conditioning_set))
        rows.append(
            (
                len(test.conditioning_set),
                test.v0,
                test.v1,
                s_labels_string,
                test.p_value,
            )
        )

    df = pd.DataFrame(data=rows, columns=columns)
    return pl.from_pandas(df)


def replace_idx_with_varnames(df, labels):
    mapping = {str(i): l for i, l in enumerate(labels, start=1)}
    df = df.with_columns(
        pl.col("X").cast(pl.Utf8).replace(mapping),
        pl.col("Y").cast(pl.Utf8).replace(mapping),
        pl.col("S")
        .str.split(",")
        .list.eval(pl.element().replace(mapping))
        .list.sort()
        .list.join(","),
    )
    return df


def remove_client_from_pooled_result(df, labels):
    df = pl.from_pandas(df)
    client_var_idx = str(labels.index("CLIENT") + 1)  # 1 indexed
    df = df.filter(
        (
            ~(
                (pl.col("X").cast(pl.Utf8) == client_var_idx)
                | (pl.col("Y").cast(pl.Utf8) == client_var_idx)
            )
            & pl.col("S")
            .str.split(",")
            # .list.filter(pl.element().str.len_chars() > 0)
            # .cast(pl.List(pl.Int32))
            .list.contains(client_var_idx)
        )
    )

    df = df.with_columns(
        pl.col("S")
        .str.split(",")
        .list.filter(pl.element() != client_var_idx)
        .list.join(","),
        pl.col("ord") - 1,
    )
    return replace_idx_with_varnames(df, labels)


def test_pooled_data(dfs, labels):
    # all_labels = set.union(*[set(l) for l in labels])
    intersection_labels = sorted(list(set.intersection(*[set(l) for l in labels])))

    intersection_df = pl.concat(
        [df.select(intersection_labels + ["CLIENT"]) for df in dfs]
    )

    l1 = labels[0]
    l2 = labels[1]

    l1_dfs = []
    l2_dfs = []
    for i, df in enumerate(dfs):
        if i % 2 == 0:
            l1_dfs.append(df.select(l1 + ["CLIENT"]))
        else:
            l2_dfs.append(df.select(l2 + ["CLIENT"]))

    print("l1", l1, "l2", l2)

    l2_result_df = remove_client_from_pooled_result(*mxm_ci_test(pl.concat(l2_dfs)))
    cb.consolewrite_print = lambda x: None
    cb.consolewrite_warnerror = lambda x: None

    l1_result_df = remove_client_from_pooled_result(*mxm_ci_test(pl.concat(l1_dfs)))
    intersection_result_df = remove_client_from_pooled_result(
        *mxm_ci_test(intersection_df)
    )

    l1_result_df = l1_result_df.join(
        intersection_result_df, on=["ord", "X", "Y", "S"], how="anti"
    )
    l2_result_df = l2_result_df.join(
        intersection_result_df, on=["ord", "X", "Y", "S"], how="anti"
    )
    result_df = pl.concat([intersection_result_df, l1_result_df, l2_result_df])
    return result_df


def test_dataset(df, labels):
    dfs = df.sort("CLIENT", pl.exclude("CLIENT")).partition_by("CLIENT")

    # with (ro.default_converter + pandas2ri.converter).context():
    #     # # load local-ci script
    #     # ro.r['source']('./local-ci.r')
    #     # # load function from R script
    #     # run_ci_test_f = ro.globalenv['run_ci_test']
    #     # converting it into r object for passing into r function
    #     df_r = ro.conversion.get_conversion().py2rpy(df.to_pandas())
    #     # Invoking the R function and getting the result
    #     result = ci_mxm_test_f(df_r, "B", "E", ["A", "C"])
    #     df_pvals = ro.conversion.get_conversion().rpy2py(result["citestResults"])
    #     print(result)
    # asd
    # Converting it back to a pandas dataframe.

    # s1 = fedci.Server([fedci.Client("x", df)])
    # x = s1.test("B", "E", ["A", "C", "CLIENT"])
    # print(x)
    # asd

    # STEP 1: Test with pooled data
    t0 = time.time()
    pooled_result_df = test_pooled_data(dfs, labels)
    t1 = time.time()

    dfs = [
        d.select(labels[0]) if i % 2 == 0 else d.select(labels[1])
        for i, d in enumerate(dfs)
    ]

    try:
        # STEP 2: Test with Meta Analysis
        meta_dfs, meta_df_labels = zip(*[mxm_ci_test(d) for d in dfs])
        meta_dfs = [
            replace_idx_with_varnames(pl.from_pandas(_df), _labels)
            for _df, _labels in zip(meta_dfs, meta_df_labels)
        ]

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
    t2 = time.time()
    # print(fisher_df)
    # STEP 3: Test with FedCI
    # client_dfs = df.partition_by("CLIENT")
    # client_dfs = [
    #     d.select(labels[0] + ["CLIENT"])
    #     if i % 2 == 0
    #     else d.select(labels[1] + ["CLIENT"])
    #     for i, d in enumerate(client_dfs)
    # ]
    # for cdf in client_dfs:
    #     print(cdf.head(1))

    # server = fedci.Server(
    #     [fedci.Client(str(i), d) for i, d in enumerate(client_dfs, start=1)]
    # )

    server = fedci.Server([fedci.Client(str(i), d) for i, d in enumerate(dfs, start=1)])

    # x = server.test("B", "C", ["E"])
    # x = server.test("A", "C", ["B", "E"])
    x = server.test("A", "C", ["B", "E"])
    print(x)
    asd

    # server.test("A", "B", ["E"])
    # x = server.test("B", "E", ["A", "C"])
    # server.test("B", "E", ["A", "C", "CLIENT"])
    fedci_results = server.run()
    t3 = time.time()
    fedci_all_labels = sorted(list(server.schema.keys()))

    fedci_df = server_results_to_dataframe(fedci_all_labels, fedci_results).sort(
        "ord", "X", "Y", "S"
    )

    return pooled_result_df, fisher_df, fedci_df, t1 - t0, t2 - t1, t3 - t2


data_dir = "experiments/fixed_effect_data/sim"


# seed = random.randint(0, 100000)
seed_start = 10031
num_runs = 20
SEEDS = range(seed_start, seed_start + num_runs)
SEEDS = [10018]  # 10001
SAMPLES = [500]  # A B C,E 12 2500 || C D A,E 12 500
CLIENTS = [12]
# SAMPLES = [5000, 10000]

"""
╞═════╪═════╪═════╪════════╪═══════╪════════════╪═════════════╪═══════════════╪══════════════╪═══════════════════╪══════════════════╪═════════════╪════════════╡
│ B   ┆ E   ┆ C   ┆ 61     ┆ 10023 ┆ 8          ┆ 1000        ┆ 0.552412      ┆ 0.0          ┆ -0.593462         ┆ -34.538776       ┆ -0.552412   ┆ -33.945315 │
│ A   ┆ C   ┆ B,E ┆ 61     ┆ 10018 ┆ 12         ┆ 500         ┆ 0.119371      ┆ 4.0427e-92   ┆ -2.125517         ┆ -34.538776       ┆ -0.119371   ┆ -32.413259 │
│ A   ┆ B   ┆ C   ┆ 61     ┆ 10023 ┆ 8          ┆ 1000        ┆ 0.113433      ┆ 0.0          ┆ -2.176541         ┆ -34.538776       ┆ -0.113433   ┆ -32.362236 │
│ A   ┆ D   ┆ C,E ┆ 61     ┆ 10026 ┆ 4          ┆ 2500        ┆ 0.001568      ┆ 6.0740e-8    ┆ -6.458134         ┆ -16.616666       ┆ -0.001568   ┆ -10.158532
"""

pag_ids_to_test = [61]

POTENTIAL_VAR_LEVELS = [
    ("continuous", 1),
    ("binary", 2),
    ("nominal", 4),
    ("ordinal", 4),
]

"""

A   ┆ C   ┆ B,E ┆ 61     ┆ 10003 ┆ 4          ┆ 1000        ┆ -0.465803
"""

from tqdm import tqdm

for seed in tqdm(SEEDS, position=0, leave=True):
    for pag_id in tqdm(pag_ids_to_test, position=1, leave=False):
        for num_clients in tqdm(CLIENTS, position=2, leave=False):
            for num_samples in tqdm(SAMPLES, position=3, leave=False):
                random.seed(seed)
                base_pag = test_pag_mapping[pag_id]
                label_split = test_label_split_mapping[pag_id]

                df_msep = df_msep_mapping[pag_id]

                fixed_effect_pag = np.zeros(
                    (base_pag.shape[0] + 1, base_pag.shape[1] + 1)
                )
                fixed_effect_pag[:-1, :-1] = base_pag
                fixed_effect_pag[:-1, -1] = np.full(base_pag.shape[0], fill_value=3)
                fixed_effect_pag[-1, :-1] = np.full(base_pag.shape[0], fill_value=2)
                all_vars = sorted(list(set(label_split[0] + label_split[1])))
                var_types = {}
                var_levels = []
                for var in all_vars:
                    choice = random.choice(POTENTIAL_VAR_LEVELS)
                    var_types[var] = choice[0]
                    var_levels.append(choice[1])

                var_types["CLIENT"] = "nominal"
                var_levels.append(num_clients)

                df = get_data(
                    fixed_effect_pag, num_samples, var_types, var_levels, seed
                )

                print(df.head())

                df_pooled, df_fisher, df_fedci, time_pooled, time_fisher, time_fedci = (
                    test_dataset(df, label_split)
                )

                df_pooled = df_pooled.rename({"pvalue": "pooled_pvalue"}).with_columns(
                    pooled_runtime=pl.lit(time_pooled)
                )
                df_fisher = df_fisher.rename({"pvalue": "fisher_pvalue"}).with_columns(
                    fisher_runtime=pl.lit(time_fisher)
                )
                df_fedci = df_fedci.rename({"pvalue": "fedci_pvalue"}).with_columns(
                    fedci_runtime=pl.lit(time_fedci)
                )

                df_result = df_pooled.join(
                    df_fisher, on=["ord", "X", "Y", "S"], how="left"
                )
                df_result = df_result.join(
                    df_fedci, on=["ord", "X", "Y", "S"], how="left"
                )

                df_result = df_result.join(
                    df_msep, on=["ord", "X", "Y", "S"], how="left"
                )

                print(df_result.drop(cs.contains("runtime")))

                print(
                    df_result.filter(
                        (pl.col("pooled_pvalue") - pl.col("fedci_pvalue")).abs() > 0.1
                    ).drop(cs.contains("runtime"))
                )
                asd

                del var_types["CLIENT"]
                df_result = df_result.with_columns(
                    seed=pl.lit(seed),
                    pag_id=pl.lit(pag_id),
                    num_samples=pl.lit(num_samples),
                    partitions=pl.lit(num_clients),
                    vars1=label_split[0],
                    vars2=label_split[1],
                    var_types=var_types,
                )

                if not os.path.exists(f"{data_dir}"):
                    os.makedirs(f"{data_dir}")
                df_result.write_parquet(
                    f"{data_dir}/{seed}-id{pag_id}-s{num_samples}-c{num_clients}.parquet"
                )

"""
┌─────┬─────┬─────┬────────┬───────┬────────────┬─────────────┬───────────┐
│ X   ┆ Y   ┆ S   ┆ pag_id ┆ seed  ┆ partitions ┆ num_samples ┆ diff      │
│ --- ┆ --- ┆ --- ┆ ---    ┆ ---   ┆ ---        ┆ ---         ┆ ---       │
│ str ┆ str ┆ str ┆ i32    ┆ i32   ┆ i32        ┆ i32         ┆ f64       │
╞═════╪═════╪═════╪════════╪═══════╪════════════╪═════════════╪═══════════╡
│ A   ┆ C   ┆ B,E ┆ 61     ┆ 10003 ┆ 4          ┆ 1000        ┆ -0.465803 │
│ A   ┆ C   ┆ D,E ┆ 61     ┆ 10008 ┆ 12         ┆ 1000        ┆ -0.131782 │
│ A   ┆ E   ┆ D   ┆ 61     ┆ 10001 ┆ 12         ┆ 1000        ┆ -0.044453 │
│ A   ┆ D   ┆ E   ┆ 61     ┆ 10001 ┆ 4          ┆ 1000        ┆ -0.034744 │
│ A   ┆ B   ┆ C,E ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.05243   │
│ A   ┆ B   ┆     ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.05487   │
│ A   ┆ B   ┆ E   ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.057148  │
│ A   ┆ E   ┆ B,C ┆ 61     ┆ 10003 ┆ 4          ┆ 1000        ┆ 0.057256  │
│ A   ┆ B   ┆ C   ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.062084  │
│ B   ┆ E   ┆ A,C ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.093737  │
│ B   ┆ E   ┆ C   ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.09575   │
│ B   ┆ E   ┆ A   ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.149566  │
│ B   ┆ E   ┆     ┆ 61     ┆ 10007 ┆ 4          ┆ 1000        ┆ 0.15249   │
└─────┴─────┴─────┴────────┴───────┴────────────┴─────────────┴───────────┘
"""
