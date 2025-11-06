import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd
import scipy

import datetime

import fedci

from collections import OrderedDict
import itertools
import os

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

import graphviz

arrow_type_lookup = {1: "odot", 2: "normal", 3: "none"}

# supress R log
import rpy2.rinterface_lib.callbacks as cb

cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

ro.r["source"]("./ci_functions.r")
aggregate_ci_results_f = ro.globalenv["aggregate_ci_results"]
run_ci_test_f = ro.globalenv["run_ci_test"]
get_data_f = ro.globalenv["get_data"]

load_pags = ro.globalenv["load_pags"]
truePAGs, subsetsList = load_pags()

ALPHA = 0.05
NUM_SAMPLES = [10_000]
SPLITS = [[1, 1, 1, 1]]  # , [2,1], [1,2], [3,1], [1,3], [1,1,1,1], [2,2,1,1]]
COEF_THRESHOLD = 0.2  # 0.1

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

import random


def get_data(true_pag, num_samples, all_labels):
    # var_types = {'A': 'continuous', 'B': 'continuous', 'C': 'nominal', 'D': 'nominal', 'E': 'continuous'}
    # var_levels = [1,1,3,3,1]

    potential_var_types = {
        "continuous": [1],
        "binary": [2],
        "ordinal": [4],
        "nominal": [4],
    }
    var_types = {}
    var_levels = []
    for label in sorted(all_labels):
        var_type = random.choice(list(potential_var_types.keys()))
        var_types[label] = var_type
        var_levels += [random.choice(potential_var_types[var_type])]

    dat = get_data_f(true_pag, num_samples, var_levels, "mixed", COEF_THRESHOLD)
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(dat[0])
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


def run_pval_agg_iod(dfs, client_labels, alpha=0.05, procedure="original"):
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        label_list = [ro.StrVector(v) for v in client_labels]

        result = aggregate_ci_results_f(label_list, r_dfs, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]
        gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
        gi_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["Gi_PAG_Label_List"].items()
        ]

        found_correct_pag = bool(result["found_correct_pag"][0])
        g_pag_shd = [x[1][0].item() for x in result["G_PAG_SHD"].items()]
        g_pag_for = [x[1][0].item() for x in result["G_PAG_FOR"].items()]
        g_pag_fdr = [x[1][0].item() for x in result["G_PAG_FDR"].items()]

    return (
        g_pag_list,
        g_pag_labels,
        gi_pag_list,
        gi_pag_labels,
        {
            "found_correct": found_correct_pag,
            "SHD": g_pag_shd,
            "FOR": g_pag_for,
            "FDR": g_pag_fdr,
            "MEAN_SHD": sum(g_pag_shd) / len(g_pag_shd) if len(g_pag_shd) > 0 else None,
            "MEAN_FOR": sum(g_pag_for) / len(g_pag_for) if len(g_pag_for) > 0 else None,
            "MEAN_FDR": sum(g_pag_fdr) / len(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
            "MIN_SHD": min(g_pag_shd) if len(g_pag_shd) > 0 else None,
            "MIN_FOR": min(g_pag_for) if len(g_pag_for) > 0 else None,
            "MIN_FDR": min(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
            "MAX_SHD": max(g_pag_shd) if len(g_pag_shd) > 0 else None,
            "MAX_FOR": max(g_pag_for) if len(g_pag_for) > 0 else None,
            "MAX_FDR": max(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
        },
    )


def run_riod(df, labels, client_labels, alpha=0.05, procedure="original"):
    # let index start with 1
    df.index += 1

    label_list = [ro.StrVector(v) for v in client_labels.values()]
    users = list(client_labels.keys())

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        suff_stat = [
            ("citestResults", ro.conversion.get_conversion().py2rpy(df)),
            ("all_labels", ro.StrVector(labels)),
        ]
        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)

        result = iod_on_ci_data_f(label_list, suff_stat, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]
        g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
        gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
        gi_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["Gi_PAG_Label_List"].items()
        ]
        gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

        found_correct_pag = bool(result["found_correct_pag"][0])

        g_pag_shd = [x[1][0].item() for x in result["G_PAG_SHD"].items()]
        g_pag_for = [x[1][0].item() for x in result["G_PAG_FOR"].items()]
        g_pag_fdr = [x[1][0].item() for x in result["G_PAG_FDR"].items()]

    return (
        g_pag_list,
        g_pag_labels,
        gi_pag_list,
        gi_pag_labels,
        {
            "found_correct": found_correct_pag,
            "SHD": g_pag_shd,
            "FOR": g_pag_for,
            "FDR": g_pag_fdr,
            "MEAN_SHD": sum(g_pag_shd) / len(g_pag_shd) if len(g_pag_shd) > 0 else None,
            "MEAN_FOR": sum(g_pag_for) / len(g_pag_for) if len(g_pag_for) > 0 else None,
            "MEAN_FDR": sum(g_pag_fdr) / len(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
            "MIN_SHD": min(g_pag_shd) if len(g_pag_shd) > 0 else None,
            "MIN_FOR": min(g_pag_for) if len(g_pag_for) > 0 else None,
            "MIN_FDR": min(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
            "MAX_SHD": max(g_pag_shd) if len(g_pag_shd) > 0 else None,
            "MAX_FOR": max(g_pag_for) if len(g_pag_for) > 0 else None,
            "MAX_FDR": max(g_pag_fdr) if len(g_pag_fdr) > 0 else None,
        },
    )


def mxm_ci_test(df):
    df = df.with_columns(cs.string().cast(pl.Categorical()))
    df = df.to_pandas()
    with (ro.default_converter + pandas2ri.converter).context():
        df_r = ro.conversion.get_conversion().py2rpy(df)
        result = run_ci_test_f(df_r, 999, "./examples/", "dummy")
        df_pvals = ro.conversion.get_conversion().rpy2py(result["citestResults"])
        labels = list(result["labels"])
    return df_pvals, labels


def server_results_to_dataframe(server, labels, results):
    likelihood_ratio_tests = server.get_likelihood_ratio_tests()

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
                test.p_val,
            )
        )

    df = pd.DataFrame(data=rows, columns=columns)
    return pl.from_pandas(df)


def test_dataset(dfs):
    # p values for pooled data
    df1 = pl.concat([d for i, d in enumerate(dfs) if i % 2 == 0])
    df2 = pl.concat([d for i, d in enumerate(dfs) if i % 2 == 1])
    labels_intersect = sorted(list(set(df1.columns) & set(df2.columns)))
    df_intersect = pl.concat(
        [df1.select(labels_intersect), df2.select(labels_intersect)]
    )

    df1r, df1l = mxm_ci_test(df1)
    df2r, df2l = mxm_ci_test(df2)
    df_intersectr, df_intersectl = mxm_ci_test(df_intersect)

    df1r = pl.from_pandas(df1r)
    label_mapping = {str(i): l for i, l in enumerate(df1l, start=1)}
    df1r = df1r.with_columns(
        pl.col("X").cast(pl.Utf8).replace(label_mapping),
        pl.col("Y").cast(pl.Utf8).replace(label_mapping),
        pl.col("S")
        .str.split(",")
        .list.eval(pl.element().replace(label_mapping))
        .list.sort()
        .list.join(","),
    )

    df2r = pl.from_pandas(df2r)
    label_mapping = {str(i): l for i, l in enumerate(df2l, start=1)}
    df2r = df2r.with_columns(
        pl.col("X").cast(pl.Utf8).replace(label_mapping),
        pl.col("Y").cast(pl.Utf8).replace(label_mapping),
        pl.col("S")
        .str.split(",")
        .list.eval(pl.element().replace(label_mapping))
        .list.sort()
        .list.join(","),
    )

    df_intersectr = pl.from_pandas(df_intersectr)
    label_mapping = {str(i): l for i, l in enumerate(df_intersectl, start=1)}
    df_intersectr = df_intersectr.with_columns(
        pl.col("X").cast(pl.Utf8).replace(label_mapping),
        pl.col("Y").cast(pl.Utf8).replace(label_mapping),
        pl.col("S")
        .str.split(",")
        .list.eval(pl.element().replace(label_mapping))
        .list.sort()
        .list.join(","),
    )

    df1r = df1r.with_columns(
        X=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("Y")).otherwise(pl.col("X")),
        Y=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("X")).otherwise(pl.col("Y")),
    )
    df2r = df2r.with_columns(
        X=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("Y")).otherwise(pl.col("X")),
        Y=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("X")).otherwise(pl.col("Y")),
    )
    df_intersectr = df_intersectr.with_columns(
        X=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("Y")).otherwise(pl.col("X")),
        Y=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("X")).otherwise(pl.col("Y")),
    )

    df1r = df1r.join(df_intersectr, on=["ord", "X", "Y", "S"], how="anti")
    df2r = df2r.join(df_intersectr, on=["ord", "X", "Y", "S"], how="anti")

    df_pooled = pl.concat([df1r, df2r, df_intersectr])
    df_pooled = df_pooled.rename({"pvalue": "pvalue_pooled"})

    # df_pooled = df_pooled.with_columns(
    #     X=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("Y")).otherwise(pl.col("X")),
    #     Y=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("X")).otherwise(pl.col("Y")),
    # )

    # STEP 2: IOD WITH META-ANALYSIS
    # -> Load results df and just get pvalues from there
    # pvalue_df_meta = pvalue_df.select('ord', 'X', 'Y', 'S', pvalue=pl.col('pvalue_fisher'))
    # meta_dfs, meta_labels = zip(*[mxm_ci_test(d) for d in dfs])

    client_ci_info = [mxm_ci_test(d) for d in dfs]
    client_ci_dfs, client_ci_labels = zip(*client_ci_info)

    client_dfs = []
    for ci_df, ci_labels in zip(client_ci_dfs, client_ci_labels):
        ci_df = pl.from_pandas(ci_df)
        label_mapping = {str(i): l for i, l in enumerate(ci_labels, start=1)}
        ci_df = ci_df.with_columns(
            pl.col("X").cast(pl.Utf8).replace(label_mapping),
            pl.col("Y").cast(pl.Utf8).replace(label_mapping),
            pl.col("S")
            .str.split(",")
            .list.eval(pl.element().replace(label_mapping))
            .list.sort()
            .list.join(","),
        )
        client_dfs.append(ci_df)

    fisher_df = pl.concat(client_dfs)

    fisher_df = fisher_df.with_columns(
        X=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("Y")).otherwise(pl.col("X")),
        Y=pl.when(pl.col("X") > pl.col("Y")).then(pl.col("X")).otherwise(pl.col("Y")),
    )

    # print(
    #     fisher_df.filter(pl.col("X") == "A")
    #     .filter(pl.col("Y") == "B")
    #     .filter(pl.col("ord") == 0)
    # )
    fisher_df = fisher_df.group_by(["ord", "X", "Y", "S"]).agg(pl.col("pvalue"))

    fisher_df = fisher_df.with_columns(
        DOFs=2 * pl.col("pvalue").list.len(),
        T=-2 * (pl.col("pvalue").list.eval(pl.element().log()).list.sum()),
    )

    fisher_df = fisher_df.with_columns(
        pvalue_fisher=pl.struct(["DOFs", "T"]).map_elements(
            lambda row: scipy.stats.chi2.sf(row["T"], row["DOFs"]),
            return_dtype=pl.Float64,
        )
    ).drop("DOFs", "T", "pvalue")

    # STEP 3: IOD WITH FEDCI
    # -> Load results df and just get pvalues from there
    server = fedci.Server({str(i): fedci.Client(d) for i, d in enumerate(dfs, start=1)})
    fedci_results = server.run()
    fedci_all_labels = sorted(list(server.schema.keys()))
    fedci_df = server_results_to_dataframe(server, fedci_all_labels, fedci_results)

    label_mapping = {str(i): l for i, l in enumerate(fedci_all_labels, start=1)}
    fedci_df = fedci_df.with_columns(
        pl.col("X").cast(pl.Utf8).replace(label_mapping),
        pl.col("Y").cast(pl.Utf8).replace(label_mapping),
        pl.col("S")
        .str.split(",")
        .list.eval(pl.element().replace(label_mapping))
        .list.sort()
        .list.join(","),
    )

    fedci_df = fedci_df.rename({"pvalue": "pvalue_fedci"})

    return df_pooled, fedci_df, fisher_df


def split_data(df, splits, labels1, labels2):
    dfs = []
    for split in splits:
        _dfs = []
        for i in range(1, len(split) + 1):
            from_idx = int(sum(split[: i - 1]) / sum(split) * len(df))
            to_idx = int(sum(split[:i]) / sum(split) * len(df))
            # print(i, len(df), from_idx, to_idx)
            df_i = df[from_idx:to_idx]
            if i % 2 == 0:
                df_i = df_i.select(labels1)
            else:
                df_i = df_i.select(labels2)
            _dfs.append(df_i)

        dfs.append((split, _dfs))
    return dfs


SAVE_DIR = "experiments/mixed_pag_pvalues3"

from tqdm import tqdm

import os

# done_experiments = os.listdir(SAVE_DIR)
# done_experiments = [
#     int(f.split("-")[0]) for f in done_experiments if f.endswith(".parquet")
# ]
# from collections import Counter

# cnt = Counter(done_experiments)

# missing_three_tail_pags = []
# for k, v in cnt.items():
#     if v < 10:
#         missing_three_tail_pags.append(k)
# three_tail_pags = missing_three_tail_pags

# print(cnt)
# print(three_tail_pags)
# asd
# print(cnt)
# asd

# three_tail_pags = [52]
# three_tail_pags = [82]


# three_tail_pags = [56]
# three_tail_pags = [65]

for _ in tqdm(range(10), position=0):
    for num_samples in tqdm(NUM_SAMPLES, position=1, leave=False):
        for true_pag_id in tqdm(three_tail_pags, position=2, leave=False):
            # print(f'Running for pag {true_pag_id} with {num_samples} samples')

            # load pag and labels per partition
            true_pag = truePAGs[true_pag_id]
            labels1, labels2 = subsetsList[true_pag_id]
            all_labels = sorted(list(set(labels1) | set(labels2)))
            intersection_labels = sorted(list(set(labels1) & set(labels2)))

            df_msep = (
                pl.read_parquet(f"experiments/pag_msep/pag-{true_pag_id}.parquet")
                .with_columns(pl.col("S").list.join(","))
                .with_columns(
                    ord=pl.when(pl.col("S").str.len_chars() == 0)
                    .then(pl.lit(0))
                    .otherwise(pl.col("S").str.count_matches(",") + 1)
                )
            )

            df = get_data(true_pag, num_samples, all_labels)
            dfs = split_data(df, SPLITS, labels1, labels2)

            split, dfs = dfs[0]

            identifier = (
                f"{true_pag_id}-{num_samples}-{'_'.join([str(s) for s in split])}"
            )

            test_results = test_dataset(dfs)

            result_pooled, result_fedci, result_fisher = test_results

            # print(result_fisher.filter(pl.col("X") == "A").filter(pl.col("ord") == 0))

            combined_df = (
                result_pooled.join(result_fedci, on=["ord", "X", "Y", "S"], how="left")
                .join(result_fisher, on=["ord", "X", "Y", "S"], how="left")
                .join(df_msep, on=["ord", "X", "Y", "S"], how="inner")
            )

            # print(len(combined_df))
            # print("")
            # pl.Config.set_tbl_rows(50)
            # print(combined_df.sort("ord", "X", "Y", "S"))
            # print(
            #     combined_df.filter(pl.col("pvalue_fedci").is_null()).sort(
            #         "ord", "X", "Y", "S"
            #     )
            # )
            # print(combined_df.null_count())
            now = int(datetime.datetime.utcnow().timestamp() * 1e3)
            combined_df.write_parquet(f"{SAVE_DIR}/{identifier}-{now}.parquet")
