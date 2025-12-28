import itertools
import os
import random
import time

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import rpy2

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
msep_f = ro.globalenv["msep"]
load_pags = ro.globalenv["load_pags"]


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

    columns = (
        "ord",
        "X",
        "Y",
        "S",
        "pvalue",
        "norm_X_res",
        "norm_X_unres",
        "norm_Y_res",
        "norm_Y_unres",
    )
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
        betas = test.get_betas()

        _betas = {}
        sorted_betas = []
        for key, beta in betas.items():
            resp_var, cond_set, _ = key
            if resp_var not in _betas:
                _betas[resp_var] = {}
            _betas[resp_var][cond_set] = beta
        for resp_var, test_dict in sorted(_betas.items()):
            for cond_set, value in sorted(test_dict.items(), key=lambda x: len(x[0])):
                value = np.sqrt(np.sum(value**2, axis=-1)).tolist()
                sorted_betas.append(value)
        lrt0_restricted = sorted_betas[0]
        lrt0_unrestricted = sorted_betas[1]
        lrt1_restricted = sorted_betas[2]
        lrt1_unrestricted = sorted_betas[3]

        s_labels_string = ",".join(sorted(test.conditioning_set))
        rows.append(
            (
                len(test.conditioning_set),
                test.v0,
                test.v1,
                s_labels_string,
                test.p_value,
                lrt0_restricted,
                lrt0_unrestricted,
                lrt1_restricted,
                lrt1_unrestricted,
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

    intersection_df = intersection_df.select(
        [
            col.name
            for col in intersection_df.select(pl.all().n_unique() > 1)
            if col.item()
        ]
    )

    l1_df = pl.concat(l1_dfs)
    l1_df = l1_df.select(
        [col.name for col in l1_df.select(pl.all().n_unique() > 1) if col.item()]
    )

    # if len(l1_df.columns) != 5:
    #     print("L1 lacks col")

    l2_df = pl.concat(l2_dfs)
    l2_df = l2_df.select(
        [col.name for col in l2_df.select(pl.all().n_unique() > 1) if col.item()]
    )
    # if len(l2_df.columns) != 5:
    #     print("L2 lacks col")

    intersection_result_df = remove_client_from_pooled_result(
        *mxm_ci_test(intersection_df)
    )
    # if len(intersection_df.columns) != 4:
    #     print("Linter lacks col")
    l1_result_df = remove_client_from_pooled_result(*mxm_ci_test(l1_df))
    l2_result_df = remove_client_from_pooled_result(*mxm_ci_test(l2_df))

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

    # STEP 1: Test with pooled data
    t0 = time.time()
    pooled_result_df = test_pooled_data(dfs, labels)
    t1 = time.time()

    dfs = [
        d.select(labels[0]) if i % 2 == 0 else d.select(labels[1])
        for i, d in enumerate(dfs)
    ]

    # try:
    # STEP 2: Test with Meta Analysis
    _dfs = [
        _df.select(
            [col.name for col in _df.select(pl.all().n_unique() > 1) if col.item()],
        )
        for _df in dfs
    ]
    meta_dfs, meta_df_labels = zip(*[mxm_ci_test(d) for d in _dfs])
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
    # except rpy2.rinterface_lib.embedded.RRuntimeError:
    #     fisher_df = pooled_result_df.with_columns(pvalue=pl.lit(None))
    t2 = time.time()
    # print(fisher_df)
    # STEP 3: Test with FedCI
    server = fedci.Server([fedci.Client(str(i), d) for i, d in enumerate(dfs, start=1)])

    fedci_results = server.run()
    t3 = time.time()
    fedci_all_labels = sorted(list(server.schema.keys()))

    fedci_df = server_results_to_dataframe(fedci_all_labels, fedci_results).sort(
        "ord", "X", "Y", "S"
    )

    return pooled_result_df, fisher_df, fedci_df, t1 - t0, t2 - t1, t3 - t2


data_dir = "experiments/fixed_effect_data/sim2"


# seed = random.randint(0, 100000)
seed_start = 10000
num_runs = 30
SEEDS = range(seed_start, seed_start + num_runs)
# SEEDS = [10011]
SAMPLES = [500, 1000, 2500, 5000]
CLIENTS = [4, 8, 12]

# stop waehrend 22tem seed.

# pag_ids_to_test = [61]
# pag_ids_to_test_no_slides_pag = [pag_id for pag_id in three_tail_pags if pag_id != 61]
# pag_ids_to_test_no_slides_pag_batched = []
# bs = 3
# for i in range(0, len(pag_ids_to_test_no_slides_pag), bs):
#     pag_ids_to_test_no_slides_pag_batched.append(
#         pag_ids_to_test_no_slides_pag[i : i + bs]
#     )
# pag_ids_to_test = pag_ids_to_test_no_slides_pag_batched[1]
pag_ids_to_test = three_tail_pags
pag_ids_to_test = [18, 61, 1]

# print(pag_ids_to_test)
# asd
POTENTIAL_VAR_LEVELS = [
    ("continuous", 1),
    ("binary", 2),
    ("nominal", 4),
    ("ordinal", 4),
]

from tqdm import tqdm

for pag_id in tqdm(pag_ids_to_test, position=0, leave=True):
    for seed in tqdm(SEEDS, position=1, leave=False):
        random.seed(seed)

        base_pag = test_pag_mapping[pag_id]
        label_split = test_label_split_mapping[pag_id]

        df_msep = df_msep_mapping[pag_id]

        fixed_effect_pag = np.zeros((base_pag.shape[0] + 1, base_pag.shape[1] + 1))
        fixed_effect_pag[:-1, :-1] = base_pag
        fixed_effect_pag[:-1, -1] = np.full(base_pag.shape[0], fill_value=3)
        fixed_effect_pag[-1, :-1] = np.full(base_pag.shape[0], fill_value=2)

        all_vars = sorted(list(set(label_split[0] + label_split[1])))

        var_types = {}
        var_levels = {}
        for var in all_vars:
            choice = random.choice(POTENTIAL_VAR_LEVELS)
            var_types[var] = choice[0]
            var_levels[var] = choice[1]
        var_types["CLIENT"] = "nominal"
        var_levels["CLIENT"] = -1
        while (
            sum([True if vt == "continuous" else False for vt in var_types.values()])
            < 2
        ):
            var = random.choice(all_vars)
            if var == "CLIENT":
                continue
            choice = random.choice(POTENTIAL_VAR_LEVELS)
            var_types[var] = choice[0]
            var_levels[var] = choice[1]

        for num_clients in tqdm(CLIENTS, position=2, leave=False):
            var_levels["CLIENT"] = num_clients
            var_levels_list = list(var_levels.values())

            for num_samples in tqdm(SAMPLES, position=3, leave=False):
                file_key = f"{seed}-id{pag_id}-s{num_samples}-c{num_clients}.parquet"
                target_file = f"{data_dir}/{file_key}"
                if os.path.exists(target_file):
                    continue

                df = get_data(
                    fixed_effect_pag, num_samples, var_types, var_levels_list, seed
                )

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

                df_result = df_fedci.join(
                    df_pooled, on=["ord", "X", "Y", "S"], how="left"
                )
                df_result = df_result.join(
                    df_fisher, on=["ord", "X", "Y", "S"], how="left"
                )

                df_result = df_result.join(
                    df_msep, on=["ord", "X", "Y", "S"], how="left"
                )

                # del var_types["CLIENT"]
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
                df_result.write_parquet(target_file)


"""
bad tests table:
    ┌─────┬─────┬─────┬─────────┬────────────┬────────────────────┬────────────────────┬────────────────────┬────────────────────┬───────────────┬──────────────┬────────────┐
    │ X   ┆ Y   ┆ S   ┆ X_type  ┆ Y_type     ┆ norm_X_res         ┆ norm_X_unres       ┆ norm_Y_res         ┆ norm_Y_unres       ┆ pooled_pvalue ┆ fedci_pvalue ┆ diff_log   │
    │ --- ┆ --- ┆ --- ┆ ---     ┆ ---        ┆ ---                ┆ ---                ┆ ---                ┆ ---                ┆ ---           ┆ ---          ┆ ---        │
    │ str ┆ str ┆ str ┆ str     ┆ str        ┆ list[f64]          ┆ list[f64]          ┆ list[f64]          ┆ list[f64]          ┆ f64           ┆ f64          ┆ f64        │
    ╞═════╪═════╪═════╪═════════╪════════════╪════════════════════╪════════════════════╪════════════════════╪════════════════════╪═══════════════╪══════════════╪════════════╡
    │ A   ┆ D   ┆ B,C ┆ nominal ┆ continuous ┆ [7244.220793,      ┆ [1983.496532,      ┆ [0.896941]         ┆ [0.926268]         ┆ 0.823524      ┆ 0.0          ┆ -34.344613 │
    │     ┆     ┆     ┆         ┆            ┆ 4317.826296, 204…  ┆ 4413.652583, 211…  ┆                    ┆                    ┆               ┆              ┆            │
    │ B   ┆ E   ┆ C   ┆ nominal ┆ binary     ┆ [1798.772668,      ┆ [1.484077,         ┆ [1.996099]         ┆ [2.084913]         ┆ 0.552412      ┆ 0.0          ┆ -33.945315 │
    │     ┆     ┆     ┆         ┆            ┆ 1385.368786, 160…  ┆ 2.846759,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 3.235494]          ┆                    ┆                    ┆               ┆              ┆            │
    │ C   ┆ E   ┆ A,B ┆ nominal ┆ nominal    ┆ [1713.176329,      ┆ [1.391501,         ┆ [1.363572,         ┆ [1.408117,         ┆ 0.484193      ┆ 0.0          ┆ -33.813506 │
    │     ┆     ┆     ┆         ┆            ┆ 2709.815838, 239…  ┆ 1.975904,          ┆ 2.471352,          ┆ 2.425159,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 2.937138]          ┆ 2.789982]          ┆ 2.858974]          ┆               ┆              ┆            │
    │ B   ┆ C   ┆ D,E ┆ nominal ┆ binary     ┆ [420.944634,       ┆ [1.980797,         ┆ [2.298209]         ┆ [2.414169]         ┆ 0.361778      ┆ 0.0          ┆ -33.522051 │
    │     ┆     ┆     ┆         ┆            ┆ 361.520271, 384.5… ┆ 2.083179,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 4.439207]          ┆                    ┆                    ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C   ┆ nominal ┆ continuous ┆ [1172.429684,      ┆ [2.097953,         ┆ [0.319701]         ┆ [0.733341]         ┆ 0.200311      ┆ 0.0          ┆ -32.930892 │
    │     ┆     ┆     ┆         ┆            ┆ 702.877977, 31.7…  ┆ 2.323634,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 1.656134]          ┆                    ┆                    ┆               ┆              ┆            │
    │ A   ┆ C   ┆ B,E ┆ binary  ┆ nominal    ┆ [0.576171]         ┆ [1.394011]         ┆ [214.013673,       ┆ [284.674724,       ┆ 0.119371      ┆ 4.0427e-92   ┆ -32.413259 │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 132.298833, 391.5… ┆ 287.208427, 139.4… ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C   ┆ binary  ┆ nominal    ┆ [1.494225]         ┆ [2.20666]          ┆ [1798.774232,      ┆ [1.91135,          ┆ 0.113433      ┆ 0.0          ┆ -32.362236 │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 1385.369393, 160…  ┆ 3.421028,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆                    ┆ 3.434708]          ┆               ┆              ┆            │
    │ B   ┆ C   ┆ A,E ┆ nominal ┆ nominal    ┆ [6851.672449,      ┆ [1623.589437,      ┆ [0.329901,         ┆ [0.537536,         ┆ 0.094734      ┆ 0.0          ┆ -32.182096 │
    │     ┆     ┆     ┆         ┆            ┆ 179.075564, 0.73…  ┆ 189.208736, 0.00…  ┆ 2.732416,          ┆ 3.032242,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 0.599711]          ┆ 1.426264]          ┆               ┆              ┆            │
    │ B   ┆ C   ┆ D,E ┆ ordinal ┆ nominal    ┆ [2.116581,         ┆ [2.923916,         ┆ [235.806265,       ┆ [893.691871,       ┆ 0.092157      ┆ 2.3001e-19   ┆ -32.154513 │
    │     ┆     ┆     ┆         ┆            ┆ 1.941542,          ┆ 2.355817,          ┆ 327.077225, 222.2… ┆ 445.722624, 225.1… ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 2.328682]          ┆ 2.719277]          ┆                    ┆                    ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C,D ┆ nominal ┆ continuous ┆ [671.544457,       ┆ [2.059166,         ┆ [0.421303]         ┆ [0.766617]         ┆ 0.07659       ┆ 0.0          ┆ -31.969491 │
    │     ┆     ┆     ┆         ┆            ┆ 1119.249615, 42.4… ┆ 2.294816,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 2.144249]          ┆                    ┆                    ┆               ┆              ┆            │
    │ B   ┆ E   ┆ C,D ┆ nominal ┆ nominal    ┆ [444.176381,       ┆ [1.980797,         ┆ [1.065587,         ┆ [1.7764, 1.927746, ┆ 0.027855      ┆ 0.0          ┆ -30.958033 │
    │     ┆     ┆     ┆         ┆            ┆ 355.804491, 377.7… ┆ 2.083179,          ┆ 1.191318,          ┆ 2.849133]          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 4.439207]          ┆ 1.718076]          ┆                    ┆               ┆              ┆            │
    │ B   ┆ D   ┆ C   ┆ binary  ┆ nominal    ┆ [2.245082]         ┆ [2.465336]         ┆ [1447.51252,       ┆ [2.339669,         ┆ 0.009618      ┆ 0.0          ┆ -29.894691 │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 377.125536, 663.5… ┆ 5.478128,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆                    ┆ 9.511977]          ┆               ┆              ┆            │
    │ A   ┆ B   ┆ E   ┆ binary  ┆ ordinal    ┆ [0.243654]         ┆ [0.878189]         ┆ [1.155412,         ┆ [1.147143,         ┆ 1.0           ┆ 2.0842e-12   ┆ -26.896614 │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 0.705135,          ┆ 0.84245, 3.305207] ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 3.307871]          ┆                    ┆               ┆              ┆            │
    │ B   ┆ E   ┆ C,D ┆ nominal ┆ nominal    ┆ [370.556477,       ┆ [3.119661,         ┆ [0.511177,         ┆ [1.12796,          ┆ 1.4832e-8     ┆ 0.0          ┆ -16.512265 │
    │     ┆     ┆     ┆         ┆            ┆ 1472.277439, 645.… ┆ 3.194002,          ┆ 1.105483,          ┆ 2.576671,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 2.320232]          ┆ 1.562903]          ┆ 3.196079]          ┆               ┆              ┆            │
    │ A   ┆ B   ┆ E   ┆ ordinal ┆ ordinal    ┆ [0.041479,         ┆ [1.402464,         ┆ [0.157363,         ┆ [3.186194,         ┆ 1.5530e-7     ┆ 4.3732e-14   ┆ -15.082778 │
    │     ┆     ┆     ┆         ┆            ┆ 0.76955, 3.185659] ┆ 0.905891,          ┆ 1.283773,          ┆ 3.518659,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 4.605686]          ┆ 0.069042]          ┆ 2.582438]          ┆               ┆              ┆            │
    │ B   ┆ E   ┆ C,D ┆ nominal ┆ nominal    ┆ [2533.887947,      ┆ [1.340102,         ┆ [0.269653,         ┆ [0.724598,         ┆ 3.1527e-11    ┆ 0.0          ┆ -10.358586 │
    │     ┆     ┆     ┆         ┆            ┆ 1782.822062, 145…  ┆ 2.565449,          ┆ 0.885307,          ┆ 1.869881,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 3.507102]          ┆ 1.560901]          ┆ 2.749586]          ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C,D ┆ ordinal ┆ ordinal    ┆ [2.591165,         ┆ [3.789152,         ┆ [2.648673,         ┆ [2.662377,         ┆ 2.0979e-9     ┆ 7.5487e-14   ┆ -10.232493 │
    │     ┆     ┆     ┆         ┆            ┆ 0.850576,          ┆ 0.761021,          ┆ 4.470852,          ┆ 4.857798,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 1.079674]          ┆ 1.767155]          ┆ 4.388853]          ┆ 4.589971]          ┆               ┆              ┆            │
    │ A   ┆ D   ┆ C,E ┆ binary  ┆ ordinal    ┆ [1.424201]         ┆ [1.652333]         ┆ [3.00414,          ┆ [3.05313, 3.91762, ┆ 0.001568      ┆ 6.0741e-8    ┆ -10.158517 │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 3.878827,          ┆ 3.271338]          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 3.266372]          ┆                    ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C,D ┆ ordinal ┆ continuous ┆ [1.652268,         ┆ [1.781757,         ┆ [2.625048]         ┆ [2.854097]         ┆ 0.000148      ┆ 7.0649e-8    ┆ -7.648713  │
    │     ┆     ┆     ┆         ┆            ┆ 1.011874,          ┆ 1.011793,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 5.171796]          ┆ 5.460007]          ┆                    ┆                    ┆               ┆              ┆            │
    │ C   ┆ E   ┆ A,D ┆ nominal ┆ ordinal    ┆ [2.381269,         ┆ [3.36431,          ┆ [2.28115,          ┆ [3.994954,         ┆ 2.5360e-10    ┆ 3.1768e-13   ┆ -6.682472  │
    │     ┆     ┆     ┆         ┆            ┆ 2.587995,          ┆ 3.701739,          ┆ 1.590668,          ┆ 3.211074,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 3.009647]          ┆ 4.051698]          ┆ 1.557205]          ┆ 2.124408]          ┆               ┆              ┆            │
    │ …   ┆ …   ┆ …   ┆ …       ┆ …          ┆ …                  ┆ …                  ┆ …                  ┆ …                  ┆ …             ┆ …            ┆ …          │
    │ B   ┆ D   ┆ E   ┆ nominal ┆ nominal    ┆ [1.394473,         ┆ [1.914013,         ┆ [0.348839,         ┆ [0.488642,         ┆ 3.0181e-13    ┆ 1.1433e-11   ┆ 3.634415   │
    │     ┆     ┆     ┆         ┆            ┆ 0.901947,          ┆ 1.398763,          ┆ 0.475194,          ┆ 0.695972,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 0.000626]          ┆ 0.000609]          ┆ 0.489721]          ┆ 1.796987]          ┆               ┆              ┆            │
    │ B   ┆ D   ┆ C,E ┆ nominal ┆ nominal    ┆ [2.646024,         ┆ [2.592877,         ┆ [1.375441,         ┆ [1.416976,         ┆ 3.4961e-13    ┆ 1.3292e-11   ┆ 3.638076   │
    │     ┆     ┆     ┆         ┆            ┆ 2.07425, 0.000826] ┆ 3.123002,          ┆ 2.457979,          ┆ 2.524352,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 0.000793]          ┆ 4.494416]          ┆ 4.948652]          ┆               ┆              ┆            │
    │ C   ┆ D   ┆ A,B ┆ nominal ┆ nominal    ┆ [1.330129,         ┆ [5.831642,         ┆ [3.56671,          ┆ [5.450405,         ┆ 8.8613e-16    ┆ 3.8464e-14   ┆ 3.649734   │
    │     ┆     ┆     ┆         ┆            ┆ 2.79183, 2.611494] ┆ 8.08896, 3.836758] ┆ 5.258522,          ┆ 9.367793,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 2.943593]          ┆ 5.105809]          ┆               ┆              ┆            │
    │ B   ┆ D   ┆ E   ┆ nominal ┆ nominal    ┆ [0.263102,         ┆ [3.89438,          ┆ [0.071236,         ┆ [2.504032,         ┆ 2.5322e-10    ┆ 1.0044e-8    ┆ 3.680432   │
    │     ┆     ┆     ┆         ┆            ┆ 0.151201,          ┆ 3.145949, 1.71468] ┆ 0.405309,          ┆ 4.803164,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 0.356998]          ┆                    ┆ 0.000244]          ┆ 0.000306]          ┆               ┆              ┆            │
    │ B   ┆ D   ┆ E   ┆ nominal ┆ nominal    ┆ [0.460102,         ┆ [2.367472,         ┆ [0.220199,         ┆ [3.757331,         ┆ 2.3658e-8     ┆ 9.5552e-7    ┆ 3.698577   │
    │     ┆     ┆     ┆         ┆            ┆ 0.346977,          ┆ 4.402851,          ┆ 0.188202,          ┆ 4.800445,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 0.178142]          ┆ 3.378122]          ┆ 0.000127]          ┆ 0.000177]          ┆               ┆              ┆            │
    │ A   ┆ C   ┆ D,E ┆ binary  ┆ ordinal    ┆ [0.321567]         ┆ [1.652333]         ┆ [2.861659,         ┆ [3.024498,         ┆ 2.6143e-9     ┆ 1.3460e-7    ┆ 3.941311   │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 3.895731,          ┆ 3.980815,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 2.071256]          ┆ 2.204424]          ┆               ┆              ┆            │
    │ B   ┆ D   ┆ C,E ┆ nominal ┆ nominal    ┆ [1.114039,         ┆ [4.47739,          ┆ [4.392475,         ┆ [5.694094,         ┆ 5.6526e-12    ┆ 3.6006e-10   ┆ 4.15414    │
    │     ┆     ┆     ┆         ┆            ┆ 1.01982, 1.984469] ┆ 4.466297,          ┆ 5.330249,          ┆ 7.954281,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆ 3.604044]          ┆ 0.000283]          ┆ 0.000342]          ┆               ┆              ┆            │
    │ C   ┆ D   ┆ B,E ┆ nominal ┆ nominal    ┆ [1.36344,          ┆ [2.521086,         ┆ [3.757331,         ┆ [5.210895,         ┆ 0.000003      ┆ 0.000186     ┆ 4.205128   │
    │     ┆     ┆     ┆         ┆            ┆ 0.655829,          ┆ 4.765529,          ┆ 4.800445,          ┆ 6.919893,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 1.339878]          ┆ 1.839221]          ┆ 0.000177]          ┆ 0.000203]          ┆               ┆              ┆            │
    │ A   ┆ C   ┆ B,E ┆ ordinal ┆ nominal    ┆ [1.954188,         ┆ [2.087436,         ┆ [0.92162,          ┆ [1.643393,         ┆ 0.000038      ┆ 0.002686     ┆ 4.26036    │
    │     ┆     ┆     ┆         ┆            ┆ 2.904035,          ┆ 3.314433, 3.11164] ┆ 1.221003,          ┆ 1.821496,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 3.047048]          ┆                    ┆ 1.176399]          ┆ 1.376834]          ┆               ┆              ┆            │
    │ B   ┆ C   ┆ A,D ┆ binary  ┆ ordinal    ┆ [1.358237]         ┆ [2.007725]         ┆ [2.712219,         ┆ [2.726895,         ┆ 1.0019e-17    ┆ 7.7157e-14   ┆ 4.345837   │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 1.709935, 4.13614] ┆ 1.845615,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆                    ┆ 4.097877]          ┆               ┆              ┆            │
    │ B   ┆ D   ┆ C,E ┆ nominal ┆ nominal    ┆ [0.58103,          ┆ [2.600343,         ┆ [3.016611,         ┆ [5.210895,         ┆ 4.4043e-9     ┆ 3.8518e-7    ┆ 4.471153   │
    │     ┆     ┆     ┆         ┆            ┆ 0.871181,          ┆ 4.738388,          ┆ 4.050229,          ┆ 6.919893,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 0.899511]          ┆ 3.880443]          ┆ 0.000161]          ┆ 0.000202]          ┆               ┆              ┆            │
    │ C   ┆ D   ┆ E   ┆ nominal ┆ nominal    ┆ [1.240857,         ┆ [4.080882,         ┆ [0.071236,         ┆ [4.392475,         ┆ 7.6760e-16    ┆ 9.1624e-14   ┆ 4.517695   │
    │     ┆     ┆     ┆         ┆            ┆ 0.394911,          ┆ 5.11181, 1.339475] ┆ 0.405309,          ┆ 5.330249,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 0.859935]          ┆                    ┆ 0.000244]          ┆ 0.000284]          ┆               ┆              ┆            │
    │ A   ┆ D   ┆ B,C ┆ ordinal ┆ binary     ┆ [0.722199,         ┆ [0.729245,         ┆ [3.202444]         ┆ [3.024942]         ┆ 0.000023      ┆ 0.003384     ┆ 4.972028   │
    │     ┆     ┆     ┆         ┆            ┆ 3.302545,          ┆ 3.180597,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 3.727833]          ┆ 3.707913]          ┆                    ┆                    ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C,D ┆ ordinal ┆ ordinal    ┆ [3.111291,         ┆ [4.420777,         ┆ [0.628322,         ┆ [2.54435,          ┆ 0.000105      ┆ 0.049928     ┆ 6.164002   │
    │     ┆     ┆     ┆         ┆            ┆ 2.611613,          ┆ 2.877463,          ┆ 0.753263,          ┆ 1.451116,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 3.638036]          ┆ 3.847059]          ┆ 0.640922]          ┆ 1.382098]          ┆               ┆              ┆            │
    │ A   ┆ B   ┆ C,E ┆ binary  ┆ nominal    ┆ [2.927145]         ┆ [152.287059]       ┆ [2.690269,         ┆ [1623.589485,      ┆ 0.002045      ┆ 1.0          ┆ 6.192327   │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 3.441842,          ┆ 189.208682, 0.00…  ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 0.000612]          ┆                    ┆               ┆              ┆            │
    │ A   ┆ B   ┆ E   ┆ binary  ┆ nominal    ┆ [0.822571]         ┆ [140.688027]       ┆ [2.435724, 2.2214, ┆ [6851.670053,      ┆ 0.000947      ┆ 1.0          ┆ 6.961695   │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 0.000493]          ┆ 179.075225, 0.73…  ┆               ┆              ┆            │
    │ B   ┆ C   ┆ A,E ┆ ordinal ┆ continuous ┆ [3.378231,         ┆ [3.620933,         ┆ [0.149784]         ┆ [0.997851]         ┆ 4.6636e-15    ┆ 1.5969e-11   ┆ 8.138622   │
    │     ┆     ┆     ┆         ┆            ┆ 3.416716,          ┆ 3.596831,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 3.190239]          ┆ 3.317851]          ┆                    ┆                    ┆               ┆              ┆            │
    │ B   ┆ C   ┆ E   ┆ ordinal ┆ continuous ┆ [0.240935,         ┆ [0.49671,          ┆ [0.134696]         ┆ [2.20164]          ┆ 7.7847e-21    ┆ 0.000007     ┆ 22.714802  │
    │     ┆     ┆     ┆         ┆            ┆ 1.571949,          ┆ 1.727077,          ┆                    ┆                    ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆ 0.272601]          ┆ 0.657429]          ┆                    ┆                    ┆               ┆              ┆            │
    │ A   ┆ C   ┆ E   ┆ ordinal ┆ continuous ┆ [0.147401,         ┆ [0.160534,         ┆ [0.134696]         ┆ [1.639405]         ┆ 1.1571e-23    ┆ 0.184666     ┆ 32.849571  │
    │     ┆     ┆     ┆         ┆            ┆ 1.733942]          ┆ 1.756871]          ┆                    ┆                    ┆               ┆              ┆            │
    │ A   ┆ B   ┆ E   ┆ ordinal ┆ ordinal    ┆ [0.147401,         ┆ [0.432617,         ┆ [0.240935,         ┆ [2.568657,         ┆ 7.9991e-22    ┆ 0.767254     ┆ 34.273839  │
    │     ┆     ┆     ┆         ┆            ┆ 1.733942]          ┆ 3.168274]          ┆ 1.571949,          ┆ 3.310036,          ┆               ┆              ┆            │
    │     ┆     ┆     ┆         ┆            ┆                    ┆                    ┆ 0.272601]          ┆ 1.655968]          ┆               ┆              ┆            │
    └─────┴─────┴─────┴─────────┴────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┴───────────────┴──────────────┴────────────┘

== Nulls in predictions
shape: (1, 3)
┌───────────────┬───────────────┬──────────────┐
│ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue │
│ ---           ┆ ---           ┆ ---          │
│ f64           ┆ f64           ┆ f64          │
╞═══════════════╪═══════════════╪══════════════╡
│ 0.0           ┆ 0.011895      ┆ 0.0          │
└───────────────┴───────────────┴──────────────┘
== Correlation to MSep
shape: (4, 4)
┌──────────┬───────────────┬───────────────┬──────────────┐
│ MSep     ┆ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue │
│ ---      ┆ ---           ┆ ---           ┆ ---          │
│ f64      ┆ f64           ┆ f64           ┆ f64          │
╞══════════╪═══════════════╪═══════════════╪══════════════╡
│ 1.0      ┆ 0.386167      ┆ 0.344249      ┆ 0.386217     │
│ 0.386167 ┆ 1.0           ┆ 0.779219      ┆ 0.996115     │
│ 0.344249 ┆ 0.779219      ┆ 1.0           ┆ 0.777373     │
│ 0.386217 ┆ 0.996115      ┆ 0.777373      ┆ 1.0          │
└──────────┴───────────────┴───────────────┴──────────────┘

== MSep Agreement
Dependent!
shape: (10, 4)
┌───────────────┬───────────────┬──────────────┬───────┐
│ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len   │
│ ---           ┆ ---           ┆ ---          ┆ ---   │
│ bool          ┆ bool          ┆ bool         ┆ u32   │
╞═══════════════╪═══════════════╪══════════════╪═══════╡
│ false         ┆ null          ┆ false        ┆ 225   │
│ false         ┆ false         ┆ false        ┆ 11249 │
│ false         ┆ false         ┆ true         ┆ 54    │
│ false         ┆ true          ┆ false        ┆ 820   │
│ false         ┆ true          ┆ true         ┆ 6     │
│ true          ┆ null          ┆ true         ┆ 321   │
│ true          ┆ false         ┆ false        ┆ 39    │
│ true          ┆ false         ┆ true         ┆ 3002  │
│ true          ┆ true          ┆ false        ┆ 20    │
│ true          ┆ true          ┆ true         ┆ 30167 │
└───────────────┴───────────────┴──────────────┴───────┘
Independent!
shape: (10, 4)
┌───────────────┬───────────────┬──────────────┬──────┐
│ pooled_pvalue ┆ fisher_pvalue ┆ fedci_pvalue ┆ len  │
│ ---           ┆ ---           ┆ ---          ┆ ---  │
│ bool          ┆ bool          ┆ bool         ┆ u32  │
╞═══════════════╪═══════════════╪══════════════╪══════╡
│ false         ┆ null          ┆ false        ┆ 1    │
│ false         ┆ null          ┆ true         ┆ 2    │
│ false         ┆ false         ┆ false        ┆ 63   │
│ false         ┆ false         ┆ true         ┆ 1    │
│ false         ┆ true          ┆ false        ┆ 108  │
│ false         ┆ true          ┆ true         ┆ 4    │
│ true          ┆ null          ┆ true         ┆ 39   │
│ true          ┆ false         ┆ true         ┆ 145  │
│ true          ┆ true          ┆ false        ┆ 7    │
│ true          ┆ true          ┆ true         ┆ 3161 │
└───────────────┴───────────────┴──────────────┴──────┘
"""
