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


from itertools import chain, combinations


def is_m_separable(pag, labels, non_sharable_vars=None):
    # pag = test_setup[0]
    # labels = set(sorted(list(set(test_setup[1][0] + test_setup[1][1]))))

    result = []
    for x in labels:
        label_wo_x = labels - {x}
        for y in label_wo_x:
            if x > y:
                continue
            conditioning_set = chain.from_iterable(
                combinations(label_wo_x - {y}, r) for r in range(0, len(label_wo_x))
            )
            for s in conditioning_set:
                if non_sharable_vars is not None and non_sharable_vars.issubset(
                    {x} | {y} | set(s)
                ):
                    continue
                with (
                    ro.default_converter + pandas2ri.converter + numpy2ri.converter
                ).context():
                    is_msep = msep_f(pag, sorted(list(labels)), x, y, list(s))
                r = {
                    "ord": len(s),
                    "X": x,
                    "Y": y,
                    "S": sorted(list(s)),
                    "MSep": bool(is_msep[0]),
                }
                result.append(r)
                # print(x,y,s, bool(is_msep[0]))
    df = pl.from_dicts(result)
    return df


pag_msep_location = "pag_msep/pag-{}.parquet"
if not os.path.exists(pag_msep_location.rpartition("/")[0]):
    os.makedirs(pag_msep_location.rpartition("/")[0])

for i in three_tail_pags:
    if os.path.exists(pag_msep_location.format(i)):
        continue
    pag = test_pag_mapping[i]
    label_split = test_label_split_mapping[i]
    labels = set(label_split[0] + label_split[1])
    non_sharable_vars = set(label_split[0]).symmetric_difference(set(label_split[1]))
    msep = is_m_separable(pag, labels, non_sharable_vars)
    msep.write_parquet(pag_msep_location.format(i))

df_msep_mapping = {
    i: pl.read_parquet(pag_msep_location.format(i))
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
        "fedci_bad_fit",
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
                test.bad_fit,
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


### iod funcs


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
    )


def orcale_test_iod(df_msep, vars, non_shared_vars):
    pass


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
    base_pag = test_pag_mapping[pag_id]
    label_split = test_label_split_mapping[pag_id]

    df_msep = df_msep_mapping[pag_id]

    print(df_msep)
    asd

    fixed_effect_pag = np.zeros((base_pag.shape[0] + 1, base_pag.shape[1] + 1))
    fixed_effect_pag[:-1, :-1] = base_pag
    fixed_effect_pag[:-1, -1] = np.full(base_pag.shape[0], fill_value=3)
    fixed_effect_pag[-1, :-1] = np.full(base_pag.shape[0], fill_value=2)

    all_vars = sorted(list(set(label_split[0] + label_split[1])))

    for seed in tqdm(SEEDS, position=1, leave=False):
        random.seed(seed)

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
