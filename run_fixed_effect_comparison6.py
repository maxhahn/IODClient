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

    # l1_missing_ref_cat = any(
    #     l1_df.group_by("CLIENT")
    #     .agg((cs.string() == "A").any())
    #     .sort("CLIENT")
    #     .select(cs.boolean())
    #     .to_numpy()
    # )
    # l2_missing_ref_cat = any(
    #     l2_df.group_by("CLIENT")
    #     .agg((cs.string() == "A").any())
    #     .sort("CLIENT")
    #     .select(cs.boolean())
    #     .to_numpy()
    # )

    l2_result_df = remove_client_from_pooled_result(*mxm_ci_test(l2_df))

    cb.consolewrite_print = lambda x: None
    cb.consolewrite_warnerror = lambda x: None

    intersection_result_df = remove_client_from_pooled_result(
        *mxm_ci_test(intersection_df)
    )
    # if len(intersection_df.columns) != 4:
    #     print("Linter lacks col")
    l1_result_df = remove_client_from_pooled_result(*mxm_ci_test(l1_df))

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

    # xdfs = df.sort("CLIENT", pl.exclude("CLIENT")).partition_by("CLIENT")
    # xdfs = [d.select(labels[1] + ["CLIENT"]) for i, d in enumerate(xdfs) if i % 2 == 1]
    # server = fedci.Server(
    #     [fedci.Client(str(i), d) for i, d in enumerate(xdfs, start=1)]
    # )

    # print(fisher_df)
    # STEP 3: Test with FedCI
    server = fedci.Server([fedci.Client(str(i), d) for i, d in enumerate(dfs, start=1)])

    # x = server.test("A", "C", ["B", "D"])
    # x = server.test("A", "C", ["E"])
    # x = server.test("C", "E", [])
    # x = server.test("A", "E", ["B", "C", "CLIENT"])
    # x = server.test("A", "E", ["C"])
    # print(x)
    # asd

    fedci_results = server.run()
    t3 = time.time()
    fedci_all_labels = sorted(list(server.schema.keys()))

    fedci_df = server_results_to_dataframe(fedci_all_labels, fedci_results).sort(
        "ord", "X", "Y", "S"
    )

    return pooled_result_df, fisher_df, fedci_df, t1 - t0, t2 - t1, t3 - t2


data_dir = "experiments/fixed_effect_data/sim"


# seed = random.randint(0, 100000)
seed_start = 10000
num_runs = 30
SEEDS = range(seed_start, seed_start + num_runs)
SEEDS = [10001]
SAMPLES = [500]
CLIENTS = [8]

pag_ids_to_test = [18]
#  A   ┆ C   ┆ E   ┆ 18     ┆ 10000 ┆ 8          ┆ 500


# print(pag_ids_to_test)
# asd
POTENTIAL_VAR_LEVELS = [
    ("continuous", 1),
    ("binary", 2),
    ("nominal", 4),
    ("ordinal", 4),
]
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_rows(80)
from tqdm import tqdm

for pag_id in tqdm(pag_ids_to_test, position=0, leave=True):
    for seed in tqdm(SEEDS, position=1, leave=False):
        random.seed(seed)

        base_pag = test_pag_mapping[pag_id]
        label_split = test_label_split_mapping[pag_id]
        print(label_split)

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
                # file_key = f"{seed}-id{pag_id}-s{num_samples}-c{num_clients}.parquet"
                # target_file = f"{data_dir}/{file_key}"
                # if os.path.exists(target_file):
                #     continue

                df = get_data(
                    fixed_effect_pag, num_samples, var_types, var_levels_list, seed
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

                df_result = df_fedci.join(
                    df_pooled, on=["ord", "X", "Y", "S"], how="left"
                )
                df_result = df_result.join(
                    df_fisher, on=["ord", "X", "Y", "S"], how="left"
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
Testing if X= 1 is indep of Y= 4 given S={ 2,3,5 }
Running ordinal regression for  4
[1] "DOFs 9; T 13.1320471215107; pval 0.156721738701227"
$message
NULL

$be
                     Y1          Y2          Y3
(Intercept) -1.82325542 -1.00713444 -0.29457017
B            0.42745797  0.40077724  0.43147958
C           -0.51752131 -0.56988198 -0.46443973
CLIENTD      0.26868128  0.26568472  0.57208661
CLIENTF      0.01957009  0.22799575  0.54803317
CLIENTH     -0.04516974  0.03609004 -0.03229317

$devi
[1] 606.1323

$message
[1] "problematic region"

$be
                      Y1          Y2           Y3
(Intercept)  14.39200422  14.6064102  14.38424983
B             0.67065969   0.5237809   0.56805249
C            -0.63172937  -0.6311986  -0.50819549
CLIENTD       0.36816309   0.3407560   0.61221563
CLIENTF       0.28743679   0.3783428   0.69301630
CLIENTH      -0.02259935   0.0523801  -0.05860584
AB          -16.42537671 -15.5887023 -13.92284988
AC          -16.77638679 -15.9000058 -14.94477864
AD          -15.70943546 -15.3969806 -14.77796058

$devi
[1] 593.0003

Running multinomial regression for  1
Call:
nnet::multinom(formula = ydat ~ ., data = ds0, trace = FALSE)

Coefficients:
  (Intercept)          B          C  CLIENTD  CLIENTF  CLIENTH
B    3.764781 -2.4125371 -0.7830898 8.556505 6.935251 8.860562
C    4.631544 -0.2427312 -1.2475728 8.623797 8.559865 7.949359
D    4.459566 -2.5626729 -0.5671576 8.338900 7.006989 8.640469

Residual Deviance: 324.1434
AIC: 360.1434
Call:
nnet::multinom(formula = ydat ~ ., data = ds1, trace = FALSE)

Coefficients:
  (Intercept)        B         C  CLIENTD  CLIENTF  CLIENTH      E.L       E.Q
B    578.2766 90.33196 -193.3160 364.6822 314.6807 466.0497 493.9996 -263.5795
C    578.8170 92.65458 -193.9120 365.0285 316.7151 465.2229 494.5150 -262.8914
D    578.5735 90.24801 -193.1502 364.7972 315.3003 466.0260 493.7284 -262.1260
       E.C
B 154.8655
C 155.3483
D 155.3182

Residual Deviance: 303.0852
AIC: 357.0852
p1: 0.1567217 and p2: 0.01239417

"""

"""
*** Combining p values for symmetry of tests between A and E given {'B', 'C'}
p value A: 2.463955422229743e-06
p value E: 0.15562032008618165
p value = 0.0000
*** Final betas
A ~ B,C,1 after 2 iterations
[-1.9530366274629047, -0.8892184313447579, -0.24933323106083555]
[-0.3236959045428933, -1.176390984967738, -0.9129326262439513]
[-2.055023172437948, -0.7029239730195322, -0.3272504597970568]
A ~ B,C,E,1 after 7 iterations
[15.57148601345034, -36.85794679706305, -70.98864283306466, -34.58835930022906, -132.9615037329495, 79.7453506521997]
[17.88843167473169, -37.45149294121802, -71.8937280997925, -35.51924052205936, -134.30967132002115, 79.46650612412486]
[15.486681931786734, -36.69165440800862, -70.82655368029373, -35.59625642219544, -134.69761011749506, 80.0323470641888]
E ~ B,C,1 after 4 iterations
[0.42707325152022085, -0.517371475997744, -0.29311366763609426]
[0.40004919561602603, -0.5694688522239814, -0.2496235210821584]
[0.4302684751715038, -0.4635085423351382, -0.21224084185237208]
E ~ A,B,C,1 after 4 iterations
[-8.226395823198583, -8.574910247282777, -7.510208734437245, 0.6687044779885277, -0.6309269331256153, 7.817971327430821]
[-6.227188133915537, -6.53519233431494, -6.035308374053146, 0.5210322257141814, -0.6297707800641453, 6.051120293671874]
[-4.4394322853483175, -5.455965760775356, -5.293555305352442, 0.5636747558507881, -0.5056436288604196, 5.011902091736474]

"""


"""
A ~ B,C,CLIENT,E,1 after 16 iterations
[2.7830023431884356, -9.943077391660708, 5.611796044457006, 8.83756638164465, 13.071594527539556, -18.850020472575128, 0.6943110692355211, 1.034069258038215, 23.13555037084006]
[5.114771708837004, -10.542948789641384, 5.965033296169628, 10.884728909358884, 12.245951687932335, -19.768892937500713, -0.24244859318622053, -0.31832405367194433, 24.46882611120256]
"""
