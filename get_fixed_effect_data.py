import graphviz
import numpy as np
import pandas as pd
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri

arrow_type_lookup = {1: "odot", 2: "normal", 3: "none"}

# supress R log
import rpy2.rinterface_lib.callbacks as cb

cb.consolewrite_print = lambda x: None
cb.consolewrite_warnerror = lambda x: None

ro.r["source"]("./ci_functions2.r")
get_data_f = ro.globalenv["get_data"]
# 337009
ALPHA = 0.05
NUM_SAMPLES = [
    200,
    400,
    500,
    1_000,
    2_000,
    3_000,
    4_000,
    5_000,
    6_000,
    8_000,
    10_000,
    20_000,
    30_000,
]
SPLITS = [[1, 1]]  # , [2,1], [1,2], [3,1], [1,3], [1,1,1,1], [2,2,1,1]]
SEEDS = (x + 337132 for x in range(100_000))
COEF_THRESHOLD = 0.3  # 0.1

DF_MSEP = (
    pl.read_parquet("experiments/pag_msep/pag-slides.parquet")
    .with_columns(pl.col("S").list.join(","))
    .with_columns(
        ord=pl.when(pl.col("S").str.len_chars() == 0)
        .then(pl.lit(0))
        .otherwise(pl.col("S").str.count_matches(",") + 1)
    )
)

TRUE_PAG = np.array(
    [
        [0, 0, 2, 2, 0],
        [0, 0, 2, 0, 0],
        [2, 1, 0, 2, 2],
        [2, 0, 3, 0, 2],
        [0, 0, 3, 3, 0],
    ]
)

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
# print(data2graph(FIXED_EFFECT_PAG, ["A", "B", "C", "D", "E", "CLIENT"]))


def get_data(num_samples, seed):
    var_types = {
        "A": "continuous",
        "B": "continuous",
        "C": "nominal",
        "D": "continuous",
        "E": "continuous",
        "CLIENT": "nominal",
    }
    var_levels = [1, 1, 3, 1, 1, 2]

    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        dat = get_data_f(
            FIXED_EFFECT_PAG,
            num_samples,
            list(var_types.keys()),
            var_levels,
            "mixed",
            COEF_THRESHOLD,
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


data_dir = "experiments/fixed_effect_data"
seed = 42
df = get_data(10_000, seed)

df1 = df.filter(pl.col("CLIENT") == "A").select("A", "C", "D", "E")
df2 = df.filter(pl.col("CLIENT") == "B").select("A", "B", "C", "E")

df1.write_parquet(f"{data_dir}/{seed}-c1-1.parquet")
df2.write_parquet(f"{data_dir}/{seed}-c2-1.parquet")
