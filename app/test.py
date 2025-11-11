import os
import sys

import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fedci

df = pl.read_parquet("IOD/client-data/uploaded_files/wicked-data-01.parquet")
# df = pl.concat([df, df])
print(df.head())


# clientb = fedci.Client(df.select("A", "B", "C"))
# clienta = fedci.Client(df.select("B", "C", "D"))
# server = fedci.Server({"1": clienta, "2": clientb})

df = df.select("B", "C", "D")
client = fedci.Client(df)
server = fedci.Server({"1": client})
results = server.run(1)

all_labels = df.columns

import pandas as pd

columns = ("ord", "X", "Y", "S", "pvalue")
rows = []
for test in sorted(results):
    s_labels_string = ",".join(
        sorted([str(all_labels.index(l) + 1) for l in test.conditioning_set])
    )
    rows.append(
        (
            len(test.conditioning_set),
            all_labels.index(test.v0) + 1,
            all_labels.index(test.v1) + 1,
            s_labels_string,
            test.p_val,
        )
    )

df_test = pd.DataFrame(data=rows, columns=columns)
df_test.index += 1
print(df_test)

from collections import OrderedDict


def iod_r_call(df, client_labels, alpha=0.05, procedure="original"):
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r["source"]("./scripts/iod.r")
        iod_on_ci_data_f = ro.globalenv["iod_on_ci_data"]

        labels = sorted(list(set().union(*(client_labels.values()))))

        lvs = []
        # r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        suff_stat = [
            ("citestResults", ro.conversion.get_conversion().py2rpy(df)),
            ("all_labels", ro.StrVector(labels)),
        ]
        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)
        users = client_labels.keys()
        label_list = [ro.StrVector(v) for v in client_labels.values()]

        result = iod_on_ci_data_f(label_list, suff_stat, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]
        gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
        gi_pag_labels = [list(x[1]) for x in result["Gi_PAG_Label_List"].items()]
        # gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

        user_pags = {u: r for u, r in zip(users, gi_pag_list)}
        user_labels = {u: l for u, l in zip(users, gi_pag_labels)}

    return g_pag_list, g_pag_labels, user_pags, user_labels


def iod_r_callx(dfs, client_labels, alpha=0.05, procedure="original"):
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r["source"]("./scripts/iod.r")
        aggregate_ci_results_f = ro.globalenv["aggregate_ci_results"]

        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]

        suff_stat = [
            (
                "citestResultsList",
                [ro.conversion.get_conversion().py2rpy(_df) for _df in dfs],
            ),
            (
                "labelList",
                [ro.StrVector(_labels) for _labels in client_labels.values()],
            ),
        ]
        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)

        users = client_labels.keys()
        label_list = [ro.StrVector(v) for v in client_labels.values()]

        result = aggregate_ci_results_f(suff_stat, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]
        gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
        gi_pag_labels = [list(x[1]) for x in result["Gi_PAG_Label_List"].items()]
        # gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

        user_pags = {u: r for u, r in zip(users, gi_pag_list)}
        user_labels = {u: l for u, l in zip(users, gi_pag_labels)}

    return g_pag_list, g_pag_labels, user_pags, user_labels


def iod_r_callxx(dfs, client_labels, alpha=0.05, procedure="original"):
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r["source"]("./scripts/iod.r")
        aggregate_ci_results_f = ro.globalenv["aggregate_ci_results"]

        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        users = client_labels.keys()
        label_list = [ro.StrVector(v) for v in client_labels.values()]

        result = aggregate_ci_results_f(label_list, r_dfs, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]
        gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
        gi_pag_labels = [list(x[1]) for x in result["Gi_PAG_Label_List"].items()]
        # gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

        user_pags = {u: r for u, r in zip(users, gi_pag_list)}
        user_labels = {u: l for u, l in zip(users, gi_pag_labels)}

    return g_pag_list, g_pag_labels, user_pags, user_labels


def iod_r_callxxx(dfs, client_labels, alpha=0.05, procedure="original"):
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r["source"]("./scripts/iod.r")
        aggregate_ci_results_f = ro.globalenv["aggregate_ci_results"]

        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        label_list = [ro.StrVector(v) for v in client_labels]

        result = aggregate_ci_results_f(label_list, r_dfs, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]

    return g_pag_list, g_pag_labels


# x = iod_r_call(df_test, {"1": ["A", "B", "C"], "2": ["B", "C", "D"]})
# x = iod_r_call(df_test, {"1": df_test.columns})

# print(x[0])
# print(x[2])

y = iod_r_callxx([df_test], {"1": df.columns})

print(y[0])
print(y[2])
