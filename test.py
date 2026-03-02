import datetime
from collections import OrderedDict
from dataclasses import asdict
from functools import partial
from typing import Optional, Set

import numpy as np
import pandas as pd
import polars as pl
import rpy2.rinterface as ri

ri.embedded._C_stack_limit = 10**8  # ~100MB

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri

import fedci

# ro.r["source"]("app/scripts/iod.r")
# # ro.r["source"]("ci_functions2.r")


# def iod_r_call_on_combined_data(df, client_labels, alpha=0.05, procedure="original"):
#     with (ro.default_converter + pandas2ri.converter).context():
#         iod_on_ci_data_f = ro.globalenv["iod_on_ci_data"]
#         labels = sorted(list(set().union(*(client_labels.values()))))

#         suff_stat = [
#             ("citestResults", ro.conversion.get_conversion().py2rpy(df)),
#             ("all_labels", ro.StrVector(labels)),
#         ]
#         suff_stat = OrderedDict(suff_stat)
#         suff_stat = ro.ListVector(suff_stat)
#         users = client_labels.keys()
#         label_list = [ro.StrVector(v) for v in client_labels.values()]

#         result = iod_on_ci_data_f(label_list, suff_stat, alpha, procedure)

#         g_pag_list = [x.value for x in result.getbyname("G_PAG_List").items()]
#         g_pag_labels = [
#             list([str(a) for a in x.value])
#             for x in result.getbyname("G_PAG_Label_List").items()
#         ]
#         g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]

#         gi_pag_list = [x.value for x in result.getbyname("Gi_PAG_list").items()]
#         gi_pag_labels = [
#             list([str(a) for a in x.value])
#             for x in result.getbyname("Gi_PAG_Label_List").items()
#         ]
#         gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

#         user_pags = {u: r for u, r in zip(users, gi_pag_list)}
#         user_labels = {u: l for u, l in zip(users, gi_pag_labels)}

#         print(g_pag_list)
#         print(g_pag_labels)
#         print(gi_pag_list)
#         print(gi_pag_labels)

#     return g_pag_list, g_pag_labels, user_pags, user_labels


# df1 = pl.read_parquet("data-1.parquet")
# df2 = pl.read_parquet("data-2.parquet")

# c1 = fedci.Client("1", df1)
# c2 = fedci.Client("2", df2)
# server = fedci.Server([c1, c2])

# # likelihood_ratio_tests = server.run(max_cond_size=5)


# # all_labels = sorted(list(server.schema.keys()))

# # columns = ("ord", "X", "Y", "S", "pvalue")
# # rows = []
# # for test in sorted(likelihood_ratio_tests):
# #     s_labels_string = ",".join(
# #         sorted([str(all_labels.index(l) + 1) for l in test.conditioning_set])
# #     )
# #     rows.append(
# #         (
# #             len(test.conditioning_set),
# #             all_labels.index(test.v0) + 1,
# #             all_labels.index(test.v1) + 1,
# #             s_labels_string,
# #             test.p_value,
# #         )
# #     )

# # df = pd.DataFrame(data=rows, columns=columns)


# # _df = pl.from_pandas(df).write_parquet("testres.parquet")
# df = pl.read_parquet("testres.parquet").to_pandas()

# # let index start with 1
# df.index += 1

# alpha = 0.05

# participant_data_labels = {}
# participants = [c.id for c in server.clients.values()]
# participant_data_labels = {
#     str(i + 1): c.schema.keys() for i, c in enumerate(list(server.clients.values()))
# }


# print(df)
# print(participant_data_labels)

# result, result_labels, user_results, user_labels = iod_r_call_on_combined_data(
#     df,
#     participant_data_labels,
#     alpha=alpha,
# )


data = [
    [
        [0, 2, 2, 0, 0],
        [2, 0, 2, 2, 1],
        [2, 3, 0, 2, 0],
        [0, 3, 3, 0, 0],
        [0, 2, 0, 0, 0],
    ]
]
labels = [["A", "C", "D", "E", "B"]]
import graphviz

arrow_type_lookup = {1: "odot", 2: "normal", 3: "none"}


def data2graph(data, labels):
    graph = graphviz.Digraph(format="png")
    for label in labels:
        graph.node(label)
    import os

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            arrhead = data[i][j]
            arrtail = data[j][i]
            if data[i][j] == 1:
                graph.edge(
                    labels[i],
                    labels[j],
                    arrowtail=arrow_type_lookup[arrtail],
                    arrowhead=arrow_type_lookup[arrhead],
                    dir="both",
                )
            elif data[i][j] == 2:
                graph.edge(
                    labels[i],
                    labels[j],
                    arrowtail=arrow_type_lookup[arrtail],
                    arrowhead=arrow_type_lookup[arrhead],
                    dir="both",
                )
            elif data[i][j] == 3:
                graph.edge(
                    labels[i],
                    labels[j],
                    arrowtail=arrow_type_lookup[arrtail],
                    arrowhead=arrow_type_lookup[arrhead],
                    dir="both",
                )

    return graph


for d, l in zip(data, labels):
    g = data2graph(d, l)
    print(g)
