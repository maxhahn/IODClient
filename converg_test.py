# (['A', 'B', 'C', 'D'], ['B', 'C', 'D', 'E'])
#  B indep C ┆ D,E
#
# test2
# ['A', 'C', 'D', 'E'], ['A', 'B', 'D', 'E'])
# 10002 ┆ 97     ┆ 1000        ┆ 4          ┆ A   ┆ D   ┆ B,E ┆ 0.590267      ┆ 0.220164      ┆ 0.287829                  ┆ 0.223485     │
import importlib
import os

mult_test = 2


import polars as pl
import polars.selectors as cs
import fedci

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))

pl.Config.set_tbl_rows(80)

if mult_test==1:
    df = pl.read_parquet('test_data.parquet')
    test_case = ('B', 'C', ['D', 'E'])
    client_vars0 = ['A', 'B', 'C', 'D', "CLIENT"]
    client_vars = ['B', 'C', 'D', 'E', 'CLIENT']
    client_type = 1
elif mult_test==2:
    df = pl.read_parquet('test_data2.parquet')
    test_case = ('A', 'D', ['B', 'E'])
    client_vars0 = ['A', 'C', 'D', 'E', "CLIENT"]
    client_vars = ['A', 'B', 'D', 'E', 'CLIENT']
    client_type = 1
elif mult_test==0:
    df = pl.read_parquet('simple_test.parquet')
    test_case = ('X', 'Y', ['Z'])

print(df.head(2))

import rpy2.rinterface_lib.callbacks as cb
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri

# cb.consolewrite_print = lambda x: None
# cb.consolewrite_warnerror = lambda x: None


ro.r["source"]("./ci_functions2.r")
run_ci_test_f = ro.globalenv["run_ci_test"]
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
    df_pvals = pl.from_pandas(df_pvals)
    mapping = {str(i): l for i, l in enumerate(labels, start=1)}
    df_pvals = df_pvals.with_columns(
        pl.col("X").cast(pl.Utf8).replace(mapping),
        pl.col("Y").cast(pl.Utf8).replace(mapping),
        pl.col("S")
        .str.split(",")
        .list.eval(pl.element().replace(mapping))
        .list.sort()
        .list.join(","),
    )
    df_pvals = df_pvals.filter((pl.col('X')!='CLIENT') & (pl.col('Y')!='CLIENT') & (pl.col('S').str.contains('CLIENT')))
    df_pvals = df_pvals.with_columns(pl.col('S').str.split(',').list.filter(pl.element()!='CLIENT').list.join(','), pl.col('ord')-1)
    return df_pvals



# if mult_test > 0 :
#     mapping = {c:str(i) for i,c in enumerate(sorted(df['CLIENT'].unique()),start=1)}

#     df = df.filter(pl.col('CLIENT').replace(mapping).cast(pl.Int64)%2 != client_type)
#     df = df.select(client_vars).sort("CLIENT")

dfs = df.sort('CLIENT').partition_by('CLIENT')

if mult_test > 0:
    dfs = [d.select(client_vars0) if i%2==0 else d.select(client_vars) for i,d in enumerate(dfs)]


_dfs = [d.select(client_vars) for i,d in enumerate(dfs) if i%2==client_type]
print(pl.concat(_dfs)['CLIENT'].unique().to_list())
r_test = mxm_ci_test(pl.concat(_dfs))
x,y,s = test_case
r_test = r_test.filter((pl.col('X')==x) & (pl.col('Y')==y))
for _s in s:
    r_test = r_test.filter(pl.col('S').str.contains(_s))

print(r_test)

os.environ['CLIENT_HETEROGENIETY'] = "0"
clients = [fedci.Client(0, pl.concat(_dfs))]
server = fedci.Server(clients)
x,y,s = test_case
#result = server.test(x,y,s)
result = server.test(x,y,s+['CLIENT'])
print("pool")
print(result)



os.environ['CLIENT_HETEROGENIETY'] = "1"
clients = [fedci.Client(i, d) for i, d in enumerate(dfs, start=1)]
server = fedci.Server(clients)
result = server.test(*test_case)
print(fedci.get_env_client_heterogeniety())
print(result)

os.environ['CLIENT_HETEROGENIETY'] = "2"
importlib.reload(fedci)
clients = [fedci.Client(i, d) for i, d in enumerate(dfs, start=1)]
server = fedci.Server(clients)
result = server.test(*test_case)
print(fedci.get_env_client_heterogeniety())
print(result)
