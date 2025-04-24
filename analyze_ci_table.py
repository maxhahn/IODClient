import polars as pl
import polars.selectors as cs

import os

dir = 'experiments/simulation/comp2/'

all_faithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/f2')]
all_unfaithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/uf2')]

files = os.listdir(dir)

files_by_type = {}
for f in files:
    faithfulness_type = f.split('-')[-3]
    if faithfulness_type not in files_by_type:
        files_by_type[faithfulness_type] = []
    files_by_type[faithfulness_type].append(dir+f)
#files_faithful = [dir+f for f in files if '-g-' in f]
#files_unfaithful = [dir+f for f in files if '-g-' not in f]

#print(len(files_faithful), len(files_unfaithful))

dfs = []
for t,f in files_by_type.items():
    df = pl.read_parquet(f).with_columns(faithfulness=pl.lit(t))
    dfs.append(df)

#dfs = []
#if len(files_faithful) > 0:
#    df_faithful = pl.read_parquet(files_faithful).with_columns(faithful=True)
#    dfs.append(df_faithful)
# if len(files_unfaithful) > 0:
#     df_unfaithful = pl.read_parquet(files_unfaithful).with_columns(faithful=False)
#     dfs.append(df_unfaithful)

df = pl.concat(dfs)

#df = pl.read_parquet(dir)

df = df.with_columns(
    correct_fisher=pl.col('MSep')==pl.col('indep_fisher'),
    correct_fedci=pl.col('MSep')==pl.col('indep_fedci')
)

print(df.select(cs.starts_with('correct_')).mean())
print(df.group_by('faithfulness').agg(cs.starts_with('correct_').mean()).with_columns(diff=pl.col('correct_fedci')-pl.col('correct_fisher')))
