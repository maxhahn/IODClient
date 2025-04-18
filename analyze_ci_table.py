import polars as pl
import polars.selectors as cs

import os

dir = 'experiments/simulation/algo_comp_faith/'

all_faithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/faithful')]
all_unfaithful_ids = [f.rpartition('-')[0] for f in os.listdir('experiments/datasets/unfaithful')]

files = os.listdir(dir)

files_faithful = [dir+f for f in files if f.rpartition('-')[0][:-6] in all_faithful_ids]
files_unfaithful = [dir+f for f in files if f.rpartition('-')[0][:-6] in all_unfaithful_ids]

print(len(files_faithful), len(files_unfaithful))

df_faithful = pl.read_parquet(files_faithful).with_columns(faithful=True)
df_unfaithful = pl.read_parquet(files_unfaithful).with_columns(faithful=False)

df = pl.concat([df_faithful, df_unfaithful])

#df = pl.read_parquet(dir)

df = df.with_columns(
    correct_fisher=pl.col('indep')==pl.col('indep_fisher'),
    correct_fedci=pl.col('indep')==pl.col('indep_fedci')
)

print(df.select(cs.starts_with('correct_')).mean())
print(df.group_by('faithful').agg(cs.starts_with('correct_').mean()))
