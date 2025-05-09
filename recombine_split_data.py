import polars as pl
import glob
import os

dir = 'experiments/datasets/data6'
target_dir = 'experiments/datasets/data7'
files = os.listdir(dir)
files = [f.rpartition('-')[0] for f in files]
files = list(set(files))

for f in files:
    f1_0 = f'{dir}/{f}-d1_0.parquet'
    f1_1 = f'{dir}/{f}-d1_1.parquet'
    #f1_2 = f'{dir}/{f}-d1_2.parquet'

    f2_0 = f'{dir}/{f}-d2_0.parquet'
    f2_1 = f'{dir}/{f}-d2_1.parquet'
    #f2_2 = f'{dir}/{f}-d2_2.parquet'

    df1_0 = pl.read_parquet(f1_0)
    df1_1 = pl.read_parquet(f1_1)
    #df1_2 = pl.read_parquet(f1_2)

    df2_0 = pl.read_parquet(f2_0)
    df2_1 = pl.read_parquet(f2_1)
    #df2_2 = pl.read_parquet(f2_2)

    df1 = pl.concat([df1_0, df1_1])#, df1_2])
    df2 = pl.concat([df2_0, df2_1])#, df2_2])

    df1.write_parquet(f'{target_dir}/{f}-p1.parquet')
    df2.write_parquet(f'{target_dir}/{f}-p2.parquet')
