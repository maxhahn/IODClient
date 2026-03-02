import polars as pl

df_base = pl.read_parquet("simulations/sim-data.parquet").drop("oracle_og_iod_len")
df_redone = pl.read_parquet("simulations/redone-sim.parquet")
df_redone2 = pl.read_parquet("simulations/redone2-sim.parquet")

print(len(df_base), len(df_redone), len(df_redone2))

df_base = df_base.join(
    df_redone, on=["seed", "pag_id", "num_samples", "partitions"], how="anti"
)
df_base = df_base.join(
    df_redone2, on=["seed", "pag_id", "num_samples", "partitions"], how="anti"
)
df_redone = df_redone.join(
    df_redone2, on=["seed", "pag_id", "num_samples", "partitions"], how="anti"
)

print(len(df_base), len(df_redone), len(df_redone2))

df_base = df_base.drop_nulls()
df_redone = df_redone.drop_nulls()
df_redone2 = df_redone2.drop_nulls()

print(len(df_base), len(df_redone), len(df_redone2))


df = pl.concat([df_base, df_redone, df_redone2], how="diagonal_relaxed")

print(len(df), 35 * 30 * 4 * 3 * 42)


# print(
#     df.group_by(["seed", "pag_id"])
#     .len()
#     .sort("len")
#     .filter(pl.col("len") < 504)
#     .select(pl.all().n_unique())
# )

df = df.filter(pl.col("seed") != 10025)

df_30seeds = (
    df.unique(subset=["pag_id", "partitions", "num_samples", "seed"])
    .sort(["pag_id", "partitions", "num_samples", "seed"])
    .group_by("pag_id", "partitions", "num_samples")
    .agg(pl.col("seed").head(30))
    .explode("seed")
)
df = df.join(df_30seeds, on=["seed", "partitions", "num_samples", "pag_id"], how="semi")

print(len(df), 30 * 30 * 4 * 3 * 42)
assert len(df) == 30 * 30 * 4 * 3 * 42
print(df.select(pl.col(["seed", "partitions", "num_samples", "pag_id"]).n_unique()))
df.write_parquet("simulations/final-data.parquet")
