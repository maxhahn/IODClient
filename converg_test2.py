import polars as pl
import polars.selectors as cs
import numpy as np



df = pl.read_parquet("test_data.parquet")

print(df)

df = df.to_dummies("CLIENT", drop_first=True)
df = df.filter(pl.col("CLIENT_C")==1)

X = df.select("B", "D", "E").to_numpy()
W = np.eye(len(df))
G = df.select(cs.contains("CLIENT_")).to_numpy()

print(X.shape, W.shape, G.shape)

print(X.T @ W @ G)
print(G.T @ W @ G)


# H_aa diagonal -> each client 1 element
# H_ba full -> each client 1 row/col
# H_bb full
# https://pubmed.ncbi.nlm.nih.gov/18672430/
