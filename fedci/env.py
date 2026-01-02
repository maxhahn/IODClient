import os

DEBUG = 0 if (v := os.getenv("DEBUG")) is None else int(v)
RIDGE = 0.0001 if (v := os.getenv("RIDGE")) is None else float(v)
LR = 1 if (v := os.getenv("LR")) is None else float(v)
LINE_SEARCH = True if (v := os.getenv("LINE_SEARCH")) is None else bool(int(v))
LM_DAMPING = True if (v := os.getenv("LM_DAMPING")) is None else bool(int(v))
FIT_INTERCEPT = True if (v := os.getenv("FIT_INTERCEPT")) is None else bool(int(v))
CLIENT_HETEROGENIETY = (
    True if (v := os.getenv("CLIENT_HETEROGENIETY")) is None else bool(int(v))
)
ADDITIVE_MASKING = (
    True if (v := os.getenv("ADDITIVE_MASKING")) is None else bool(int(v))
)
