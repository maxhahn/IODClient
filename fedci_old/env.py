import os

def get_env_debug(): return 0 if (v := os.getenv("DEBUG")) is None else int(v)
def get_env_ridge(): return 0.0001 if (v := os.getenv("RIDGE")) is None else float(v)
def get_env_line_search(): return True if (v := os.getenv("LINE_SEARCH")) is None else bool(int(v))
def get_env_lm_damping(): return True if (v := os.getenv("LM_DAMPING")) is None else bool(int(v))
def get_env_fit_intercept(): return True if (v := os.getenv("FIT_INTERCEPT")) is None else bool(int(v))
def get_env_client_heterogeniety(): return 2 if (v := os.getenv("CLIENT_HETEROGENIETY")) is None else int(v)
def get_env_additive_masking(): return True if (v := os.getenv("ADDITIVE_MASKING")) is None else bool(int(v))
