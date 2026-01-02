from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import rpyc
import statsmodels.api as sm
import statsmodels.genmod.families.family as fam
from scipy.special import softmax

from .env import ADDITIVE_MASKING, CLIENT_HETEROGENIETY, FIT_INTERCEPT, LR
from .utils import (
    BetaUpdateData,
    VariableType,
    categorical_separator,
    constant_colname,
    ordinal_separator,
    polars_dtype_map,
)


class DistributionalFamily:
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def inverse_deriv(eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def loglik(y: np.ndarray, mu: np.ndarray) -> float:
        raise NotImplementedError()


class Gaussian(DistributionalFamily):
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return mu

    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return eta

    @staticmethod
    def inverse_deriv(eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    @staticmethod
    def loglik(y: np.ndarray, mu: np.ndarray) -> float:
        resid = y - mu
        n = y.shape[0]
        sigma2 = np.mean(resid**2)
        sigma2 = np.clip(sigma2, 1e-10, None)
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * n
        return ll


class Binomial(DistributionalFamily):
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return np.log(mu / (1 - mu))

    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(eta, -350, 350)))

    @staticmethod
    def inverse_deriv(eta: np.ndarray) -> np.ndarray:
        mu = Binomial.inverse_link(eta)
        return mu * (1 - mu)

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)

    @staticmethod
    def loglik(y: np.ndarray, mu: np.ndarray) -> float:
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        return np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))


class ComputationHelper:
    @staticmethod
    def run_const_model(
        y: np.ndarray,
        family: DistributionalFamily,
    ):
        eta = np.zeros_like(y)
        if CLIENT_HETEROGENIETY:
            gamma = ComputationHelper.fit_local_gamma(y=y, offset=eta, family=family)
            eta += gamma
        mu: np.ndarray = family.inverse_link(eta)

        llf: float = family.loglik(y, mu)

        xwx = np.empty((0, 0))
        xwz = np.empty((0, 0))

        # only use non-dummy values in rss and n for gaussian regression
        result = {
            "llf": llf,
            "xwx": xwx,
            "xwz": xwz,
            "rss": 0,
            "n": 0,
        }
        if family == Gaussian:
            result["rss"] = np.sum((y - mu) ** 2).item()
            result["n"] = y.shape[0]
        return result

    @staticmethod
    def run_const_predict(
        y: np.ndarray,
        family: DistributionalFamily,
    ):
        eta = np.zeros_like(y)
        if CLIENT_HETEROGENIETY:
            gamma = ComputationHelper.fit_local_gamma(y=y, offset=eta, family=family)
            eta += gamma
        else:
            gamma = np.zeros_like(eta)
        mu: np.ndarray = family.inverse_link(eta)

        return eta, mu, gamma

    @staticmethod
    def run_prediction(
        y: np.ndarray,
        X: np.ndarray,
        beta: np.ndarray,
        family: DistributionalFamily,
    ):
        eta: np.ndarray = X @ beta

        if CLIENT_HETEROGENIETY:
            gamma = ComputationHelper.fit_local_gamma(y=y, offset=eta, family=family)
        else:
            gamma = np.zeros_like(eta)
        eta += gamma
        mu: np.ndarray = family.inverse_link(eta)
        return eta, mu, gamma

    @staticmethod
    def get_irls_step(
        y: np.ndarray,
        X: np.ndarray,
        eta: np.ndarray,
        mu: np.ndarray,
        gamma,
        family: DistributionalFamily,
    ):
        dmu_deta: np.ndarray = family.inverse_deriv(eta)
        var_y: np.ndarray = family.variance(mu)
        dmu_deta = np.clip(dmu_deta, 1e-10, None)
        var_y = np.clip(var_y, 1e-10, None)

        W = np.diag(((dmu_deta**2) / var_y).reshape(-1))
        z: np.ndarray = (eta - gamma) + LR * (y - mu) / dmu_deta
        # z: np.ndarray = eta + (y - mu) / dmu_deta

        xw = X.T @ W
        xwx = xw @ X
        # xwx *= LR
        xwz = xw @ z
        return xwx, xwz

    @staticmethod
    def run_regression(
        y: np.ndarray,
        X: np.ndarray,
        beta: np.ndarray,
        family: DistributionalFamily,
    ):
        eta, mu, gamma = ComputationHelper.run_prediction(y, X, beta, family)
        xwx, xwz = ComputationHelper.get_irls_step(y, X, eta, mu, gamma, family)
        llf: float = family.loglik(y, mu)
        # only use non-dummy values in rss and n for gaussian regression
        result = {"llf": llf, "xwx": xwx, "xwz": xwz, "rss": 0, "n": 0}
        if family == Gaussian:
            result["rss"] = np.sum((y - mu) ** 2).item()
            result["n"] = y.shape[0]
        return result

    @staticmethod
    def fit_local_gamma(y, offset, family, max_iter=100, tol=1e-8):
        """
        Parameters
        ----------
        y : array (n,)
            Response variable.
        offset : array (n,)
            Fixed offset added to the linear predictor.
        family : statsmodels.genmod.families.Family
            GLM family (Gaussian, Poisson, Binomial, etc.)
        max_iter : int
        tol : float

        Returns
        -------
        gamma : float
            Estimated intercept.
        """

        n = len(y)

        # Design matrix: intercept only
        X = np.ones((n, 1))

        smfamily = fam.Gaussian() if family == Gaussian else fam.Binomial()
        model = sm.GLM(y.reshape(-1), X, family=smfamily, offset=offset.reshape(-1))

        res = model.fit(maxiter=max_iter, tol=tol, disp=False)

        # Single intercept parameter
        gamma = float(res.params[0])

        return gamma

    @staticmethod
    def fit_local_gamma_ordinal(
        y, offset, family, current_gamma, lr=0.3, momentum=0.1, prev_step=0.0
    ):
        # 1. Calculate Initial State
        eta = current_gamma + offset
        mu = family.inverse_link(eta)

        # Safety Check: Ordinal models fail if mu hits 0 or 1
        mu = np.clip(mu, 1e-6, 1 - 1e-6)

        dmu_deta = family.inverse_deriv(eta)
        var = family.variance(mu)
        var = np.clip(var, 1e-8, None)

        # 2. Compute Gradient (Score)
        grad = np.sum((y - mu) * dmu_deta / var)

        # 3. Compute Fisher Info (Hessian)
        # Using a relative ridge: 1% of the average information
        w = (dmu_deta**2) / var
        fisher_info = np.sum(w)
        ridge = max(1e-4, 0.01 * fisher_info)
        fisher_info += ridge

        # 4. Compute Raw Step
        raw_step = grad / fisher_info

        # --- ROBUSTNESS LAYER: Clipping ---
        # Never allow gamma to jump more than 1.0 units in one go
        # Large jumps in ordinal models often move mu to 0 or 1, killing the gradient
        max_change = 1.0
        if abs(raw_step) > max_change:
            raw_step = np.sign(raw_step) * max_change

        # 5. Apply Damping and Momentum
        total_update = (lr * raw_step) + (momentum * prev_step)

        # Final check: If the update creates NaNs, kill the update
        if np.isnan(total_update):
            return current_gamma, 0.0

        updated_gamma = current_gamma + total_update

        return updated_gamma, total_update

    @staticmethod
    def fit_local_gammax(y, offset, family, max_iter=5, tol=1e-8):
        gamma = 0.0
        for _ in range(max_iter):
            eta = gamma + offset
            mu = family.inverse_link(eta)
            dmu_deta = family.inverse_deriv(eta)
            var = family.variance(mu)
            var = np.clip(var, 1e-10, None)

            # Score (first derivative)
            grad = np.sum((y - mu) * dmu_deta / var)

            # Fisher information (negative expected Hessian)
            w = (dmu_deta**2) / var
            fisher_info = np.sum(w)
            fisher_info = np.clip(fisher_info, 1e-10, None)

            step = grad / fisher_info

            if np.linalg.norm(step) < tol:
                break
        return gamma

    @staticmethod
    def fit_multinomial_gamma(y, offset, max_iter=100, tol=1e-8):
        n, K_minus_1 = y.shape
        K = K_minus_1 + 1

        # Full one-hot
        y_full = np.column_stack([1 - y.sum(axis=1), y])
        y_cat = np.argmax(y_full, axis=1)

        # Intercept-only design
        X = np.ones((n, 1))

        offset_full = np.column_stack([np.zeros(n), offset])

        # offset_full = softmax(offset_full, axis=1)

        model = sm.MNLogit(y_cat, X, offset=offset_full)

        res = model.fit(method="newton", maxiter=max_iter, tol=tol, disp=False)

        # ---- CORRECT ALIGNMENT ----
        gamma_est = res.params[0, :]  # length = len(estimated_cats)
        # gamma_full = np.zeros(K_minus_1)
        gamma_full = np.full(K_minus_1, fill_value=-float("inf"))

        ynames_map = res.model._ynames_map
        del ynames_map[0]

        for k, v in ynames_map.items():
            gamma_full[int(v) - 1] = gamma_est[k - 1]
        return gamma_full

    @staticmethod
    def fit_local_alphax(y_obs, offset, max_iter=10, tol=1e-6):
        """
        Fits local fixed effects (alpha) given a fixed global offset (X * beta).

        Parameters:
        - y_indices: 1D array of class labels (0 to K-1) for N observations.
        - offset: N x (K-1) array representing the global component X * beta.
        - num_categories: Total number of categories (K).
        - max_iter: Maximum IRLS iterations.

        Returns:
        - alpha: Array of shape (K-1,) representing the local fixed effects.
        """
        N = y_obs.shape[0]
        K_minus_1 = offset.shape[1]

        # Initialize alpha at zeros
        alpha = np.zeros(K_minus_1)

        # # Convert y to one-hot encoding (excluding the reference category 0)
        # # y_obs shape: (N, K-1)
        # y_obs = np.zeros((N, K_minus_1))
        # for i, val in enumerate(y_indices):
        #     if val > 0:
        #         y_obs[i, val-1] = 1

        for iteration in range(max_iter):
            # 1. Compute Pi (N x K)
            # eta includes the current alpha + the global offset
            eta = offset + alpha  # Broadcasting alpha across N rows

            # Add a column of zeros for the reference category (category 0)
            eta_full = np.hstack([np.zeros((N, 1)), eta])

            # Softmax calculation
            exp_eta = np.exp(eta_full - np.max(eta_full, axis=1, keepdims=True))
            pi_full = exp_eta / np.clip(
                np.sum(exp_eta, axis=1, keepdims=True), 1e-8, None
            )

            pi_full = np.clip(pi_full, 1e-6, 1 - 1e-6)
            pi_full /= np.sum(pi_full, axis=1, keepdims=True)

            # Pi for non-reference categories (N x K-1)
            pi = pi_full[:, 1:]

            # 2. Compute Score (Gradient) for alpha: shape (K-1,)
            # Gradient is the sum of (y_obs - pi) over all observations
            score = np.sum(y_obs - pi, axis=0)

            # 3. Compute Fisher Information (Hessian)
            # For an intercept-only model, the Hessian is the sum of the
            # multinomial variance-covariance matrices across all N.
            # Hessian shape: (K-1, K-1)

            # Diagonal part: sum of pi_k * (1 - pi_k)
            diag_part = np.diag(np.sum(pi * (1 - pi), axis=0))

            # Off-diagonal part: - sum of pi_k * pi_j
            off_diag_part = np.zeros((K_minus_1, K_minus_1))
            for k in range(K_minus_1):
                for j in range(k + 1, K_minus_1):
                    val = -np.sum(pi[:, k] * pi[:, j])
                    off_diag_part[k, j] = val
                    off_diag_part[j, k] = val

            hessian = diag_part + off_diag_part
            hessian += 1e-3 * np.eye(K_minus_1)

            # 4. Update alpha
            try:
                # step = np.linalg.solve(hessian, score)
                step = np.linalg.solve(hessian, score)
                alpha += step
            except np.linalg.LinAlgError:
                # hinv = np.linalg.pinv(hessian)
                # step = hinv @ score
                # alpha += step
                # print("Singular matrix encountered. Data may be perfectly separated.")
                break
            # alpha -= np.mean(alpha)
            # Check convergence
            if np.linalg.norm(step) < tol:
                break
        # print(alpha)
        return alpha

    @staticmethod
    def fit_local_alphax(y_obs, offset, max_iter=20, tol=1e-8):
        """
        Profile multinomial intercepts alpha given offset = X beta.

        Parameters
        ----------
        y_obs : (N, K-1) one-hot encoded (excluding reference)
        offset : (N, K-1)
        """

        N, K_minus_1 = y_obs.shape
        alpha = np.zeros(K_minus_1)

        # observed class counts
        n_k = np.sum(y_obs, axis=0)

        for _ in range(max_iter):
            eta = offset + alpha
            eta_full = np.hstack([np.zeros((N, 1)), eta])

            exp_eta = np.exp(eta_full - np.max(eta_full, axis=1, keepdims=True))
            pi_full = exp_eta / np.sum(exp_eta, axis=1, keepdims=True)

            pi = pi_full[:, 1:]
            mu_k = np.sum(pi, axis=0)

            # fixed-point update
            delta = np.log(np.clip(n_k, 1e-12, None)) - np.log(
                np.clip(mu_k, 1e-12, None)
            )
            alpha += delta

            # enforce identifiability
            # alpha -= np.mean(alpha)

            if np.linalg.norm(delta) < tol:
                break
        return alpha

    @staticmethod
    def fit_local_alpha(y_obs, offset, max_iter=10, tol=1e-6):
        """
        Fits local fixed effects (alpha) given a fixed global offset (X * beta).

        Parameters:
        - y_obs: (N, K-1) one-hot encoded responses (excluding reference category)
        - offset: (N, K-1) global offset
        - max_iter: maximum Fisher scoring iterations
        - tol: convergence tolerance

        Returns:
        - alpha: (K-1,) local fixed effects
        """

        N, K_minus_1 = y_obs.shape
        alpha = np.zeros(K_minus_1)

        # --------------------------------------------------
        # 1. Detect locally present categories
        # --------------------------------------------------
        counts = y_obs.sum(axis=0)
        present = counts > 0

        # If *no* non-reference category is present, nothing is identifiable
        if not np.any(present):
            return alpha

        # Optional: approximate -inf for absent categories
        NEG_INF = -float("inf")  # -3000.0
        alpha[~present] = NEG_INF

        # Work only on present categories
        idx = np.where(present)[0]
        y_obs_p = y_obs[:, idx]
        offset_p = offset[:, idx]
        alpha_p = alpha[idx]

        for _ in range(max_iter):
            # --------------------------------------------------
            # 2. Linear predictors
            # --------------------------------------------------
            eta_p = offset_p + alpha_p
            eta_full = np.hstack(
                [
                    np.zeros((N, 1)),  # reference category
                    eta_p,
                ]
            )

            # --------------------------------------------------
            # 3. Softmax
            # --------------------------------------------------
            exp_eta = np.exp(eta_full - np.max(eta_full, axis=1, keepdims=True))
            pi_full = exp_eta / np.sum(exp_eta, axis=1, keepdims=True)

            # Non-reference probabilities for present categories
            pi_p = pi_full[:, 1:][:, : len(idx)]

            # --------------------------------------------------
            # 4. Score
            # --------------------------------------------------
            score = np.sum(y_obs_p - pi_p, axis=0)

            # --------------------------------------------------
            # 5. Fisher Information
            # --------------------------------------------------
            Kp = len(idx)
            hessian = np.zeros((Kp, Kp))

            for k in range(Kp):
                hessian[k, k] = np.sum(pi_p[:, k] * (1 - pi_p[:, k]))

            for k in range(Kp):
                for j in range(k + 1, Kp):
                    val = -np.sum(pi_p[:, k] * pi_p[:, j])
                    hessian[k, j] = val
                    hessian[j, k] = val

            # Small ridge for numerical stability (does NOT bias meaningfully)
            hessian += 1e-6 * np.eye(Kp)

            # --------------------------------------------------
            # 6. Fisher scoring update
            # --------------------------------------------------
            try:
                step = np.linalg.solve(hessian, score)
            except np.linalg.LinAlgError:
                break

            alpha_p += step

            if np.linalg.norm(step) < tol:
                break

        # --------------------------------------------------
        # 7. Write back results
        # --------------------------------------------------
        alpha[idx] = alpha_p
        return alpha


def get_data(
    data: pl.DataFrame, response: str | List[str], predictors: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    if len(predictors) > 0:
        if FIT_INTERCEPT:
            assert predictors[-1] == constant_colname, (
                "Constant column is not last predictor"
            )
            data = data.with_columns(pl.lit(1).alias(constant_colname))
        X: np.ndarray = data.select(predictors).to_numpy().astype(float)
    else:
        X = None
    y: np.ndarray = data.select(response).to_numpy().astype(float)
    return (y, X)


class ComputationUnit:
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
    ):
        raise NotImplementedError()


class ContinousComputationUnit(ComputationUnit):
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
        **kwargs,
    ):
        y, X = get_data(data, response, predictors)
        if X is None:
            return ComputationHelper.run_const_model(y, Gaussian)

        assert y.shape[0] == X.shape[0] and X.shape[1] == beta.shape[0], (
            "Shape mismatch between response, predictors, and beta"
        )
        return ComputationHelper.run_regression(y=y, X=X, beta=beta, family=Gaussian)


class BinaryComputationUnit(ComputationUnit):
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
        **kwargs,
    ):
        y, X = get_data(data, response, predictors)
        if X is None:
            return ComputationHelper.run_const_model(y, Binomial)
        assert y.shape[0] == X.shape[0] and X.shape[1] == beta.shape[0], (
            "Shape mismatch between response, predictors, and beta"
        )
        return ComputationHelper.run_regression(y=y, X=X, beta=beta, family=Binomial)


class CategoricalComputationUnit(ComputationUnit):
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
        **kwargs,
    ):
        # Identify the dummy columns for the response
        response_dummy_columns = [
            c
            for c in data.columns
            if c.startswith(f"{response}{categorical_separator}")
        ]
        response_dummy_columns = sorted(response_dummy_columns)

        y_full, X = get_data(data, response_dummy_columns, predictors)

        # X = np.vstack([X, np.zeros((y_full.shape[1], X.shape[1]))])
        # y_full = np.vstack([y_full, np.eye(y_full.shape[1])])

        y = y_full[:, 1:]

        if X is not None:
            num_categories = len(response_dummy_columns)  # J
            num_features = len(predictors)  # K

            # Reshape beta (K x (J-1))
            beta = beta.reshape(num_features, -1, order="F")

        if CLIENT_HETEROGENIETY:
            if X is None:
                offset = np.zeros_like(y)
            else:
                offset = X @ beta
            # gamma = ComputationHelper.fit_multinomial_gamma(y=y, offset=offset)
            gamma = ComputationHelper.fit_local_alpha(y, X @ beta)
            # if response == "A":
            #     print(response, "~", predictors)
            # print(gamma)
            gamma = np.tile(np.array(gamma), (y.shape[0], 1))
        else:
            gamma = np.zeros_like(y)

        if X is None:
            eta = np.zeros_like(y)
            mu = np.clip(
                softmax(np.column_stack([np.zeros(y.shape[0]), eta + gamma]), axis=1),
                1e-8,
                1 - 1e-8,
            )  # N x J
            # renormalize
            mu = mu / np.sum(mu, axis=1, keepdims=True)
            # mu = mu[:, 1:]

            # y_full = data.to_pandas()[response_dummy_columns].to_numpy()  # N x J
            logprob = np.log(np.clip(mu, 1e-8, 1))
            llf = np.sum(y_full * logprob)

            xwx = np.empty((0, 0))
            xwz = np.empty((0, 0))

            return {"llf": llf, "xwx": xwx, "xwz": xwz, "rss": 0, "n": 0}

        # Compute eta and mu
        eta = np.clip(X @ beta, -350, 350)  # N x (J-1)
        mu = np.clip(
            softmax(np.column_stack([np.zeros(y.shape[0]), eta + gamma]), axis=1),
            1e-8,
            1 - 1e-8,
        )  # N x J
        # renormalize
        mu = mu / np.sum(mu, axis=1, keepdims=True)
        mu_reduced = mu[:, 1:]  # N x (J-1)

        # Initialize accumulators for XWX and XWz
        XWX = np.zeros(
            (num_features * (num_categories - 1), num_features * (num_categories - 1))
        )
        XWz = np.zeros(num_features * (num_categories - 1))

        # Construct W blocks and z
        for i in range(y.shape[0]):
            y_i = y[i]  # (J-1)
            p_i = mu_reduced[i]
            var_i = np.diag(p_i) - np.outer(p_i, p_i)  # (J-1) x (J-1)

            try:
                var_i_inv = np.linalg.inv(var_i)
            except np.linalg.LinAlgError:
                var_i_inv = np.linalg.pinv(var_i)

            # z_i = (eta[i] - gamma[i]) + var_i_inv @ (y_i - p_i)  # (J-1)
            # because gamma is only added to mu and not to eta, gamma does not need to be subtracted from eta here
            z_i = (eta[i]) + LR * var_i_inv @ (y_i - p_i)  # (J-1)

            # Compute local contributions to XWX and XWz
            Xi = np.kron(np.eye(num_categories - 1), X[i : i + 1])  # (J-1) x (J-1)*K
            Wi = var_i  # (J-1) x (J-1)
            XWX += Xi.T @ Wi @ Xi
            XWz += Xi.T @ Wi @ z_i

        # XWX *= LR

        # Compute log-likelihood
        logprob = np.log(np.clip(mu, 1e-8, 1))
        llf = np.sum(y_full * logprob)

        # print(np.linalg.norm(XWz.reshape(-1), 2))

        # only use non-dummy values in rss and n for gaussian regression
        return {"llf": llf, "xwx": XWX, "xwz": XWz.reshape(-1, 1), "rss": 0, "n": 0}


class OrdinalComputationUnit:  # (ComputationUnit):
    @staticmethod
    def fix_sign(mus_diff):
        # fix negative probs
        sign_fix = np.column_stack(mus_diff)
        problematic_indices = np.where(sign_fix < 0)[0]
        if len(problematic_indices) > 0:
            problem_probs = np.abs(sign_fix[problematic_indices])
            row_sums = np.clip(np.sum(problem_probs, axis=1, keepdims=True), 1e-8, None)
            normalized_probs = problem_probs / row_sums
            sign_fix[problematic_indices] = normalized_probs
            mus_diff = [sign_fix[:, i] for i in range(len(mus_diff))]
        mus_diff = [np.clip(p, 1e-8, None) for p in mus_diff]
        return mus_diff

    # @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
        **kwargs,
    ):
        # gamma_store = kwargs["ordinal_gamma_store"]
        # regression_key = (response, tuple(sorted(predictors)))

        # Identify the dummy columns for the response
        response_dummy_columns = [
            c for c in data.columns if c.startswith(f"{response}{ordinal_separator}")
        ]
        response_dummy_columns = sorted(
            response_dummy_columns, key=lambda x: int(x.split(ordinal_separator)[1])
        )
        y, X = get_data(data, response, predictors)

        num_levels = len(response_dummy_columns)  # J
        num_features = len(predictors)  # K

        if X is not None:
            # empty init
            xwx = np.zeros((len(beta), len(beta)))
            xwz = np.zeros((len(beta), 1))

            # Reshape beta (K x (J-1))
            beta = beta.reshape(num_features, num_levels - 1, order="F")  # -1 for ref

            mus = []
            for i, (level, beta_i) in enumerate(
                zip(response_dummy_columns[:-1], beta.T)
            ):
                level_int = int(level.split(ordinal_separator)[-1])
                level_y = (y.squeeze() <= level_int).astype(float)

                level_eta, level_mu, level_gamma = ComputationHelper.run_prediction(
                    y=level_y, X=X, beta=beta_i, family=Binomial
                )
                # print(gamma_store)
                # if regression_key not in gamma_store:
                #     gamma_store[regression_key] = {}
                # if level_int not in gamma_store[regression_key]:
                #     gamma_store[regression_key][level_int] = 0
                # level_gamma, update = ComputationHelper.fit_local_gamma_ordinal(
                #     y,
                #     X @ beta,
                #     Binomial,
                #     gamma_store[regression_key][level_int],
                # )
                # gamma_store[regression_key][level_int] = level_gamma
                # level_gamma = self.gamma_store[regression_key][
                #     level_int
                # ] + gamma_blend_factor * (
                #     level_gamma - self.gamma_store[regression_key][level_int]
                # )
                # self.gamma_store[regression_key][level_int] = level_gamma

                # if len(predictors) == 2:
                #     print(level_int, level_gamma)
                mus.append(level_mu)
                level_xwx, level_xwz = ComputationHelper.get_irls_step(
                    y=level_y,
                    X=X,
                    eta=level_eta,
                    mu=level_mu,
                    gamma=level_gamma,
                    family=Binomial,
                )

                offset = i * num_features
                xwx[offset : offset + num_features, offset : offset + num_features] = (
                    level_xwx
                )
                xwz[offset : offset + num_features, :] = level_xwz.reshape((-1, 1))
        else:
            xwx = np.empty((0, 0))
            xwz = np.empty((0, 0))
            mus = []
            for i, level in enumerate(response_dummy_columns[:-1]):
                level_int = int(level.split(ordinal_separator)[-1])
                level_y = (y.squeeze() <= level_int).astype(float)

                _, level_mu, _ = ComputationHelper.run_const_predict(
                    y=level_y, family=Binomial
                )
                mus.append(level_mu)

        mus_diff = [mus[0]]  # P(Y=0)
        mus_diff.extend(
            [mus[i] - mus[i - 1] for i in range(1, len(mus))]
        )  # P(Y=i) = P(Y<=i)-P(Y<=i-1)
        mus_diff.append(
            1 - mus[-1]
        )  # P(Y=K) = 1-P(Y<=K-1) # TODO: ref class is last in this setup -check again

        mus_diff = OrdinalComputationUnit.fix_sign(mus_diff)

        llf = 0
        reference_level_indices = np.ones(len(data))
        for i, level in enumerate(response_dummy_columns[:-1]):
            level_int = int(level.split(ordinal_separator)[-1])
            mu_diff = mus_diff[i]
            current_level_indices = data[response].to_numpy() == level_int
            reference_level_indices = reference_level_indices * (
                1 - current_level_indices
            )

            llf += np.sum(np.log(np.take(mu_diff, current_level_indices.nonzero()[0])))
        mu_diff = mus_diff[-1]
        llf += np.sum(np.log(np.take(mu_diff, reference_level_indices.nonzero()[0])))
        result = {"llf": llf, "xwx": xwx, "xwz": xwz, "rss": 0, "n": 0}
        return result


regression_computation_map = {
    VariableType.CONTINUOS: ContinousComputationUnit,
    VariableType.BINARY: BinaryComputationUnit,
    VariableType.CATEGORICAL: CategoricalComputationUnit,
    VariableType.ORDINAL: OrdinalComputationUnit,
}


class Client:
    def __init__(
        self, id: str, data: pl.DataFrame, _network_fetch_function=lambda x: x
    ):
        self._network_fetch_function = _network_fetch_function
        self.id = id
        self.data: pl.DataFrame = data
        self.schema: Dict[str, VariableType] = {
            column: polars_dtype_map[dtype]
            for column, dtype in dict(self.data.schema).items()
        }

        for column in self.schema:
            assert categorical_separator not in column, (
                f"Variable name {column} contains reserved substring {categorical_separator}"
            )
            assert ordinal_separator not in column, (
                f"Variable name {column} contains reserved substring {ordinal_separator}"
            )
            assert constant_colname != column, (
                f"Variable name {column} is a reserved name"
            )

        self.categorical_expressions: Dict[str, List[str]] = {
            column: self.data.select(column)
            .to_dummies(separator=categorical_separator)
            .columns
            for column, dtype in self.schema.items()
            if dtype == VariableType.CATEGORICAL
        }
        self.ordinal_expressions: Dict[str, List[str]] = {
            column: self.data.select(pl.col(column))
            .to_dummies(separator=ordinal_separator)
            .columns
            for column, dtype in self.schema.items()
            if dtype == VariableType.ORDINAL
        }

        self.global_categorical_expressions: Optional[Dict[str, List[str]]] = None
        self.global_ordinal_expressions: Optional[Dict[str, List[str]]] = None
        self.expanded_data: Optional[pl.DataFrame] = None

        self.contributing_clients: Dict[str, Client] = {}
        self.response_masking = {}

    def get_id(self):
        return self.id

    def get_schema(self):
        return self.schema

    def get_categorical_expressions(self):
        return self.categorical_expressions

    def get_ordinal_expressions(self):
        return self.ordinal_expressions

    def set_clients(self, clients):
        del clients[self.id]  # remove self vom clients
        self.contributing_clients = clients

    def set_global_expressions(
        self,
        categorical_expressions: Dict[str, List[str]],
        ordinal_expressions: Dict[str, List[str]],
    ):
        self.global_categorical_expressions = categorical_expressions
        self.global_ordinal_expressions = ordinal_expressions

        # expand categoricals
        all_possible_categorical_expressions = set(
            [li for l in categorical_expressions.values() for li in l]
        )
        temp = self.data.select(
            [
                column
                for column, dtype in self.schema.items()
                if dtype == VariableType.CATEGORICAL
            ]
        )
        _data = self.data.to_dummies(
            [
                column
                for column, dtype in self.schema.items()
                if dtype == VariableType.CATEGORICAL
            ],
            separator=categorical_separator,
        )
        _data = _data.with_columns(temp)
        missing_cols = list(all_possible_categorical_expressions - set(_data.columns))
        _data = _data.with_columns(*[pl.lit(0.0).alias(c) for c in missing_cols])

        # expand ordinals
        all_possible_ordinal_expressions = set(
            [li for l in ordinal_expressions.values() for li in l]
        )
        _data = _data.to_dummies(
            [
                column
                for column, dtype in self.schema.items()
                if dtype == VariableType.ORDINAL
            ],
            separator=ordinal_separator,
        )
        missing_cols = list(all_possible_ordinal_expressions - set(_data.columns))
        _data = _data.with_columns(*[pl.lit(0.0).alias(c) for c in missing_cols])

        # keep original ordinal variables
        _data = _data.with_columns(
            self.data.select(
                [
                    column
                    for column, dtype in self.schema.items()
                    if dtype == VariableType.ORDINAL
                ]
            )
        )

        self.expanded_data = _data
        self.local_effect_store_ordinal = {}

    def exchange_masks(self, betas):
        betas = self._network_fetch_function(betas)
        for client_id, client in self.contributing_clients.items():
            for test_key, beta in betas.items():
                if test_key not in self.response_masking:
                    self.response_masking[test_key] = {}
                if client_id in self.response_masking[test_key]:
                    continue

                llf_mask = np.random.uniform(-1e8, 1e8, 1).item()
                xwx_mask = np.random.uniform(-1e8, 1e8, (len(beta), len(beta))).astype(
                    np.float128
                )
                xwz_mask = np.random.uniform(-1e8, 1e8, (len(beta), 1)).astype(
                    np.float128
                )
                rss_mask = np.random.uniform(-1e8, 1e8, 1).astype(
                    np.float128
                )  # .item()
                n_mask = np.random.uniform(-1e8, 1e8, 1).astype(np.float128)  # .item()

                client_id_pair = (
                    (self.id, client_id)
                    if self.id < client_id
                    else (client_id, self.id)
                )

                self.add_mask(
                    client_id_pair,
                    test_key,
                    llf_mask=-llf_mask,
                    xwx_mask=-xwx_mask,
                    xwz_mask=-xwz_mask,
                    rss_mask=-rss_mask,
                    n_mask=-n_mask,
                )
                client.add_mask(
                    client_id_pair,
                    test_key,
                    llf_mask=llf_mask,
                    xwx_mask=xwx_mask,
                    xwz_mask=xwz_mask,
                    rss_mask=rss_mask,
                    n_mask=n_mask,
                )

    def add_mask(
        self, sender_id, test_key, llf_mask, xwx_mask, xwz_mask, rss_mask, n_mask
    ):
        xwx_mask = self._network_fetch_function(xwx_mask)
        xwz_mask = self._network_fetch_function(xwz_mask)
        if test_key not in self.response_masking:
            self.response_masking[test_key] = {}
        self.response_masking[test_key][sender_id] = {
            "llf": llf_mask,
            "xwx": xwx_mask,
            "xwz": xwz_mask,
            "rss": rss_mask,
            "n": n_mask,
        }

    def apply_masks(self, test_key, irls_step_result):
        if not ADDITIVE_MASKING:
            return irls_step_result
        assert (
            len(self.response_masking[test_key])
            == len(self.contributing_clients)  # you dont have a mask with yourself
        )
        llf_mask = sum(mask["llf"] for mask in self.response_masking[test_key].values())
        xwx_mask = sum(mask["xwx"] for mask in self.response_masking[test_key].values())
        xwz_mask = sum(mask["xwz"] for mask in self.response_masking[test_key].values())
        rss_mask = sum(mask["rss"] for mask in self.response_masking[test_key].values())
        n_mask = sum(mask["n"] for mask in self.response_masking[test_key].values())
        irls_step_result["llf"] += llf_mask  # self.response_masking[test_key]["llf"]
        irls_step_result["xwx"] += xwx_mask  # self.response_masking[test_key]["xwx"]
        irls_step_result["xwz"] += xwz_mask  # self.response_masking[test_key]["xwz"]
        irls_step_result["rss"] += rss_mask  # self.response_masking[test_key]["rss"]
        irls_step_result["n"] += n_mask  # self.response_masking[test_key]["n"]
        return irls_step_result

    def compute(
        self,
        required_variables: Set[str],
        betas: Dict[Tuple[str, Tuple[str], int], np.ndarray],
    ):
        betas = self._network_fetch_function(betas)

        if any([v not in self.schema for v in required_variables]):
            result = {}
            for test_key, beta in betas.items():
                result[test_key] = self.apply_masks(
                    test_key,
                    {
                        "llf": 0,
                        "xwx": np.zeros((len(beta), len(beta))),
                        "xwz": np.zeros((len(beta), 1)),
                        "rss": 0,
                        "n": 0,
                    },
                )
            return result

        # print("Client", list("ABCDEFGH")[int(self.id) - 1])

        results = {}
        for test_key, beta in betas.items():
            resp_var, cond_vars, _ = test_key

            new_cond_vars = []
            for cond_var in cond_vars:
                if cond_var in self.global_categorical_expressions:
                    new_cond_vars.extend(
                        self.global_categorical_expressions[cond_var][1:]
                    )
                elif cond_var in self.global_ordinal_expressions:
                    new_cond_vars.extend(self.global_ordinal_expressions[cond_var][:-1])
                else:
                    new_cond_vars.append(cond_var)
            new_cond_vars = sorted(new_cond_vars)
            if FIT_INTERCEPT:
                new_cond_vars.append(constant_colname)
            cond_vars = new_cond_vars
            result = regression_computation_map[self.schema[resp_var]].compute(
                self.expanded_data,
                resp_var,
                cond_vars,
                beta,
                ordinal_gamma_store=self.local_effect_store_ordinal,
            )
            if len(self.contributing_clients) > 0:
                result = self.apply_masks(test_key, result)

            results[test_key] = BetaUpdateData(**result)
        # TODO: maybe just exchange difference of LLF of full and nested models
        return results


class ProxyClient(rpyc.Service):
    def __init__(self, id, data):
        self.client = Client(id, data, _network_fetch_function=rpyc.classic.obtain)
        self.server: rpyc.utils.server.ThreadedServer = None

        # expose alle Methoden sofort:
        for name in dir(self.client):
            if callable(getattr(self.client, name)) and not name.startswith("_"):
                setattr(self, name, getattr(self.client, name))

    def __del__(self):
        self.close()

    def start(self, port):
        if self.server is not None:
            self.close()
        self.server = rpyc.utils.server.ThreadedServer(
            self,
            port=port,
            protocol_config={"allow_public_attrs": True, "allow_pickle": True},
        )
        self.server.start()

    def close(self):
        if self.server is None:
            return
        self.server.close()
        self.server = None
