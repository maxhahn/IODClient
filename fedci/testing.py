from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy

from .env import get_env_debug, get_env_fit_intercept, get_env_line_search, get_env_lm_damping, get_env_ridge
from .utils import BetaUpdateData, VariableType


class RegressionTest:
    def __init__(
        self,
        response: str,
        response_type: VariableType,
        predictors: List[str],
        params: Tuple[int, int],
        convergence_threshold: float = 1e-6,
        max_iterations: int = 25,
    ):
        self.response: str = response
        self.response_type: VariableType = response_type
        self.predictors: Set[str] = set(predictors)

        self.num_classes, self.num_parameters = params
        self.dof = self.num_classes * self.num_parameters
        self.beta = np.zeros((self.dof, 1))
        self.alpha = 1.0

        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.early_stop = False
        self.bad_fit = False

        self.previous_xwx = None
        self.previous_xwz = None
        self.previous_llf: float = -float("inf")
        self.llf: float = -float("inf")
        self.iterations: int = 0
        self.reposition_count = 0
        self.lm_lambda = 1

    def __repr__(self):
        (
            response,
            predictors,
            iteration,
        ) = (
            self.response,
            tuple(sorted(list(self.predictors))),
            self.iterations,
        )
        predictors = sorted(list(predictors))
        if get_env_fit_intercept():
            predictors.append("1")
        return f"RegressionTest {response} ~ {', '.join(predictors)} - iteration {iteration}/{self.max_iterations}{' - not finished' if not self.is_finished() else ''}"

    def get_change_in_llf(self):
        if self.previous_llf is None:
            return 1
        return abs(self.llf - self.previous_llf)

    def get_relative_change_in_llf(self):
        if self.previous_llf is None:
            return 1
        return abs(self.llf - self.previous_llf) / (1 + abs(self.previous_llf))

    def is_finished(self):
        return (
            self.get_change_in_llf() < self.convergence_threshold
            or self.iterations >= self.max_iterations
            or self.early_stop
        )

    def get_test_parameters(self):
        return (
            self.response,
            tuple(sorted(list(self.predictors))),
            self.iterations,
            self.beta,
            self.num_classes,
            self.num_parameters,
        )

    def _get_new_beta(self, xwx, xwz):
        k = xwx.shape[0]
        if get_env_ridge() > 0:
            if get_env_fit_intercept():
                penalty_matrix = np.zeros((k, k))
                penalty_matrix[:-1, :-1] = get_env_ridge() * np.eye(k - 1)
            else:
                penalty_matrix = get_env_ridge() * np.eye(k)
            xwx += penalty_matrix

        if get_env_lm_damping():
            lm_penalty_matrix = self.lm_lambda * np.eye(k)
            xwx += lm_penalty_matrix

        try:
            xwx_inv = np.linalg.inv(xwx)
        except np.linalg.LinAlgError:
            xwx_inv = np.linalg.pinv(xwx)

        if get_env_debug() > 3:
            print(
                f"{self.response} ~ {sorted(list(self.predictors))} - Iteration {self.iterations}/{self.max_iterations}"
            )
            print(
                f"{self.beta.reshape(-1).tolist()} -> {(xwx_inv @ xwz).reshape(-1).tolist()}"
            )
            print(
                f"{'None' if self.previous_llf is None else self.previous_llf} -> {self.llf}"
            )

        self.previous_xwx = xwx
        self.previous_xwz = xwz
        return self.beta + self.alpha * ((xwx_inv @ xwz) - self.beta)

    def update_parameters(self, update: List[BetaUpdateData]):
        if self.is_finished():
            return

        llf = sum([_update.llf for _update in update])
        xwx = sum([_update.xwx for _update in update])
        xwz = sum([_update.xwz for _update in update])
        n = int(sum([_update.n for _update in update]))

        # if self.response == "A" and self.predictors == {"B", "C", "E"}:
        #     print(self.lm_lambda)
        # if abs(self.llf - llf) > 10:
        #    print(self.llf, llf, self.alpha, self.response, self.predictors)
        if self.response_type == VariableType.CONTINUOS and n > 0:
            rss = sum([_update.rss for _update in update])
            sigma2 = np.clip(rss / n, 1e-10, None)
            llf = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * n
            llf = llf.astype(np.float64).item()  # uses np.float128 for higher precision

        #print(np.linalg.cond(xwx))

        if get_env_line_search() and self.llf > llf:
            # if self.response == "E" and self.predictors == {"C"}:
            #     print(self.llf, llf)
            # if no small step improves llf, stop fitting
            if self.alpha <= 1e-8:
                self.alpha = 1.0
                if self.reposition_count >= 10:
                    # asd
                    self.bad_fit = True
                    self.early_stop = True
                    #print('Early stop no improvement')
                    return
                if self.lm_lambda <= 1:
                    self.lm_lambda = 1
                if self.lm_lambda <= 1e8:
                    self.lm_lambda *= 2
                self.beta = self._get_new_beta(
                    self.previous_xwx,
                    self.previous_xwz
                    #+ np.random.normal(0, 1e-3, self.previous_xwz.shape),
                )
                self.reposition_count += 1
                return
            # got worse
            # For this step, no llf is available. So check with previous llf and redo last update with new alpha

            self.alpha *= 0.5

            # self.iterations += 1
            # self.max_iterations += 1

            self.beta = self._get_new_beta(self.previous_xwx, self.previous_xwz)
            return

        self.previous_llf = self.llf
        self.llf = llf
        self.alpha = 1.0
        if self.lm_lambda > 1e-12:
            self.lm_lambda /= 10
        beta = self._get_new_beta(xwx, xwz)
        # TODO: improve with variables
        # if alpha <1e-8 und update wird trotzdem gemacht: break, test ende
        if (
            self.iterations == 0
            and np.linalg.norm(self.beta - beta) < 1e-4
            or np.linalg.norm(self.beta - beta) < 1e-8
        ):
            #('Early stop small step')
            self.early_stop = True
            return
        self.beta = beta
        self.iterations += 1


class LikelihoodRatioTest:
    def __init__(
        self, restricted_test: RegressionTest, unrestricted_test: RegressionTest
    ):
        assert restricted_test.response == unrestricted_test.response, (
            "The provided tests do not fit the same response variable"
        )
        self.response: str = restricted_test.response

        assert (
            len(set(restricted_test.predictors) - set(unrestricted_test.predictors))
            == 0
        ), "The provided tests are not properly nested"

        predictor_difference = list(
            set(unrestricted_test.predictors) - set(restricted_test.predictors)
        )

        assert len(predictor_difference) == 1, (
            "The provided tests differ in more than one variable"
        )

        self.test_variable: str = predictor_difference[0]
        self.conditioning_set: Set[str] = restricted_test.predictors

        self.restricted_test: RegressionTest = restricted_test
        self.unrestricted_test: RegressionTest = unrestricted_test

        self.iterations = 0
        self.bad_fit = False

        self.p_value: Optional[float] = None

    def __repr__(self):
        if self.p_value is None:
            val = "not finished"
        else:
            val = f"p: {self.p_value:.4f}"
        restricted_test_string = self.restricted_test.__repr__()
        restricted_test_string = "\n\t".join(restricted_test_string.split("\n"))
        unrestricted_test_string = self.unrestricted_test.__repr__()
        unrestricted_test_string = "\n\t".join(unrestricted_test_string.split("\n"))
        return f"LikelihoodRatioTest - y: {self.response}, x: {self.test_variable}, S: {sorted(list(self.conditioning_set))}, {val}\n\t- {restricted_test_string}\n\t- {unrestricted_test_string}"

    def __lt__(self, other):
        if len(self.conditioning_set) < len(other.conditioning_set):
            return True
        if (
            len(self.conditioning_set) == len(other.conditioning_set)
            and self.response < other.response
        ):
            return True
        if (
            len(self.conditioning_set) == len(other.conditioning_set)
            and self.response == other.response
            and self.test_variable < other.test_variable
        ):
            return True
        if (
            len(self.conditioning_set) == len(other.conditioning_set)
            and self.response == other.response
            and self.test_variable == other.test_variable
            and sorted(list(self.conditioning_set))
            < sorted(list(other.conditioning_set))
        ):
            return True
        return False

    def is_finished(self):
        finished = (
            self.restricted_test.is_finished() and self.unrestricted_test.is_finished()
        )
        if finished and self.p_value is None:
            self._set_p_value()
            self.bad_fit = self.restricted_test.bad_fit | self.unrestricted_test.bad_fit
        return finished

    def get_test_parameters(self):
        test_parameters = {}
        if not self.restricted_test.is_finished():
            response, predictors, iteration, beta, _, _ = (
                self.restricted_test.get_test_parameters()
            )
            test_parameters[(response, predictors, iteration)] = beta
        if not self.unrestricted_test.is_finished():
            response, predictors, iteration, beta, _, _ = (
                self.unrestricted_test.get_test_parameters()
            )
            test_parameters[(response, predictors, iteration)] = beta
        return test_parameters

    def get_betas(self):
        betas = {}
        response, predictors, iteration, beta, num_classes, num_predictors = (
            self.restricted_test.get_test_parameters()
        )
        betas[(response, predictors, iteration)] = beta.reshape(
            num_classes, num_predictors
        )
        response, predictors, iteration, beta, num_classes, num_predictors = (
            self.unrestricted_test.get_test_parameters()
        )
        betas[(response, predictors, iteration)] = beta.reshape(
            num_classes, num_predictors
        )
        return betas

    def update_parameters(self, update: List[Dict[Tuple[str], BetaUpdateData]]):
        if not self.restricted_test.is_finished():
            self.restricted_test.update_parameters(
                [
                    _update[tuple(sorted(list(self.restricted_test.predictors)))]
                    for _update in update
                ]
            )
        if not self.unrestricted_test.is_finished():
            self.unrestricted_test.update_parameters(
                [
                    _update[tuple(sorted(list(self.unrestricted_test.predictors)))]
                    for _update in update
                ]
            )
        self.iterations += 1

    def _set_p_value(self):
        t0_llf = self.restricted_test.llf
        t1_llf = self.unrestricted_test.llf

        t0_dof = self.restricted_test.dof
        t1_dof = self.unrestricted_test.dof

        self.p_value = scipy.stats.chi2.sf(
            2 * (t1_llf - t0_llf), t1_dof - t0_dof
        ).item()

        if get_env_debug() >= 2:
            print(
                f"*** Calculating p value for independence of {self.response} from {self.test_variable} given {sorted(list(self.conditioning_set))}"
            )
            print(f"{t1_dof - t0_dof} DOFs = {t1_dof} T1 DOFs - {t0_dof} T0 DOFs")
            print(
                f"{2 * (t1_llf - t0_llf):.4f} Test statistic = 2*({t1_llf:.4f} T1 LLF - {t0_llf:.4f} T0 LLF)"
            )
            print(f"p value = {self.p_value:.6f}")


class SymmetricLikelihoodRatioTest:
    def __init__(self, lrt1: LikelihoodRatioTest, lrt2: LikelihoodRatioTest):
        assert (
            lrt1.response == lrt2.test_variable
            and lrt1.test_variable == lrt2.response
            and lrt1.conditioning_set == lrt2.conditioning_set
        ), "The provided tests are not symmetrical"

        self.v0 = lrt1.response
        self.v1 = lrt2.response
        self.conditioning_set = lrt1.conditioning_set
        self.iterations = 0
        self.bad_fit = False

        if lrt1.response < lrt2.response:
            self.lrt1: LikelihoodRatioTest = lrt1
            self.lrt2: LikelihoodRatioTest = lrt2
        else:
            self.lrt1: LikelihoodRatioTest = lrt2
            self.lrt2: LikelihoodRatioTest = lrt1

        self.p_value: Optional[float] = None

    def __repr__(self):
        if self.p_value is None:
            val = "not finished"
        else:
            val = f"p: {self.p_value:.4f}"
        lrt1_string = self.lrt1.__repr__()
        lrt1_string = "\n\t".join(lrt1_string.split("\n"))
        lrt2_string = self.lrt2.__repr__()
        lrt2_string = "\n\t".join(lrt2_string.split("\n"))
        return f"SymmetricLikelihoodRatioTest - {self.v0} indep {self.v1}{' | ' + ', '.join(sorted(list(self.conditioning_set))) if len(self.conditioning_set) > 0 else ''}, {val}\n\t- {lrt1_string}\n\t- {lrt2_string}"

    def __lt__(self, other):
        if self.lrt1 < other.lrt1:
            return True
        if self.lrt1 == other.lrt1 and self.lrt2 < other.lrt2:
            return True
        return False

    def is_finished(self):
        finished = self.lrt1.is_finished() and self.lrt2.is_finished()
        if finished and self.p_value is None:
            self._set_p_value()
            self.bad_fit = self.lrt1.bad_fit | self.lrt2.bad_fit
        return finished

    def get_test_parameters(self):
        test_parameters = (
            self.lrt1.get_test_parameters() | self.lrt2.get_test_parameters()
        )
        return test_parameters

    def get_betas(self):
        betas = self.lrt1.get_betas() | self.lrt2.get_betas()
        return betas

    def update_parameters(
        self, update: List[Dict[str, Dict[Tuple[str], BetaUpdateData]]]
    ):
        if not self.lrt1.is_finished():
            self.lrt1.update_parameters(
                [_update[self.lrt1.response] for _update in update]
            )
        if not self.lrt2.is_finished():
            self.lrt2.update_parameters(
                [_update[self.lrt2.response] for _update in update]
            )
        self.iterations += 1

    def _set_p_value(self):
        # if self.lrt1.bad_fit and self.lrt2.bad_fit:
        #     self.p_value = -1
        # elif self.lrt1.bad_fit:
        #     self.p_value = self.lrt2.p_value
        # elif self.lrt2.bad_fit:
        #     self.p_value = self.lrt1.p_value
        # else:
        self.p_value = min(
            2 * min(self.lrt1.p_value, self.lrt2.p_value),
            max(self.lrt1.p_value, self.lrt2.p_value),
        )

        if get_env_debug() >= 2:
            print(
                f"*** Combining p values for symmetry of tests between {self.lrt1.response} and {self.lrt2.response} given {self.lrt1.conditioning_set}"
            )
            print(f"p value {self.lrt1.response}: {self.lrt1.p_value}")
            print(f"p value {self.lrt2.response}: {self.lrt2.p_value}")
            print(f"p value = {self.p_value:.4f}")


class TestEngine:
    def __init__(
        self,
        schema,
        category_expressions,
        ordinal_expressions,
        convergence_threshold=1e-3,
        max_iterations=25,
    ):
        self.schema = schema
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.category_expressions = category_expressions
        self.ordinal_expressions = ordinal_expressions
        self.test = None

    def _get_number_of_parameters(self, resp_var, cond_vars):
        num_params = 1 if get_env_fit_intercept() else 0

        for var in cond_vars:
            if var in self.category_expressions:
                num_params += (
                    len(self.category_expressions[var]) - 1
                )  # -1 for ref category
            elif var in self.ordinal_expressions:
                num_params += (
                    len(self.ordinal_expressions[var]) - 1
                )  # -1 for ref category
            else:
                num_params += 1

        if resp_var in self.category_expressions:
            num_cats = len(self.category_expressions[resp_var]) - 1
        elif resp_var in self.ordinal_expressions:
            num_cats = len(self.ordinal_expressions[resp_var]) - 1
        else:
            num_cats = 1
        return (num_cats, num_params)

    def start_test(self, x: str, y: str, s: List[str]):
        if any([v not in self.schema for v in s + [x, y]]):
            raise Exception("The requested test requires unknown variables")

        x_s_params = self._get_number_of_parameters(x, s)
        x_sy_params = self._get_number_of_parameters(x, s + [y])
        y_s_params = self._get_number_of_parameters(y, s)
        y_sx_params = self._get_number_of_parameters(y, s + [x])

        test_x_restricted = RegressionTest(
            response=x,
            response_type=self.schema[x],
            predictors=s,
            params=x_s_params,
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )
        test_x_unrestricted = RegressionTest(
            response=x,
            response_type=self.schema[x],
            predictors=s + [y],
            params=x_sy_params,
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )
        test_y_restricted = RegressionTest(
            response=y,
            response_type=self.schema[y],
            predictors=s,
            params=y_s_params,
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )
        test_y_unrestricted = RegressionTest(
            response=y,
            response_type=self.schema[y],
            predictors=s + [x],
            params=y_sx_params,
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )

        lrt_x = LikelihoodRatioTest(
            restricted_test=test_x_restricted, unrestricted_test=test_x_unrestricted
        )
        lrt_y = LikelihoodRatioTest(
            restricted_test=test_y_restricted, unrestricted_test=test_y_unrestricted
        )

        self.test = SymmetricLikelihoodRatioTest(lrt1=lrt_x, lrt2=lrt_y)

    def is_finished(self):
        if self.test is None:
            return True
        return self.test.is_finished()

    def get_test_parameters(self):
        return self.test.get_test_parameters()

    def get_required_variables(self):
        return {self.test.v0} | {self.test.v1} | self.test.conditioning_set

    def update_parameters(
        self, update: List[Dict[Tuple[str, Tuple[str], int], BetaUpdateData]]
    ):
        new_update = []
        for _update in update:
            _new_update = {self.test.v0: {}, self.test.v1: {}}
            for (resp_var, cond_vars, _), payload in _update.items():
                _new_update[resp_var][cond_vars] = payload
            new_update.append(_new_update)
        update: List[Dict[str, Dict[Tuple[str], BetaUpdateData]]] = new_update
        return self.test.update_parameters(update)

    def get_result(self):
        if not self.is_finished():
            return None
        return self.test
