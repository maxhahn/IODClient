
import numpy as np
from typing import List, Dict
from itertools import chain, combinations

from .env import EXPAND_ORDINALS, OVR, RIDGE
from .utils import BetaUpdateData, ClientResponseData, VariableType

class RegressionTest():
    @classmethod
    def create_and_overwrite_beta(cls, y_label, X_labels, beta):
        c = cls(y_label, X_labels)
        c.beta = beta
        return c

    def __init__(self, y_label: str, X_labels: List[str]):
        self.y_label = y_label
        self.X_labels = X_labels
        self.beta = np.zeros(len(X_labels) + 1)

    def update_beta(self, data: List[BetaUpdateData]):
        xwx = sum([d.xwx for d in data])
        xwz = sum([d.xwz for d in data])

        if RIDGE > 0:
            penalty_matrix = RIDGE * np.eye(len(xwx))
            xwx += penalty_matrix

        try:
            xwx_inv = np.linalg.inv(xwx)
        except np.linalg.LinAlgError:
            xwx_inv = np.linalg.pinv(xwx)

        if RIDGE > 0:
            self.beta = (xwx_inv @ xwz) + RIDGE * xwx_inv @ self.beta
        else:
            self.beta = xwx_inv @ xwz

    def __lt__(self, other):
        if len(self.X_labels) < len(other.X_labels): return True
        elif len(self.X_labels) > len(other.X_labels): return False

        if self.y_label < other.y_label: return True
        elif self.y_label > other.y_label: return False

        if tuple(sorted(self.X_labels)) < tuple(sorted(other.X_labels)): return True
        elif tuple(sorted(self.X_labels)) > tuple(sorted(other.X_labels)): return False

        return True

    def __repr__(self):
        return f'RegressionTest {self.y_label} ~ {", ".join(self.X_labels + ["1"])} - beta: {self.beta}'

class Test():
    def __init__(self,
        y_label,
        X_labels: List[str],
        y_labels: List[str] = None,
        max_iterations=25,
        y_type=None
    ):
        self.y_label = y_label
        self.X_labels = X_labels
        if y_labels is None: y_labels = [y_label]
        self.y_labels = y_labels
        self.tests: Dict[str, RegressionTest] = {_y_label: RegressionTest(_y_label, X_labels) for _y_label in y_labels}

        if y_type == VariableType.CATEGORICAL and OVR == 0:
            _beta = np.concatenate([t.beta for t in self.tests.values()])
            self.tests = {y_label: RegressionTest.create_and_overwrite_beta(y_label, X_labels, _beta)}

        self.llf = None
        self.last_deviance = None
        self.deviance = 0
        self.iterations = 0
        self.max_iterations = max_iterations

    def is_finished(self):
        return self.get_change_in_deviance() < 1e-3 or self.iterations >= self.max_iterations

    def update_betas(self, data: Dict[str, ClientResponseData]):
        self.llf = {client_id: client_response.llf for client_id, client_response in data.items()}
        self.last_deviance = self.deviance
        self.deviance = sum(client_response.deviance for client_response in data.values())

        if self.is_finished():
            return

        beta_update_data = [client_response.beta_update_data for client_response in data.values()]
        # Transform data from list of dicts to dict of lists => all data for one y_label grouped together
        beta_update_data = {k: [dic[k] for dic in beta_update_data] for k in beta_update_data[0]}

        for y_label, _data in beta_update_data.items():
            self.tests[y_label].update_beta(_data)
        self.iterations += 1

    def get_degrees_of_freedom(self):
        # len tests -> num_cats -1
        # len X_labels + 1 -> x vars, intercept
        return len(self.y_labels)*(len(self.X_labels) + 1)

    def get_llf(self, client_subset=None):
        if client_subset is not None:
            return sum([llf for client_id, llf in self.llf.items() if client_id in client_subset])
        return sum([llf for llf in self.llf.values()]) if self.llf is not None else 0

    def get_providing_clients(self):
        if self.llf is None:
            return []
        return set(self.llf.keys())

    def get_beta(self):
        return {t.y_label: t.beta for t in self.tests.values()}

    def get_required_labels(self):
        vars = {self.y_label}
        for var in self.X_labels:
            if '__cat__' in var:
                vars.add(var.split('__cat__')[0])
            elif '__ord__' in var:
                vars.add(var.split('__ord__')[0])
            else:
                vars.add(var)
        return vars

    def get_change_in_deviance(self):
        if self.last_deviance is None:
            return 1
        return abs(self.deviance - self.last_deviance)

    def get_relative_change_in_deviance(self):
        if self.last_deviance is None:
            return 1
        return abs(self.deviance - self.last_deviance) / (1e-5 + abs(self.deviance))

    def __repr__(self):
        test_string = "\n\t- "  + "\n\t- ".join([str(t) for t in sorted(self.tests.values())])
        test_title = f'{self.y_label} ~ {",".join(list(set([l.split("__")[0] for l in self.X_labels])))},1'
        return f'Test {test_title} - llf: {self.get_llf()}, deviance: {self.deviance}, {self.iterations}/{self.max_iterations} iterations{test_string}'

    def __eq__(self, other):
        req_labels = self.get_required_labels()
        other_labels = other.get_required_labels()
        return (
            len(req_labels) == len(other_labels) and
            self.y_label == other.y_label and
            tuple(sorted(self.X_labels)) == tuple(sorted(other.X_labels))
        )

    def __lt__(self, other):
        req_labels = self.get_required_labels()
        other_labels = other.get_required_labels()
        if len(req_labels) < len(other_labels):
            return True
        elif len(req_labels) > len(other_labels):
            return False

        if self.y_label < other.y_label:
            return True
        elif self.y_label > other.y_label:
            return False

        if tuple(sorted(self.X_labels)) < tuple(sorted(other.X_labels)):
            return True
        elif tuple(sorted(self.X_labels)) > tuple(sorted(other.X_labels)):
            return False

        return False

class TestEngine2():
    def __init__(
        self,
        client_schemas,
        schema,
        category_expressions,
        ordinal_expressions,
        max_regressors=None,
        max_iterations=25,
        test_targets=None
        ):

        self.tests = []
        self.required_tests = {}

        variable_availability = {}
        for c, _schema in client_schemas.items():
            for v in _schema:
                variable_availability[v] = variable_availability.get(v, set([])) | {c}
        print(variable_availability)

        variables = set(schema.keys())
        max_conditioning_set_size = min(len(variables)-1, max_regressors) if max_regressors is not None else len(variables)-1

        all_test_targets = set(sum([(a,b) for a,b,_ in test_targets], ())) if test_targets is not None else None

        for y_var in variables:
            variables_without_y = variables - {y_var}
            for x_var in variables_without_y:
                set_of_possible_regressors = variables_without_y - {x_var}
                powerset_of_regressors = chain.from_iterable(combinations(set_of_possible_regressors, r) for r in range(0, max_conditioning_set_size+1))
                for cond_set in powerset_of_regressors:
                    cond_set = tuple(sorted(list(cond_set)))
                    self.required_tests[(y_var, cond_set)] = self.required_tests.get((y_var, cond_set), []) + [x_var]
        print(self.required_tests)

        self.required_client_tests = {}
        for req_test, x_vars in self.required_tests.items():
            y_var, cond_set = req_test
            for x_var in x_vars:
                potential_clients = [variable_availability[v] for v in [y_var]+list(cond_set)]
                #potential_clients_full_model = potential_clients + [variable_availability[x_var]]
                potential_clients = set.intersection(*potential_clients)
                #potential_clients_full_model = set.intersection(*potential_clients_full_model)
                #print(f'{y_var} ~ {x_var}, {cond_set} -> {potential_clients}, {potential_clients_full_model}')



class TestEngine():
    def __init__(self,
                 schema,
                 category_expressions,
                 ordinal_expressions,
                 max_regressors=None,
                 max_iterations=25,
                 test_targets=None
                 ):

        self.tests = []
        self.max_iterations = max_iterations
        self.bad_tests = []

        variables = set(schema.keys())
        max_conditioning_set_size = min(len(variables)-1, max_regressors) if max_regressors is not None else len(variables)-1

        all_test_targets = set(sum([(a,b) for a,b,_ in test_targets], ())) if test_targets is not None else None

        for y_var in variables:
            if all_test_targets is not None and y_var not in all_test_targets:
                continue
            set_of_possible_regressors = variables - {y_var}
            powerset_of_regressors = chain.from_iterable(combinations(set_of_possible_regressors, r) for r in range(0, max_conditioning_set_size+1))

            # expand categorical features in regressor sets
            expanded_powerset_of_regressors = []
            for variable_set in powerset_of_regressors:
                for _var, expressions in category_expressions.items():
                    if _var in variable_set:
                        variable_set = (set(variable_set) - {_var}) | set(sorted(list(expressions)[1:])) # drop first cat
                if EXPAND_ORDINALS:
                    for _var, expressions in ordinal_expressions.items():
                        if _var in variable_set:
                            variable_set = (set(variable_set) - {_var}) | set(sorted(list(expressions)[1:])) # drop first cat
                expanded_powerset_of_regressors.append(variable_set)
            powerset_of_regressors = expanded_powerset_of_regressors

            if schema[y_var] == VariableType.CONTINUOS:
                self.tests.extend([Test(
                    y_label=y_var,
                    X_labels=sorted(list(x_vars)),
                    max_iterations=max_iterations,
                    y_type=schema[y_var]
                ) for x_vars in powerset_of_regressors])
            elif schema[y_var] == VariableType.BINARY:
                self.tests.extend([Test(
                    y_label=y_var,
                    X_labels=sorted(list(x_vars)),
                    max_iterations=max_iterations,
                    y_type=schema[y_var]
                ) for x_vars in powerset_of_regressors])
            elif schema[y_var] == VariableType.CATEGORICAL:
                assert y_var in category_expressions, f'Categorical variable {y_var} is not in expression mapping'
                self.tests.extend([Test(
                    y_label=y_var,
                    X_labels=sorted(list(x_vars)),
                    y_labels=category_expressions[y_var][:-1],
                    max_iterations=max_iterations,
                    y_type=schema[y_var]
                ) for x_vars in powerset_of_regressors])
            elif schema[y_var] == VariableType.ORDINAL:
                assert y_var in ordinal_expressions, f'Ordinal variable {y_var} is not in expression mapping'
                self.tests.extend([Test(
                    y_label=y_var,
                    X_labels=sorted(list(x_vars)),
                    y_labels=ordinal_expressions[y_var][:-1],
                    max_iterations=max_iterations,
                    y_type=schema[y_var]
                ) for x_vars in powerset_of_regressors])
            else:
                raise Exception(f'Unknown variable type {schema[y_var]} encountered!')

        self.tests: List[Test] = sorted(self.tests)
        self.current_test_index = 0

    def is_finished(self):
        return self.current_test_index >= len(self.tests)

    def get_currently_required_labels(self):
        if self.is_finished():
            return None
        current_test = self.tests[self.current_test_index]
        return current_test.get_required_labels()

    def get_current_test_parameters(self):
        if self.is_finished():
            return None, None, None
        current_test = self.tests[self.current_test_index]
        return current_test.y_label, current_test.X_labels, current_test.get_beta()

    def update_current_test(self, client_responses: Dict[str, ClientResponseData]):
        if self.is_finished():
            return
        current_test = self.tests[self.current_test_index]
        current_test.update_betas(client_responses)
        if current_test.is_finished():
            self.current_test_index += 1

    def remove_current_test(self):
        if self.is_finished():
            return
        current_test = self.tests[self.current_test_index]
        self.bad_tests.append(current_test)
        self.tests = self.tests[:self.current_test_index] + self.tests[self.current_test_index+1:]
