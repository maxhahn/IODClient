from typing import List, Optional

import rpyc

from fedci.utils import (
    InitialSchema,
    categorical_separator,
    constant_colname,
    ordinal_separator,
)

from .client import Client
from .env import ADDITIVE_MASKING, DEBUG, FIT_INTERCEPT
from .testing import TestEngine
from .utils import BetaUpdateData


class Server:
    def __init__(
        self,
        clients: List[Client],
        schema: Optional[InitialSchema] = None,
        max_regressors: Optional[int] = None,
        convergence_threshold: Optional[float] = 1e-6,
        max_iterations: Optional[int] = 25,
        _network_fetch_function=lambda x: x,
    ):
        self._network_fetch_function = _network_fetch_function
        self.clients = {client.get_id(): client for client in clients}
        assert len(self.clients) == len(clients), "Some Client IDs were duplicated"

        if schema is None:
            self.schema = {}
            for client in self.clients.values():
                client_schema = client.get_schema()
                for column, dtype in client_schema.items():
                    if column not in self.schema:
                        self.schema[column] = dtype
                        continue
                    assert self.schema[column] == dtype, (
                        f"Schema mismatch between clients detected for variable {column}!"
                    )
            self.categorical_expressions = {}
            self.ordinal_expressions = {}
            for client in self.clients.values():
                for (
                    feature,
                    expressions,
                ) in client.get_categorical_expressions().items():
                    self.categorical_expressions[feature] = sorted(
                        list(
                            set(self.categorical_expressions.get(feature, [])).union(
                                set(expressions)
                            )
                        )
                    )
                for feature, levels in client.get_ordinal_expressions().items():
                    self.ordinal_expressions[feature] = sorted(
                        list(
                            set(self.ordinal_expressions.get(feature, [])).union(
                                set(levels)
                            )
                        ),
                        key=lambda x: int(x.split(ordinal_separator)[-1]),
                    )
        else:
            self.schema = schema.schema
            self.categorical_expressions = {}
            for var, exprs in schema.categorical_expressions.items():
                assert len(exprs) > 0, (
                    f"Received empty list of categorical expressions for variable: {var}"
                )
                assert not any([categorical_separator in expr for expr in exprs]), (
                    f"Received illegal categorical expressions for variable: {var}"
                )
                self.categorical_expressions[var] = [
                    f"{var}{categorical_separator}{expr}" for expr in exprs
                ]
            self.ordinal_expressions = {}
            for var, exprs in schema.ordinal_expressions.items():
                assert len(exprs) > 0, (
                    f"Received empty list of categorical expressions for variable: {var}"
                )
                assert not any([ordinal_separator in expr for expr in exprs]), (
                    f"Received illegal ordinal expressions for variable: {var}"
                )
                try:
                    exprs = [int(expr) for expr in exprs]
                except:
                    assert False, (
                        f"Received non-integer ordinal expression for variable: {var}"
                    )
                self.ordinal_expressions[var] = [
                    f"{var}{ordinal_separator}{expr}" for expr in sorted(exprs)
                ]

        assert constant_colname not in self.schema, (
            f"Received reserved variable name: {constant_colname}"
        )

        for client in self.clients.values():
            client.set_global_expressions(
                self.categorical_expressions, self.ordinal_expressions
            )
        for client in self.clients.values():
            client.set_clients(self.clients.copy())

        self.test_engine: TestEngine = TestEngine(
            schema=self.schema,
            category_expressions=self.categorical_expressions,
            ordinal_expressions=self.ordinal_expressions,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
        )

    def test(self, x: str, y: str, s: List[str]):
        if x > y:
            x, y = y, x
        self.test_engine.start_test(x, y, s)

        while not self.test_engine.is_finished():
            betas = self.test_engine.get_test_parameters()
            clients = self.clients.values()
            if ADDITIVE_MASKING:
                for client in clients:
                    client.exchange_masks(betas)
                # for client in clients:
                #    client.combine_masks(betas)
            updates = []
            for client in clients:
                update = client.compute(betas)
                update = self._network_fetch_function(update)
                update = {
                    k: BetaUpdateData(**v) if type(v) is dict else v
                    for k, v in update.items()
                }
                updates.append(update)

            self.test_engine.update_parameters(updates)
        if DEBUG > 0:
            print("*** Final betas")
            for key, beta in self.test_engine.test.get_betas().items():
                cond_set = sorted(list(key[1]))
                if FIT_INTERCEPT:
                    cond_set.append("1")

                print(f"{key[0]} ~ {','.join(cond_set)} after {key[2]} iterations")
                for _beta in beta.tolist():
                    print(_beta)
        result = self.test_engine.get_result()
        if result.iterations == 1:
            return None
        return result

    def run(self, max_cond_size=None):
        finished_tests = []
        from itertools import chain, combinations

        def get_all_possible_tests(variables, max_cond_size):
            possible_tests = []

            if max_cond_size is None:
                max_cond_size = len(variables) - 2

            for y_var in variables:
                variables_without_y = variables - {y_var}
                for x_var in variables_without_y:
                    if y_var > x_var:
                        continue
                    set_of_possible_regressors = variables_without_y - {x_var}
                    powerset_of_regressors = chain.from_iterable(
                        combinations(set_of_possible_regressors, r)
                        for r in range(0, max_cond_size + 1)
                    )
                    for cond_set in powerset_of_regressors:
                        cond_set = list(cond_set)
                        possible_tests.append((y_var, x_var, cond_set))
            return possible_tests

        all_possible_tests = get_all_possible_tests(self.schema.keys(), max_cond_size)
        while len(all_possible_tests) > 0:
            current_test = all_possible_tests[0]
            del all_possible_tests[0]

            test_result = self.test(*current_test)
            if test_result is None:
                continue

            finished_tests.append(test_result)
        if DEBUG >= 1:
            print("*** All tests")
            for test in finished_tests:
                print(test)
        return finished_tests


class ProxyServerBuilder:
    def __init__(self, cls):
        self.clients = []
        self.cls = cls
        self.max_iterations = 25
        self.schema = None

    def set_schema(self, schema):
        self.schema = schema

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations
        return self

    def add_client(self, hostname, port):
        if (hostname, port) in self.clients:
            print("Client exists already")
            return self
        client = rpyc.connect(
            hostname, port, config={"allow_public_attrs": True, "allow_pickle": True}
        )
        self.clients.append(client)
        return self

    def build(self):
        return self.cls(
            self.clients,
            schema=self.schema,
            max_iterations=self.max_iterations,
        )


class ProxyServer:
    @classmethod
    def builder(cls, **kwargs):
        return ProxyServerBuilder(cls, **kwargs)

    def __init__(self, clients, schema, max_iterations):
        self.clients = [c.root for c in clients]
        self.server = Server(
            self.clients,
            schema=schema,
            max_iterations=max_iterations,
            _network_fetch_function=rpyc.classic.obtain,
        )

    def __getattr__(self, name):
        return getattr(self.server, name)

    def test(self, x, y, s):
        return self.server.test(x, y, s)

    def run(self, max_cond_size=None):
        return self.server.run(max_cond_size)
