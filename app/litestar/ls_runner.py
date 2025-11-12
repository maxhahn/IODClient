import datetime
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional, Set

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from litestar import MediaType, Response, get, post
from litestar.controller import Controller
from litestar.exceptions import HTTPException
from ls_data_structures import (
    Algorithm,
    ExecutionRequest,
    FEDGLMState,
    FEDGLMUpdateData,
    Room,
    RoomDetailsDTO,
)
from ls_env import connections, rooms, user2connection
from ls_helpers import validate_user_request
from rpy2.robjects import numpy2ri, pandas2ri

import fedci


class AlgorithmController(Controller):
    path = "/run"

    def run_iod_on_user_data_fisher(self, dfs, client_labels, alpha):
        ro.r["source"]("./scripts/iod.r")
        aggregate_ci_results_f = ro.globalenv["aggregate_ci_results"]

        with (
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ).context():
            r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
            # r_dfs = ro.ListVector(r_dfs)
            users = client_labels.keys()
            label_list = [ro.StrVector(v) for v in client_labels.values()]

            result = aggregate_ci_results_f(label_list, r_dfs, alpha)

            g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
            g_pag_labels = [
                list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
            ]
            g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
            gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
            gi_pag_labels = [
                list([str(a) for a in x[1]])
                for x in result["Gi_PAG_Label_List"].items()
            ]
            gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]
        return (
            g_pag_list,
            g_pag_labels,
            {u: r for u, r in zip(users, gi_pag_list)},
            {u: l for u, l in zip(users, gi_pag_labels)},
        )

    def iod_r_call_on_combined_data(
        self, df, client_labels, alpha=0.05, procedure="original"
    ):
        with (ro.default_converter + pandas2ri.converter).context():
            ro.r["source"]("./scripts/iod.r")
            iod_on_ci_data_f = ro.globalenv["iod_on_ci_data"]

            labels = sorted(list(set().union(*(client_labels.values()))))

            suff_stat = [
                ("citestResults", ro.conversion.get_conversion().py2rpy(df)),
                ("all_labels", ro.StrVector(labels)),
            ]
            suff_stat = OrderedDict(suff_stat)
            suff_stat = ro.ListVector(suff_stat)
            users = client_labels.keys()
            label_list = [ro.StrVector(v) for v in client_labels.values()]

            result = iod_on_ci_data_f(label_list, suff_stat, alpha, procedure)

            g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
            g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
            g_pag_labels = [
                list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
            ]
            gi_pag_list = [x[1].tolist() for x in result["Gi_PAG_list"].items()]
            gi_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]
            gi_pag_labels = [
                list([str(a) for a in x[1]])
                for x in result["Gi_PAG_Label_List"].items()
            ]

            user_pags = {u: r for u, r in zip(users, gi_pag_list)}
            user_labels = {u: l for u, l in zip(users, gi_pag_labels)}

        return g_pag_list, g_pag_labels, user_pags, user_labels

    def run_meta_analysis_iod(self, data, room_name):
        room = rooms[room_name]

        # gather data of all participants
        participant_data = []
        participant_data_labels = {}
        participants = room.users
        for user in participants:
            conn = user2connection[user]
            participant_data.append(conn.algorithm_data.data)
            participant_data_labels[user] = conn.algorithm_data.data_schema.keys()
        print("XXX", conn.algorithm_data)
        return self.run_iod_on_user_data_fisher(
            participant_data, participant_data_labels, alpha=data.alpha
        )

    def run_fedci_iod(self, data, room_name):
        room: Room = rooms[room_name]

        alpha = data.alpha
        max_cond_size = data.max_conditioning_set

        server = fedci.ProxyServer.builder()
        for username in room.users:
            client_data = user2connection[username].algorithm_data
            try:
                server.add_client(client_data.hostname, client_data.port)
            except:
                raise HTTPException(
                    detail=f"Could not open RPC connection to {username}",
                    status_code=404,
                )
        server = server.build()

        test_results = server.run(max_cond_size=max_cond_size)

        likelihood_ratio_tests = test_results

        all_labels = sorted(list(server.schema.keys()))

        columns = ("ord", "X", "Y", "S", "pvalue")
        rows = []
        for test in sorted(likelihood_ratio_tests):
            s_labels_string = ",".join(
                sorted([str(all_labels.index(l) + 1) for l in test.conditioning_set])
            )
            rows.append(
                (
                    len(test.conditioning_set),
                    all_labels.index(test.v0) + 1,
                    all_labels.index(test.v1) + 1,
                    s_labels_string,
                    test.p_val,
                )
            )

        df = pd.DataFrame(data=rows, columns=columns)

        # let index start with 1
        df.index += 1

        participant_data_labels = {}
        participants = room.users
        for user in participants:
            conn = user2connection[user]
            participant_data_labels[user] = conn.algorithm_data.data_schema.keys()

        try:
            if len(participant_data_labels) > 1:
                result, result_labels, user_results, user_labels = (
                    self.iod_r_call_on_combined_data(
                        df,
                        participant_data_labels,
                        alpha=alpha,
                    )
                )
            else:
                result, result_labels, user_results, user_labels = (
                    self.run_iod_on_user_data_fisher(
                        [df],
                        participant_data_labels,
                        alpha=alpha,
                    )
                )
        except:
            raise HTTPException(detail="Failed to execute IOD", status_code=500)

        room.result = result
        room.result_labels = result_labels

        room.is_processing = False
        room.is_finished = True

        return result, result_labels, user_results, user_labels

    @post("/{room_name:str}")
    async def run(self, data: ExecutionRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)
        room = rooms[room_name]

        # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
        if room.owner_name != data.username:
            raise HTTPException(
                detail="You do not have sufficient authority in this room",
                status_code=403,
            )

        room.is_processing = True
        room.is_locked = True
        room.is_hidden = True
        rooms[room_name] = room

        # ToDo change behavior based on room type:
        #   one run for pvalue aggregation only
        #   multiple runs for fedglm -> therefore, return in this function quickly. Set 'is_processing' and update FedGLMStatus etc
        if room.algorithm == Algorithm.META_ANALYSIS:
            process_func = self.run_meta_analysis_iod
        elif room.algorithm == Algorithm.FEDCI:
            process_func = self.run_fedci_iod
        else:
            raise Exception(f"Encountered unknown algorithm {room.algorithm}")

        try:
            result, result_labels, user_result, user_labels = process_func(
                data, room_name
            )
        except:
            room = rooms[room_name]
            room.is_processing = False
            room.is_locked = True
            room.is_hidden = True
            rooms[room_name] = room
            raise HTTPException(detail="Failed to execute", status_code=500)

        room = rooms[room_name]
        for user in user_result:
            if user not in room.users:
                del user_result[user]

        # print("=" * 20)
        # print(result)
        # print(result_labels)
        # print(user_result)
        # print(user_labels)
        # print("=" * 20)

        room.result = result
        room.result_labels = result_labels
        room.user_results = user_result
        room.user_labels = user_labels
        room.is_processing = False
        room.is_finished = True
        rooms[room_name] = room

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )
