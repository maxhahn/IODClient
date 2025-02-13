from litestar import Controller, Response, MediaType, post
from litestar.exceptions import HTTPException
from typing import Optional
from fedci.server import ProxyServer
from ls_data_structures import Algorithm, ExecutionRequest, Room, RoomDetailsDTO, FEDGLMState, FEDGLMUpdateData
from ls_env import connections, rooms, user2connection
from ls_helpers import validate_user_request

from dataclasses import asdict

import fedci

from collections import OrderedDict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np

import datetime
from typing import Set

class AlgorithmController(Controller):
    path = '/run'

    # TODO: Missing ord0 rows need to be added with p value 0! (see algorithm 1 of tillman and sprites 2011)
    def run_iod_on_user_data(self, data, alpha):
        users = []

        ro.r['source']('./scripts/ci_functions.r')
        aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
        # Reading and processing data
        #df = pl.read_csv("./random-data-1.csv")
        with (ro.default_converter + pandas2ri.converter).context():
            lvs = []
            for user, df, labels in data:
                #converting it into r object for passing into r function
                d = [('citestResults', ro.conversion.get_conversion().py2rpy(df)), ('labels', ro.StrVector(labels))]
                od = OrderedDict(d)
                lv = ro.ListVector(od)
                lvs.append(lv)
                users.append(user)

            result = aggregate_ci_results_f(lvs, alpha)

            g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
            g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
            gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_List'].items()]
            gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
            return g_pag_list, g_pag_labels,  {u:r for u,r in zip(users, gi_pag_list)}, {u:l for u,l in zip(users, gi_pag_labels)}

    def run_iod_on_user_data(self, users, dfs, client_labels, alpha):
        ro.r['source']('../scripts/ci_functions.r')
        aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']

        with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
            lvs = []
            r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
            #r_dfs = ro.ListVector(r_dfs)
            label_list = [ro.StrVector(v) for v in client_labels]

            result = aggregate_ci_results_f(label_list, r_dfs, alpha)

            g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
            g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
            g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
            gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
            gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
            g_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]
        return g_pag_list, g_pag_labels, {u:r for u,r in zip(users, gi_pag_list)}, {u:l for u,l in zip(users, gi_pag_labels)}

    def run_iod_on_combined_data(self, df, labels, users, user_labels, alpha):
        ro.r['source']('./ci_functions.r')
        iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

        # let index start with 1
        df.index += 1

        user_labels = [ro.StrVector(v) for v in user_labels]

        with (ro.default_converter + pandas2ri.converter).context():
            #converting it into r object for passing into r function
            suff_stat = [
                ('citestResults', ro.conversion.get_conversion().py2rpy(df)),
                ('all_labels', ro.StrVector(labels)),
            ]
            suff_stat = OrderedDict(suff_stat)
            suff_stat = ro.ListVector(suff_stat)

            result = iod_on_ci_data_f(user_labels, suff_stat, alpha)

            g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
            g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
            g_pag_list = [np.array(pag).astype(int).tolist() for pag in g_pag_list]
            gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_list'].items()]
            gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
            g_pag_list = [np.array(pag).astype(int).tolist() for pag in gi_pag_list]

            print(g_pag_list)
            print(g_pag_labels)
            user_pags = {u:r for u,r in zip(users, gi_pag_list)}
            user_labels = {u:l for u,l in zip(users, gi_pag_labels)}
            print(user_pags)
            print(user_labels)

        return g_pag_list, g_pag_labels, user_pags, user_labels

    def run_meta_analysis_iod(self, data, room_name):
        room = rooms[room_name]

        # gather data of all participants
        participant_data = []
        participant_data_labels = []
        participants = room.users
        for user in participants:
            conn = user2connection[user]
            participant_data.append(conn.algorithm_data.data)
            participant_data_labels.append(conn.algorithm_data.data_labels)

        return self.run_iod_on_user_data(participants, participant_data, participant_data_labels, alpha=data.alpha)

    def run_fedci_iod(self, data, room_name):
        room: Room = rooms[room_name]

        alpha = data.alpha
        max_cond_size = data.max_conditioning_set

        server = ProxyServer.builder().set_max_regressors(max_cond_size)
        for username in room.users:
            client_data = user2connection[username].algorithm_data
            try:
                server.add_client(client_data.hostname, client_data.port)
            except:
                raise HTTPException(detail=f'Could not open RPC connection to {username}', status_code=404)
        server = server.build()

        test_results = server.run()

        likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(test_results)

        all_labels = sorted(list(server.schema.keys()))

        columns = ('ord', 'X', 'Y', 'S', 'pvalue')
        rows = []
        for test in likelihood_ratio_tests:
            s_labels_string = ','.join(sorted([str(all_labels.index(l)+1) for l in test.conditioning_set]))
            rows.append((len(test.conditioning_set), all_labels.index(test.v0)+1, all_labels.index(test.v1)+1, s_labels_string, test.p_val))

        df = pd.DataFrame(data=rows, columns=columns)

        participant_data_labels = []
        participants = room.users
        for user in participants:
            conn = user2connection[user]
            participant_data_labels.append(conn.algorithm_data.data_labels)


        try:
            result, result_labels, _, _ = self.run_iod_on_combined_data(df, all_labels, participants , participant_data_labels, alpha=alpha)
        except:
            raise HTTPException(detail='Failed to execute IOD', status_code=500)

        room.result = result
        room.result_labels = result_labels

        room.is_processing = False
        room.is_finished = True

        return None, None, None, None

    @post("/{room_name:str}")
    async def run(self, data: ExecutionRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
        if room_name not in rooms:
            raise HTTPException(detail='The room does not exist', status_code=404)
        room = rooms[room_name]

        # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
        if room.owner_name != data.username:
            raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)

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
            raise Exception(f'Encountered unknown algorithm {room.algorithm}')

        try:
            result, result_labels, user_result, user_labels = process_func(data, room_name)
        except:
            room = rooms[room_name]
            room.is_processing = False
            room.is_locked = True
            room.is_hidden = True
            rooms[room_name] = room
            raise HTTPException(detail='Failed to execute', status_code=500)

        room = rooms[room_name]
        for user in user_result:
            if user not in room.users:
                del user_result[user]

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
            status_code=200
            )
