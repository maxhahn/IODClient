import base64
import datetime
import pickle
import uuid
from typing import Optional

import polars as pl
from litestar import Controller, MediaType, Response, post
from litestar.exceptions import HTTPException
from ls_data_structures import (
    CheckInRequest,
    Connection,
    FedCIDataSubmissionRequest,
    FedCIUserData,
    MetaAnalysisDataSubmissionRequest,
    MetaAnalysisUserData,
    Room,
    UpdateUserRequest,
    UserDTO,
)
from ls_env import connections, user2connection
from ls_helpers import validate_user_request
from shared.env import Algorithm


class UserController(Controller):
    path = "/user"

    @post("/check-in")
    async def check_in(self, data: CheckInRequest) -> Response:
        if data.username is None or len(data.username.replace(r"\s", "")) == 0:
            raise HTTPException(detail="Username is not accepted", status_code=400)

        # guarantee unused new id
        new_id = uuid.uuid4()
        while new_id in connections:
            new_id = uuid.uuid4()
        new_id = str(new_id)

        username = data.username

        username_offset = 1
        occupied_usernames = [c.username for c in connections.values()]
        new_username = username
        while new_username in occupied_usernames:
            new_username = username + f" ({username_offset})"
            username_offset += 1

        selected_algorithm = Algorithm(data.algorithm)

        conn = Connection(
            new_id, new_username, datetime.datetime.now(), algorithm=selected_algorithm
        )
        connections[new_id] = conn
        user2connection[new_username] = conn

        return Response(
            media_type=MediaType.JSON, content=UserDTO(conn), status_code=200
        )

    @post("/update")
    async def update_user(self, data: UpdateUserRequest) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )

        # if data.username is None or len(data.username.replace(r'\s', '')) == 0:
        #    raise HTTPException(detail='Username is not accepted', status_code=400)

        if (
            data.new_username is not None
            and len(data.new_username.replace(r"\s", "")) > 0
        ):
            username_offset = 1
            occupied_usernames = [c.username for c in connections.values()]
            new_username = data.new_username
            while new_username in occupied_usernames:
                new_username = data.new_username + f" ({username_offset})"
                username_offset += 1

            connections[data.id].username = new_username

        connections[data.id].algorithm = Algorithm(data.algorithm)
        conn = connections[data.id]

        return Response(
            media_type=MediaType.JSON, content=UserDTO(conn), status_code=200
        )

    @post("/submit-data")
    async def receive_meta_analysis_data(
        self, data: MetaAnalysisDataSubmissionRequest
    ) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )

        if connections[data.id].algorithm != Algorithm.META_ANALYSIS:
            raise HTTPException(
                detail=f"Incompatible algorithm {connections[data.id].algorithm} for data submission",
                status_code=403,
            )

        # data to pandas conversion
        df = pickle.loads(base64.b64decode(data.data.encode()))
        schema = {k: str(v) for k, v in dict(pl.from_pandas(df).schema).items()}
        connections[data.id].algorithm_data = MetaAnalysisUserData(
            data_schema=schema, data=df
        )

        return Response(
            media_type=MediaType.TEXT, content="Data received", status_code=200
        )

    @post("/submit-rpc-info")
    async def receive_fedci_data(self, data: FedCIDataSubmissionRequest) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )

        if connections[data.id].algorithm != Algorithm.FEDCI:
            raise HTTPException(
                detail=f"Incompatible algorithm {connections[data.id].algorithm} for RPC connection data",
                status_code=403,
            )

        # data to pandas conversion
        connections[data.id].algorithm_data = FedCIUserData(
            data_schema=data.data_schema, hostname=data.hostname, port=data.port
        )

        return Response(
            media_type=MediaType.TEXT, content="Data received", status_code=200
        )
