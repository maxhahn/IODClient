import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from ls_helpers import deserialize_numpy_array, serialize_numpy_array
from shared.env import Algorithm

import fedci


@dataclass
class UserData:
    data_schema: Dict[str, str]


@dataclass
class MetaAnalysisUserData(UserData):
    data: pd.DataFrame


@dataclass
class FedCIUserData(UserData):
    hostname: str
    port: int


@dataclass
class Connection:
    id: str
    username: str
    last_request_time: datetime.datetime
    algorithm: Algorithm
    algorithm_data: UserData | None = None


@dataclass
class FEDGLMUpdateDataDTO:
    xwx: Dict[str, object]
    xwz: Dict[str, object]
    dev: float
    llf: float


@dataclass
class FEDGLMUpdateData:
    xwx: Dict[str, np.typing.NDArray]
    xwz: Dict[str, np.typing.NDArray]
    dev: float
    llf: float

    def __init__(self, data: FEDGLMUpdateDataDTO):
        self.xwx = {c: deserialize_numpy_array(xwx) for c, xwx in data.xwx.items()}
        self.xwz = {c: deserialize_numpy_array(xwz) for c, xwz in data.xwz.items()}
        self.dev = data.dev
        self.llf = data.llf


@dataclass
class FEDGLMState:
    user_provided_schema: Dict[str, Dict[str, fedci.VariableType]]
    user_provided_labels: Dict[str, List[str]]
    user_provided_categorical_expressions: Dict[str, Dict[str, List[str]]]
    user_provided_ordinal_expressions: Dict[str, Dict[str, List[str]]]
    testing_engine: object  # fedci.TestEngine
    pending_data: Dict[str, FEDGLMUpdateData]
    start_of_last_iteration: datetime.datetime


@dataclass
class RIODState:
    user_provided_labels: Dict[str, List[str]]


@dataclass
class Room:
    name: str
    algorithm: Algorithm
    owner_name: str
    password: str | None
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    users: Set[str]
    user_provided_schema: Dict[str, Dict[str, str]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    user_results: Dict[str, List[List[int]]]
    user_labels: Dict[str, List[str]]


@dataclass
class UserDTO:
    id: str
    username: str
    algorithm: str
    data_schema: List[str]

    def __init__(self, conn: Connection):
        self.id = conn.id
        self.username = conn.username
        self.algorithm = conn.algorithm.value


@dataclass
class RoomDTO:
    name: str
    algorithm: Algorithm
    owner_name: str
    is_locked: bool
    is_protected: bool

    def __init__(self, room: Room):
        self.name = room.name
        self.algorithm = room.algorithm
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_protected = room.password is not None


@dataclass
class RoomDetailsDTO:
    name: str
    algorithm: Algorithm
    owner_name: str
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    is_protected: bool
    users: List[str]
    user_provided_schema: Dict[str, Dict[str, str]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    private_result: List[List[int]]
    private_labels: List[str]

    def __init__(self, room: Room, requesting_user: Union[str, None] = None):
        self.name = room.name
        self.algorithm = room.algorithm
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_hidden = room.is_hidden
        self.is_processing = room.is_processing
        self.is_finished = room.is_finished
        self.is_protected = room.password is not None
        self.users = sorted(list(room.users))
        self.user_provided_schema = room.user_provided_schema
        self.result = room.result
        self.result_labels = room.result_labels
        self.private_result = (
            room.user_results[requesting_user]
            if room.user_results is not None and requesting_user in room.user_results
            else None
        )
        self.private_labels = (
            room.user_labels[requesting_user]
            if room.user_labels is not None and requesting_user in room.user_labels
            else None
        )


@dataclass
class CheckInRequest:
    username: str
    algorithm: str


@dataclass
class BasicRequest:
    id: str
    username: str


@dataclass
class UpdateUserRequest(BasicRequest):
    algorithm: str
    new_username: str


@dataclass
class RoomCreationRequest(BasicRequest):
    room_name: str
    algorithm: str
    password: str | None


@dataclass
class JoinRoomRequest(BasicRequest):
    password: str | None


@dataclass
class MetaAnalysisDataSubmissionRequest(BasicRequest):
    data_schema: Dict[str, str]
    data: str


@dataclass
class FedCIDataSubmissionRequest(BasicRequest):
    data_schema: Dict[str, str]
    hostname: str
    port: int


@dataclass
class ExecutionRequest(BasicRequest):
    alpha: float
    max_conditioning_set: int | None
