import base64
import pickle
from typing import Optional

from litestar import MediaType, Response, get, post
from litestar.controller import Controller
from litestar.exceptions import HTTPException
from litestar_app.ls_runner import AlgorithmController
from ls_data_structures import (
    BasicRequest,
    JoinRoomRequest,
    Room,
    RoomCreationRequest,
    RoomDetailsDTO,
    RoomDTO,
)
from ls_env import connections, rooms, user2connection, user2room
from ls_helpers import validate_user_request
from shared.env import Algorithm


class RoomController(Controller):
    path = "/rooms"

    @post("")
    async def get_rooms(self, data: BasicRequest) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )

        return Response(
            media_type=MediaType.JSON,
            content=[
                RoomDTO(room)
                for room in rooms.values()
                if not room.is_hidden
                and room.algorithm == user2connection[data.username].algorithm
            ],
            status_code=200,
        )

    # get room info
    @post("/{room_name:str}")
    async def get_room(self, data: BasicRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)

        room = rooms[room_name]

        if data.username not in room.users:
            raise HTTPException(detail="You are not in this room", status_code=403)

        # hacky fix for R that is not being executed in main thread
        if room.test_results is not None and room.result is None:
            AlgorithmController.run_fedci_iod_on_test_results(
                room_name, room.test_results, 0.05
            )

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )

    # create room
    @post("/create")
    async def create_room(self, data: RoomCreationRequest) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if len(data.room_name) == 0:
            raise HTTPException(
                detail="The room must have a name with at least 1 character",
                status_code=400,
            )

        room_owner = data.username
        room_name = data.room_name

        room_name_offset = 1
        new_room_name = room_name
        occupied_room_names = [r.name for r in rooms.values()]
        while new_room_name in occupied_room_names:
            new_room_name = room_name + f" ({room_name_offset})"
            room_name_offset += 1

        room = Room(
            name=new_room_name,
            algorithm=Algorithm(data.algorithm),
            owner_name=room_owner,
            password=data.password,
            is_locked=False,
            is_hidden=False,
            is_processing=False,
            is_finished=False,
            users={room_owner},
            user_provided_schema={
                room_owner: connections[data.id].algorithm_data.data_schema
            },
            result=None,
            result_labels=None,
            user_results={},
            user_labels={},
            state_msg="",
        )

        rooms[new_room_name] = room
        user2room[room_owner] = room

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )

    # join room
    @post("/{room_name:str}/join")
    async def join_room(self, data: JoinRoomRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if data.username in user2room:
            raise HTTPException(
                detail="You are already assigned a room", status_code=403
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)

        room = rooms[room_name]

        if connections[data.id].algorithm_data is None:
            if connections[data.id].algorithm == Algorithm.META_ANALYSIS:
                raise HTTPException(
                    detail="You have to upload data before joining a room",
                    status_code=403,
                )
            else:
                raise HTTPException(
                    detail="You have to provide RPC details before joining a room",
                    status_code=403,
                )

        if room.is_locked:
            raise HTTPException(detail="The room is locked", status_code=403)
        if room.password != data.password:
            raise HTTPException(detail="Incorrect password", status_code=403)

        room.users.add(data.username)
        room.user_provided_schema[data.username] = connections[
            data.id
        ].algorithm_data.data_schema
        user2room[data.username] = room

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )

    # leave room
    @post("/{room_name:str}/leave")
    async def leave_room(self, data: BasicRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)

        room = rooms[room_name]

        if data.username not in room.users:
            raise HTTPException(detail="You are not in the room", status_code=403)

        room.users.remove(data.username)
        del user2room[data.username]
        # just remove room if it is empty
        if len(room.users) == 0:
            del rooms[room_name]
            return Response(
                media_type=MediaType.TEXT, content="You left the room", status_code=200
            )

        del room.user_provided_schema[data.username]

        if room.is_finished:
            del room.user_results[data.username]

        if data.username == room.owner_name:
            room.owner_name = list(room.users)[0]

        rooms[room_name] = room  # reassign

        return Response(
            media_type=MediaType.TEXT, content="You left the room", status_code=200
        )

    # kick user from room
    @post("/{room_name:str}/kick/{username_to_kick:str}")
    async def kick_user_from_room(
        self, data: BasicRequest, room_name: str, username_to_kick: str
    ) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)

        room = rooms[room_name]
        if room.is_finished:
            raise HTTPException(
                detail="Cannot kick users once IOD was executed", status_code=403
            )
        if room.is_processing:
            raise HTTPException(
                detail="Cannot kick while server processes the rooms data",
                status_code=403,
            )
        # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
        if room.owner_name != data.username:
            raise HTTPException(
                detail="You do not have sufficient authority in this room",
                status_code=403,
            )
        if room.owner_name == username_to_kick:
            raise HTTPException(
                detail="Cannot kick the owner of the room", status_code=403
            )

        if username_to_kick not in room.users:
            raise HTTPException(
                detail="The person attempted to kick is not inside the room",
                status_code=403,
            )

        room.users.remove(username_to_kick)

        del room.user_provided_schema[username_to_kick]

        if room.is_finished:
            del room.user_results[username_to_kick]
        del user2room[username_to_kick]
        rooms[room_name] = room

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )

    def change_room_lock_state(
        self, data: BasicRequest, room_name: str, new_lock_state: bool
    ):
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)
        room = rooms[room_name]

        if room.is_finished:
            raise HTTPException(
                detail="The room can no longer be changed", status_code=403
            )

        # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
        if room.owner_name != data.username:
            raise HTTPException(
                detail="You do not have sufficient authority in this room",
                status_code=403,
            )

        room.is_locked = new_lock_state
        rooms[room_name] = room

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )

    # lock room
    @post("/{room_name:str}/lock")
    async def lock_room(self, data: BasicRequest, room_name: str) -> Response:
        return self.change_room_lock_state(data, room_name, True)

    # lock room
    @post("/{room_name:str}/unlock")
    async def unlock_room(self, data: BasicRequest, room_name: str) -> Response:
        return self.change_room_lock_state(data, room_name, False)

    def change_room_hidden_state(
        self, data: BasicRequest, room_name: str, new_hidden_state: bool
    ):
        if not validate_user_request(data.id, data.username):
            raise HTTPException(
                detail="The provided identification is not recognized by the server",
                status_code=401,
            )
        if room_name not in rooms:
            raise HTTPException(detail="The room does not exist", status_code=404)

        room = rooms[room_name]

        if room.is_finished:
            raise HTTPException(
                detail="The room can no longer be changed", status_code=403
            )

        # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
        if room.owner_name != data.username:
            raise HTTPException(
                detail="You do not have sufficient authority in this room",
                status_code=403,
            )

        room.is_hidden = new_hidden_state
        rooms[room_name] = room

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200,
        )

    # hide room
    @post("/{room_name:str}/hide")
    async def hide_room(self, data: BasicRequest, room_name: str) -> Response:
        return self.change_room_hidden_state(data, room_name, True)

    # reveal room
    @post("/{room_name:str}/reveal")
    async def reveal_room(self, data: BasicRequest, room_name: str) -> Response:
        return self.change_room_hidden_state(data, room_name, False)
