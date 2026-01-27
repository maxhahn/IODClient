from litestar import Litestar, get, Response, MediaType
from typing import Optional
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ls_room import RoomController
from ls_runner import AlgorithmController
from ls_user import UserController
from ls_helpers import cleanse_inactive_users
from ls_env import last_cleanse_time
import datetime

@get("/health-check")
async def health_check() -> Response:
    global last_cleanse_time
    curr_time = datetime.datetime.now()
    if (curr_time - last_cleanse_time).total_seconds() > 60*20:
        cleanse_inactive_users(curr_time)
        last_cleanse_time = curr_time

    return Response(
        media_type=MediaType.TEXT,
        content='Hello there!',
        status_code=200
    )


app = Litestar([
    health_check,
    UserController,
    RoomController,
    AlgorithmController
])
