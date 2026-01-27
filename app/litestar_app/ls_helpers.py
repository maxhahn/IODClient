from ls_env import connections, rooms, user2room, user2connection
import datetime
import json
import numpy as np
import base64
from typing import Optional

def validate_user_request(id: str, username: str):
    if id not in connections:
        return False

    if connections[id].username != username:
        return False

    connections[id].last_request_time = datetime.datetime.now()
    return True

def cleanse_inactive_users(curr_time):
    for id, conn in connections.items():
        if (curr_time-conn.last_request_time).total_seconds() > 60*60*3:
            username = conn.username
            # if user is in room
            if username in user2room:
                # remove from room
                user2room[username].users.remove(username)
                # if room is empty -> remove room
                if len(user2room[username].users) == 0:
                    del rooms[user2room[username].name]
                del user2room[username]
            # remove connection lookup and connection
            del user2connection[username]
            del connections[id]

def deserialize_numpy_array(serialized_data):
    data = json.loads(serialized_data)
    arr_base64 = data['data']
    arr_bytes = base64.b64decode(arr_base64)
    arr = np.frombuffer(arr_bytes, dtype=data['dtype']).reshape(data['shape'])
    return arr

def serialize_numpy_array(arr):
    # Convert NumPy array to bytes
    arr_bytes = arr.tobytes()
    # Encode bytes to base64 string
    arr_base64 = base64.b64encode(arr_bytes).decode('utf-8')
    # Create a JSON-serializable dictionary
    data = {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'data': arr_base64
    }
    return json.dumps(data)
