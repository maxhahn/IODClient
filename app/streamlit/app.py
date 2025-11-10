import base64
import datetime
import io
import os
import pickle
import shutil
import sys
import threading
import zipfile
from collections import OrderedDict

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import extra_streamlit_components as stx
import graphviz
import pandas as pd
import polars as pl
import requests
import rpy2.robjects as ro
import streamlit as st
from rpy2.robjects import pandas2ri
from shared import env
from streamlit_autorefresh import st_autorefresh
from streamlit_extras.dataframe_explorer import dataframe_explorer

import fedci

# TODO: Make alpha in IOD configurable (server)
# TODO: Make m.max configurable

# launch command
# streamlit run app.py --server.enableXsrfProtection false

client_base_dir = "./IOD/client-data"
upload_dir = f"{client_base_dir}/uploaded_files"
ci_result_dir = f"{client_base_dir}/ci"

# Init upload files dir
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir, exist_ok=True)
if not os.path.exists(ci_result_dir):
    os.makedirs(ci_result_dir, exist_ok=True)

os.environ["OMP_NUM_THREADS"] = "1"

#  ,---.   ,--.            ,--.             ,--.        ,--.  ,--.
# '   .-',-'  '-. ,--,--.,-'  '-. ,---.     |  |,--,--, `--',-'  '-.
# `.  `-.'-.  .-'' ,-.  |'-.  .-'| .-. :    |  ||      \,--.'-.  .-'
# .-'    | |  |  \ '-'  |  |  |  \   --.    |  ||  ||  ||  |  |  |
# `-----'  `--'   `--`--'  `--'   `----'    `--'`--''--'`--'  `--'

if "username" not in st.session_state:
    st.session_state["username"] = None
if "server_url" not in st.session_state:
    if "LITESTAR_CONTAINER_NAME" in os.environ and "LITESTAR_PORT" in os.environ:
        st.session_state["server_url"] = (
            f"http://{os.environ['LITESTAR_CONTAINER_NAME']}:{os.environ['LITESTAR_PORT']}"  # TODO: load default url from config file
        )
    else:
        st.session_state["server_url"] = "http://127.0.0.1:8000"
if "last_health_check" not in st.session_state:
    st.session_state["last_health_check"] = None
if "is_connected_to_server" not in st.session_state:
    st.session_state["is_connected_to_server"] = None

if "server_provided_user_id" not in st.session_state:
    st.session_state["server_provided_user_id"] = None
if "current_room" not in st.session_state:
    st.session_state["current_room"] = None

if "_alpha_value" not in st.session_state:
    st.session_state["_alpha_value"] = 0.05
if "_local_alpha_value" not in st.session_state:
    st.session_state["_local_alpha_value"] = 0.05
if "_max_conditioning_set_cardinality" not in st.session_state:
    st.session_state["_max_conditioning_set_cardinality"] = 1

if "max_conditioning_set" not in st.session_state:
    st.session_state["max_conditioning_set"] = 1
if "_max_conditioning_set" not in st.session_state:
    st.session_state["_max_conditioning_set"] = 1

if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None
if "uploaded_data_filename" not in st.session_state:
    st.session_state["uploaded_data_filename"] = None
if "result_pvals" not in st.session_state:
    st.session_state["result_pvals"] = None
if "result_labels" not in st.session_state:
    st.session_state["result_labels"] = None
if "server_has_received_data" not in st.session_state:
    st.session_state["server_has_received_data"] = False

if "do_autorefresh" not in st.session_state:
    st.session_state["do_autorefresh"] = True

if "fedci_client" not in st.session_state:
    st.session_state["fedci_client"] = None
if "fedci_client_port" not in st.session_state:
    st.session_state["fedci_client_port"] = None
if "fedci_client_thread" not in st.session_state:
    st.session_state["fedci_client_thread"] = None
if "selected_algorithm" not in st.session_state:
    st.session_state["selected_algorithm"] = None

# Always update
st.session_state["existing_raw_data"] = os.listdir(upload_dir)

# ,--.  ,--.,--------.,--------.,------.     ,------.                ,--.      ,--.  ,--.       ,--.
# |  '--'  |'--.  .--''--.  .--'|  .--. '    |  .--. ' ,---.  ,---.,-'  '-.    |  '--'  | ,---. |  | ,---.  ,---. ,--.--.
# |  .--.  |   |  |      |  |   |  '--' |    |  '--' || .-. |(  .-''-.  .-'    |  .--.  || .-. :|  || .-. || .-. :|  .--'
# |  |  |  |   |  |      |  |   |  | --'     |  | --' ' '-' '.-'  `) |  |      |  |  |  |\   --.|  || '-' '\   --.|  |
# `--'  `--'   `--'      `--'   `--'         `--'      `---' `----'  `--'      `--'  `--' `----'`--'|  |-'  `----'`--'
#                                                                                                   `--'


def post_to_server(url, payload):
    r = requests.post(url=url, json=payload)
    try:
        # r = requests.post(url=url, json=payload)
        return r
    except:
        st.session_state["last_health_check"] = None
        st.error("There are problems with the server connection")
    return None


#  ,---.                                           ,--.  ,--.               ,--.  ,--.  ,--.             ,-----.,--.                  ,--.
# '   .-'  ,---. ,--.--.,--.  ,--.,---. ,--.--.    |  '--'  | ,---.  ,--,--.|  |,-'  '-.|  ,---. ,-----.'  .--./|  ,---.  ,---.  ,---.|  |,-.
# `.  `-. | .-. :|  .--' \  `'  /| .-. :|  .--'    |  .--.  || .-. :' ,-.  ||  |'-.  .-'|  .-.  |'-----'|  |    |  .-.  || .-. :| .--'|     /
# .-'    |\   --.|  |     \    / \   --.|  |       |  |  |  |\   --.\ '-'  ||  |  |  |  |  | |  |       '  '--'\|  | |  |\   --.\ `--.|  \  \
# `-----'  `----'`--'      `--'   `----'`--'       `--'  `--' `----' `--`--'`--'  `--'  `--' `--'        `-----'`--' `--' `----' `---'`--'`--'


def check_server_connection():
    curr_time = datetime.datetime.now()
    if (
        st.session_state["last_health_check"] is not None
        and (curr_time - st.session_state["last_health_check"]).total_seconds() < 60 * 5
    ):
        return True
    try:
        with st.spinner("Checking connection to server..."):
            r = requests.get(url=f"{st.session_state['server_url']}/health-check")

        if r.status_code != 200:
            st.error("There are problems with the server connection")
            st.session_state["is_connected_to_server"] = False
            return False
    except:
        st.error("There are problems with the server connection")
        st.session_state["is_connected_to_server"] = False
        return False

    st.session_state["last_health_check"] = curr_time
    st.session_state["is_connected_to_server"] = True
    return True


#  ,---.                                            ,-----.,--.                  ,--.          ,--.
# '   .-'  ,---. ,--.--.,--.  ,--.,---. ,--.--.    '  .--./|  ,---.  ,---.  ,---.|  |,-.,-----.|  |,--,--,
# `.  `-. | .-. :|  .--' \  `'  /| .-. :|  .--'    |  |    |  .-.  || .-. :| .--'|     /'-----'|  ||      \
# .-'    |\   --.|  |     \    / \   --.|  |       '  '--'\|  | |  |\   --.\ `--.|  \  \       |  ||  ||  |
# `-----'  `----'`--'      `--'   `----'`--'        `-----'`--' `--' `----' `---'`--'`--'      `--'`--''--'


def step_check_in_to_server():
    if st.session_state["server_provided_user_id"] is not None:
        st.info("Server check-in completed")

    col1, col2 = st.columns((6, 1))
    col1.write("Please enter the server URL:")
    server_url = col1.text_input(
        "Please chose your username",
        placeholder=st.session_state["server_url"],
        label_visibility="collapsed",
    )

    col2.write("Connect!")
    if (
        col2.button(":link:", help="Connect to URL", use_container_width=True)
        and server_url is not None
        and server_url != ""
    ):
        if st.session_state["username"] is not None:
            st.warning("""You are already checked in with a server.
                       In order to leave the server, refresh this page.""")
        else:
            if server_url.endswith("/"):
                server_url = server_url[:-1]
            st.session_state["server_url"] = server_url
            st.session_state["last_health_check"] = None
            st.rerun()
            return

    st.write("---")

    container = st.container()

    col1, col2, col3 = st.columns((4, 2, 1))
    col1.write("Please enter a username:")
    username = col1.text_input(
        "Please chose your username",
        placeholder=st.session_state["username"],
        label_visibility="collapsed",
    )

    col2.write("Select algorithm")
    # TODO: ensure selected algorithm is default select
    algo_type = col2.selectbox(
        "Select algorithm",
        [e.name for e in env.Algorithm],
        label_visibility="collapsed",
    )

    if algo_type == env.Algorithm.FEDCI.name:
        container.warning(
            "When using fedCI, a port on your network will be used for RPC communication"
        )

    # TODO: add description

    col3.write("Submit!")
    if st.session_state["server_provided_user_id"] is None:
        button_text = "Submit check-in request"
        request_url = f"{st.session_state['server_url']}/user/check-in"

        if algo_type == env.Algorithm.FEDCI.name:
            categorical_expressions, ordinal_expressions, schema = None, None, None
        else:
            fedci_client = st.session_state["fedci_client"]
            categorical_expressions = fedci_client.get_categorical_expressions()
            ordinal_expressions = fedci_client.get_ordinal_expressions()
            schema = fedci_client.get_data_schema()

        request_params = {
            "username": username,
            "algorithm": algo_type,
            "data_labels": st.session_state["result_labels"],
            "schema": schema,
            "categorical_expressions": categorical_expressions,
            "ordinal_expressions": ordinal_expressions,
        }
    else:
        button_text = "Update user"
        request_url = f"{st.session_state['server_url']}/user/update"
        request_params = {
            "id": st.session_state["server_provided_user_id"],
            "algorithm": algo_type,
            "username": st.session_state["username"],
            "new_username": username,
        }

    button = col3.button(
        ":arrow_heading_up:", help=button_text, use_container_width=True
    )
    if button:
        r = post_to_server(request_url, request_params)
        if r is None:
            return
        if r.status_code != 200:
            st.error("Failed to check in with the server. Please reload the page")
            return

        r = r.json()
        st.session_state["server_provided_user_id"] = r["id"]
        st.session_state["username"] = r["username"]
        st.session_state["selected_algorithm"] = r["algorithm"]
        st.rerun()
    return


# ,------.            ,--.              ,--. ,--.       ,--.                  ,--.
# |  .-.  \  ,--,--.,-'  '-. ,--,--.    |  | |  | ,---. |  | ,---.  ,--,--. ,-|  |
# |  |  \  :' ,-.  |'-.  .-'' ,-.  |    |  | |  || .-. ||  || .-. |' ,-.  |' .-. |
# |  '--'  /\ '-'  |  |  |  \ '-'  |    '  '-'  '| '-' '|  |' '-' '\ '-'  |\ `-' |
# `-------'  `--`--'  `--'   `--`--'     `-----' |  |-' `--' `---'  `--`--' `---'
#                                                `--'


def step_upload_data():
    uploaded_file = st.file_uploader("Data Upload", type=["csv", "parquet"])

    no_files_uploaded_yet = len(st.session_state["existing_raw_data"]) == 0
    st.write("Select previously uploaded data:")
    col1, col2 = st.columns((6, 1))
    previously_uploaded_file = col1.selectbox(
        "Select previously uploaded data:",
        st.session_state["existing_raw_data"],
        index=st.session_state["existing_raw_data"].index(
            st.session_state["uploaded_data_filename"]
        )
        if st.session_state["uploaded_data_filename"] is not None
        else None,
        disabled=no_files_uploaded_yet,
        label_visibility="collapsed",
        help="No files uploaded so far"
        if no_files_uploaded_yet
        else "Please select the file you want to work with",
    )

    if col2.button(
        ":wastebasket:",
        help="Remove all local data. THIS INCLUDES PROCESSED DATA AND PAGs.",
        use_container_width=True,
    ):
        shutil.rmtree(client_base_dir)
        st.rerun()
        return

    if (
        uploaded_file is None
        and previously_uploaded_file is None
        and st.session_state["uploaded_data"] is None
    ):
        return

    # read uploaded file into session state
    if uploaded_file is not None:
        filename = uploaded_file.name
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            filename = filename[:-4] + ".parquet"
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            raise Exception("Cannot handle files of the given type")
        df.to_parquet(f"{upload_dir}/{filename}", index=False)

        st.session_state["uploaded_data"] = df
        st.session_state["uploaded_data_filename"] = filename
    elif previously_uploaded_file is not None:
        filename = previously_uploaded_file
        df = pd.read_parquet(f"{upload_dir}/{previously_uploaded_file}")

        st.session_state["uploaded_data"] = df
        st.session_state["uploaded_data_filename"] = filename

    # Reset when new file is uploaded
    st.session_state["result_labels"] = list(st.session_state["uploaded_data"].columns)
    st.session_state["result_pvals"] = None

    if st.session_state["fedci_client"] is not None:
        st.session_state["fedci_client"].close()
        del st.session_state["fedci_client"]
        st.session_state["fedci_client"] = None
    if st.session_state["fedci_client_thread"] is not None:
        del st.session_state["fedci_client_thread"]
        st.session_state["fedci_client_thread"] = None

    client_port = 16016
    client = fedci.ProxyClient(pl.from_pandas(st.session_state["uploaded_data"]))
    client_thread = threading.Thread(
        target=client.start, args=(client_port,), daemon=True
    )
    client_thread.start()

    st.session_state["fedci_client"] = client
    st.session_state["fedci_client_thread"] = client_thread
    st.session_state["fedci_client_port"] = client_port

    if st.session_state["uploaded_data"] is not None:
        with st.expander("View Data"):
            df = st.session_state["uploaded_data"]
            st.dataframe(dataframe_explorer(df), use_container_width=True)


# ,------.            ,--.              ,------.                                          ,--.
# |  .-.  \  ,--,--.,-'  '-. ,--,--.    |  .--. ',--.--. ,---.  ,---. ,---.  ,---.  ,---. `--',--,--,  ,---.
# |  |  \  :' ,-.  |'-.  .-'' ,-.  |    |  '--' ||  .--'| .-. || .--'| .-. :(  .-' (  .-' ,--.|      \| .-. |
# |  '--'  /\ '-'  |  |  |  \ '-'  |    |  | --' |  |   ' '-' '\ `--.\   --..-'  `).-'  `)|  ||  ||  |' '-' '
# `-------'  `--`--'  `--'   `--`--'    `--'     `--'    `---'  `---' `----'`----' `----' `--'`--''--'.`-  /
#                                                                                                     `---'


def iod_r_call(dfs, client_labels, alpha=0.05, procedure="original"):
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r["source"]("./scripts/iod.r")
        aggregate_ci_results_f = ro.globalenv["aggregate_ci_results"]

        lvs = []
        r_dfs = [ro.conversion.get_conversion().py2rpy(df) for df in dfs]
        label_list = [ro.StrVector(v) for v in client_labels]

        result = aggregate_ci_results_f(label_list, r_dfs, alpha, procedure)

        g_pag_list = [x[1].tolist() for x in result["G_PAG_List"].items()]
        g_pag_labels = [
            list([str(a) for a in x[1]]) for x in result["G_PAG_Label_List"].items()
        ]

    return g_pag_list, g_pag_labels


def run_local_fci():
    df = st.session_state["result_pvals"]
    labels = st.session_state["result_labels"]
    alpha = st.session_state["local_alpha_value"]

    pag, pag_labels = iod_r_call([df], [labels], alpha)
    pag = pag[0]
    pag_labels = pag_labels[0]

    if any([a != b for a, b in zip(pag_labels, st.session_state["result_labels"])]):
        st.error("""A major issue with the data labels occured.
                    Please reload the page.
                    If this error persists, contact an administrator.
                    """)

    return pag


def step_process_data():
    _, col1, col2, _, col3, _ = st.columns((1, 3, 3, 1, 6, 1))

    fileid = os.path.splitext(st.session_state["uploaded_data_filename"])[0]
    filename = f"citestResults_{st.session_state['uploaded_data_filename']}"

    # c1, c2 = st.columns((1,1))
    if col2.button(
        ":wastebasket:",
        help="Delete current progress, so that data can be reprocessed from scratch",
        disabled=not os.path.exists(f"{ci_result_dir}/{filename}"),
        use_container_width=True,
    ):
        os.remove(f"{ci_result_dir}/{filename}")
        st.session_state["result_pvals"] = None
        st.rerun()
        return

    if col1.button("Process Data!", use_container_width=True):
        max_conditioning_set_cardinality = st.session_state[
            "max_conditioning_set_cardinality"
        ]

        with st.spinner("Data is being processed..."):
            # Read data from state
            df = st.session_state["uploaded_data"]

            # Call R function
            with (ro.default_converter + pandas2ri.converter).context():
                # load local-ci script
                ro.r["source"]("./scripts/local-ci.r")
                # load function from R script
                run_ci_test_f = ro.globalenv["run_ci_test"]

                # converting it into r object for passing into r function
                df_r = ro.conversion.get_conversion().py2rpy(df)
                # Invoking the R function and getting the result
                result = run_ci_test_f(
                    df_r, max_conditioning_set_cardinality, ci_result_dir + "/", fileid
                )
                # Converting it back to a pandas dataframe.
                df_pvals = ro.conversion.get_conversion().rpy2py(
                    result["citestResults"]
                )
                labels = list(result["labels"])

                if any(
                    [a != b for a, b in zip(labels, st.session_state["result_labels"])]
                ):
                    st.error("""A major issue with the data labels occured.
                                Please reload the page.
                                If this error persists, contact an administrator.
                                """)

        st.session_state["result_pvals"] = df_pvals

        st.rerun()

    if col3.button(
        "Submit Data!",
        help="Submit Data to Server",
        disabled=st.session_state["result_pvals"] is None,
        use_container_width=True,
    ):
        df_pvals = st.session_state["result_pvals"]
        labels = st.session_state["result_labels"]
        base64_df = base64.b64encode(pickle.dumps(df_pvals)).decode("utf-8")
        # send data and labels
        r = post_to_server(
            url=f"{st.session_state['server_url']}/user/submit-data",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
                "data": base64_df,
                "data_labels": labels,
            },
        )
        if r is None:
            return
        if r.status_code != 200:
            st.error("An error occured when submitting the data")
            return
        st.session_state["server_has_received_data"] = True

    # TODO: Add 2nd col to select CI Test of choice
    def change_cond_set_value():
        st.session_state["_max_conditioning_set_cardinality"] = st.session_state[
            "max_conditioning_set_cardinality"
        ]

    _, col1, _ = st.columns((1, 6, 1))
    col1.number_input(
        "Select the maximum conditiong set size:",
        value=st.session_state["_max_conditioning_set_cardinality"],
        min_value=0,
        step=1,
        key="max_conditioning_set_cardinality",
        on_change=change_cond_set_value,
    )

    if st.session_state["server_has_received_data"] == True:
        st.info("The server has received your data")

    if (
        st.session_state["result_pvals"] is not None
        and st.session_state["server_has_received_data"] == False
    ):
        st.warning("""Any submitted data can be accessed by the server.
                                 Any participants in the same room will be able to access the data labels.
                                 Once you join a room, your data can be used in the processing rooms data.
                                 Be sure no sensitive data is submitted!""")

    if st.session_state["result_pvals"] is not None:
        tab1, tab2 = st.tabs(["Processed Data", "Generated PAG"])

        df_pvals = st.session_state["result_pvals"].copy()
        labels = st.session_state["result_labels"]

        with tab1:
            df_pvals["X"] = df_pvals["X"].apply(lambda x: labels[int(x) - 1])
            df_pvals["Y"] = df_pvals["Y"].apply(lambda x: labels[int(x) - 1])
            df_pvals["S"] = df_pvals["S"].apply(
                lambda x: ",".join(
                    sorted([labels[int(xi) - 1] for xi in x.split(",") if xi != ""])
                )
            )
            st.dataframe(dataframe_explorer(df_pvals), use_container_width=True)
        with tab2:

            def change_local_alpha_value():
                st.session_state["_local_alpha_value"] = st.session_state[
                    "local_alpha_value"
                ]

            st.number_input(
                "Select alpha value:",
                value=st.session_state["_local_alpha_value"],
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="local_alpha_value",
                format="%.2f",
                on_change=change_local_alpha_value,
            )

            with st.spinner("Preparing PAG..."):
                pag_mat = run_local_fci()
                pag = data2graph(pag_mat, st.session_state["result_labels"])

            _, col1, _ = st.columns((1, 1, 1))
            col1.download_button(
                label="Download PAG",
                data=pag.pipe(format="png"),
                file_name=f"local-pag-{fileid}.png",
                mime="image/png",
                use_container_width=True,
            )
            col1.graphviz_chart(pag)

    return


# ,------.                             ,--.          ,--.   ,--.
# |  .--. ' ,---.  ,---. ,--,--,--.    |  |    ,---. |  |-. |  |-.,--. ,--.
# |  '--'.'| .-. || .-. ||        |    |  |   | .-. || .-. '| .-. '\  '  /
# |  |\  \ ' '-' '' '-' '|  |  |  |    |  '--.' '-' '| `-' || `-' | \   '
# `--' '--' `---'  `---' `--`--`--'    `-----' `---'  `---'  `---'.-'  /
#                                                                 `---'


@st.dialog("Secure your room!")
def room_creation_password_dialog(room_name):
    st.session_state["do_autorefresh"] = False
    st.write(f"### Creating room {room_name}...")
    st.write("""You may give your room a password.
             The password will be stored on the server in plain text, so beware!
             """)
    password = st.text_input(
        "Please enter your password here. Leave empty for no password."
    )
    _, col1 = st.columns((6, 1))
    if col1.button(
        ":arrow_right:", help="Continue with chosen password", use_container_width=True
    ):
        if len(password) == 0:
            password = None

        r = post_to_server(
            url=f"{st.session_state['server_url']}/rooms/create",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
                "room_name": room_name,
                "algorithm": st.session_state["selected_algorithm"],
                "password": password,
            },
        )

        if r is None:
            return
        if r.status_code == 200:
            st.session_state["do_autorefresh"] = True
            st.session_state["current_room"] = r.json()
            st.rerun()
            return
        st.error("An error occured trying to create the room")
    return


@st.dialog("This room is secured!")
def room_join_password_dialog(room_name):
    st.session_state["do_autorefresh"] = False
    st.write(f"### Joining room {room_name}...")
    st.write("""This room is protected by a password.""")
    password = st.text_input("Please enter the password:")
    _, col1 = st.columns((6, 1))
    if col1.button(
        ":arrow_right:", help="Continue with chosen password", use_container_width=True
    ):
        if len(password) == 0:
            password = None

        r = post_to_server(
            url=f"{st.session_state['server_url']}/rooms/{room_name}/join",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
                "password": password,
            },
        )

        if r is None:
            return
        if r.status_code == 200:
            st.session_state["do_autorefresh"] = True
            st.session_state["current_room"] = r.json()
            st.rerun()
            return
        st.error("""An error occured trying to join the room.
                 The password might be incorrect.""")
        # st.toast('Yay')
        # st.rerun()
    return


def step_join_rooms():
    # enter new room name and join or create it
    info_placeholder = st.empty()
    st.write("Please enter a room name:")
    col1, col2, col3, col4 = st.columns((7, 1, 1, 1))

    room_name = col1.text_input(
        "Please chose a room name", label_visibility="collapsed"
    )
    if room_name is None or len(room_name) == 0:
        room_name = f"{st.session_state['username']}'s Room"

    if col2.button(":arrow_right:", help="Join room", disabled=room_name is None):
        room_join_password_dialog(room_name)

    if col3.button(":tada:", help="Create room", disabled=room_name is None):
        room_creation_password_dialog(room_name)

    if col4.button(":arrows_counterclockwise:", help="Refresh the room"):
        st.session_state["do_autorefresh"] = True
        st.rerun()
        return

    # Get room list
    r = post_to_server(
        url=f"{st.session_state['server_url']}/rooms",
        payload={
            "id": st.session_state["server_provided_user_id"],
            "username": st.session_state["username"],
        },
    )
    if r is None:
        return
    if r.status_code != 200:
        st.error("An error occured trying to fetch room data")
        return

    rooms = r.json()

    if len(rooms) == 0:
        info_placeholder.info("There are no rooms yet, but you may create your own!")

    col_structure = (3, 3, 1)
    room_fields = ["Name", "Owner", "Join"]

    cols = st.columns(col_structure)
    for col, field_name in zip(cols, room_fields):
        col.write(field_name)
    for i, room in enumerate(rooms):
        col1, col2, col3 = st.columns(col_structure)
        col1.write(f"{room['name']} {'*(protected)*' if room['is_protected'] else ''}")
        col2.write(room["owner_name"])
        if col3.button(
            ":arrow_right:",
            help="Room is locked" if room["is_locked"] else "Join",
            disabled=room["is_locked"],
            key=f"join-button-{i}",
        ):
            if room["is_protected"]:
                room_join_password_dialog(room["name"])
            else:
                r = post_to_server(
                    url=f"{st.session_state['server_url']}/rooms/{room['name']}/join",
                    payload={
                        "id": st.session_state["server_provided_user_id"],
                        "username": st.session_state["username"],
                        "password": None,
                    },
                )

                if r is None:
                    return
                if r.status_code == 200:
                    st.session_state["do_autorefresh"] = True
                    st.session_state["current_room"] = r.json()
                    st.rerun()
                    return
                st.error("An error occured trying to join the room.")
    return


# ,------.                             ,------.           ,--.          ,--.,--.
# |  .--. ' ,---.  ,---. ,--,--,--.    |  .-.  \  ,---. ,-'  '-. ,--,--.`--'|  | ,---.
# |  '--'.'| .-. || .-. ||        |    |  |  \  :| .-. :'-.  .-'' ,-.  |,--.|  |(  .-'
# |  |\  \ ' '-' '' '-' '|  |  |  |    |  '--'  /\   --.  |  |  \ '-'  ||  ||  |.-'  `)
# `--' '--' `---'  `---' `--`--`--'    `-------'  `----'  `--'   `--`--'`--'`--'`----'


def step_show_room_details():
    room = st.session_state["current_room"]
    if room["is_processing"]:
        st.write(
            f"## Room: {room['name']} <sup>(in progress)</sup>", unsafe_allow_html=True
        )
    elif room["is_finished"]:
        st.write(
            f"## Room: {room['name']} <sup>(finished)</sup>", unsafe_allow_html=True
        )
        st.session_state["do_autorefresh"] = False
    else:
        st.write(
            f"## Room: {room['name']} <sup>({'hidden' if room['is_hidden'] else 'public'}) ({'locked' if room['is_locked'] else 'open'}) {'(protected)' if room['is_protected'] else ''}</sup>",
            unsafe_allow_html=True,
        )

    st.write(f"<sup>Room protocol: {room['algorithm']}<sup>", unsafe_allow_html=True)
    if room["algorithm"] == env.Algorithm.FEDCI and room["is_processing"]:
        current_x_labels = sorted(
            list(
                set(
                    [
                        l.split("__cat__")[0].split("__ord__")[0]
                        for l in room["algorithm_info"]["X_labels"]
                    ]
                )
            )
        )
        st.write(
            f"<sup>Currently running: {room['algorithm_info']['y_label']} ~ {','.join(current_x_labels + ['1'])}<sup>",
            unsafe_allow_html=True,
        )
    # spinner_placeholder = st.empty()

    _, col1, col2, col3, col4, col5, _ = st.columns((1, 1, 1, 1, 1, 1, 1))

    if col1.button(
        ":arrows_counterclockwise:", help="Refresh the room", use_container_width=True
    ):
        st.session_state["do_autorefresh"] = True
        st.rerun()
        return

    if room["is_locked"]:
        lock_button_text = ":lock:"
        lock_button_help_text = "Unlock the room"
    else:
        lock_button_text = ":unlock:"
        lock_button_help_text = "Lock the room"

    if col2.button(
        lock_button_text,
        help=lock_button_help_text,
        disabled=st.session_state["username"] != room["owner_name"],
        use_container_width=True,
    ):
        lock_endpoint = "unlock" if room["is_locked"] else "lock"
        r = post_to_server(
            url=f"{st.session_state['server_url']}/rooms/{room['name']}/{lock_endpoint}",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
            },
        )
        if r is None:
            return
        if r.status_code == 200:
            st.session_state["current_room"] = r.json()
            st.rerun()
            return
        st.error(f"An error occured while trying to {lock_endpoint} the room")

    if room["is_hidden"]:
        hide_button_text = ":face_in_clouds:"
        hide_button_help_text = "Reveal the room"
    else:
        hide_button_text = ":eyes:"
        hide_button_help_text = "Hide the room"

    if col3.button(
        hide_button_text,
        help=hide_button_help_text,
        disabled=st.session_state["username"] != room["owner_name"],
        use_container_width=True,
    ):
        hide_endpoint = "reveal" if room["is_hidden"] else "hide"
        r = post_to_server(
            url=f"{st.session_state['server_url']}/rooms/{room['name']}/{hide_endpoint}",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
            },
        )
        if r is None:
            return
        if r.status_code == 200:
            st.session_state["current_room"] = r.json()
            st.rerun()
            return
        st.error(f"An error occured while trying to {hide_endpoint} the room")

    if col4.button(":arrow_left:", help="Leave the room", use_container_width=True):
        r = post_to_server(
            url=f"{st.session_state['server_url']}/rooms/{room['name']}/leave",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
            },
        )
        if r is None:
            return
        if r.status_code == 200:
            st.session_state["do_autorefresh"] = True
            st.session_state["current_room"] = None
            st.rerun()
            return
        st.error(f"An error occured while trying to leave the room")

    if col5.button(
        ":fire:",
        help="Run IOD on participant data",
        disabled=st.session_state["username"] != room["owner_name"],
        use_container_width=True,
    ):
        r = post_to_server(
            url=f"{st.session_state['server_url']}/run/{room['name']}",
            payload={
                "id": st.session_state["server_provided_user_id"],
                "username": st.session_state["username"],
                "alpha": round(st.session_state["alpha_value"], 2),
                "max_conditioning_set": st.session_state["max_conditioning_set"],
            },
        )
        if r is None:
            return
        if r.status_code == 200:
            st.session_state["current_room"] = r.json()
            st.rerun()
            return
        st.error(f"An error occured while trying to run IOD")

    def change_alpha_value():
        st.session_state["_alpha_value"] = st.session_state["alpha_value"]

    # Run config
    if room["algorithm"] == "FEDGLM":
        _, col1, col2, _ = st.columns((1, 3, 3, 1))
    else:
        _, col1, _ = st.columns((1, 5, 1))
    col1.number_input(
        "Select alpha value:",
        value=st.session_state["_alpha_value"],
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="alpha_value",
        format="%.2f",
        on_change=change_alpha_value,
    )
    if room["algorithm"] == "FEDGLM":

        def change_max_conditioning_set():
            st.session_state["_max_conditioning_set"] = st.session_state[
                "max_conditioning_set"
            ]

        col2.number_input(
            "Select max conditioning set size:",
            value=st.session_state["_max_conditioning_set"],
            min_value=1,
            max_value=999,
            step=1,
            key="max_conditioning_set",
            on_change=change_max_conditioning_set,
        )
    st.empty()

    col_structure = (1, 3, 3, 1)
    room_fields = ["№", "Name", "Provided Labels", "Action"]
    cols = st.columns(col_structure)
    for col, field_name in zip(cols, room_fields):
        col.write(field_name)

    for i, user in enumerate(room["users"]):
        col1, col2, col3, col4 = st.columns(col_structure)
        col1.write(i)

        user_str = f"{user}"
        if user == st.session_state["username"]:
            user_str += " *(you)*"
        if user == room["owner_name"]:
            user_str += " *(owner)*"
        col2.write(user_str)

        with col3.expander("Show"):
            for label in sorted(room["user_provided_labels"][user]):
                st.markdown(f"- {label}")

        if user != st.session_state["username"]:
            if col4.button(
                ":x:",
                help="Kick",
                disabled=st.session_state["username"] != room["owner_name"],
                key=f"kick-button-{i}",
                use_container_width=True,
            ):
                r = post_to_server(
                    url=f"{st.session_state['server_url']}/rooms/{room['name']}/kick/{user}",
                    payload={
                        "id": st.session_state["server_provided_user_id"],
                        "username": st.session_state["username"],
                    },
                )
                if r is None:
                    return
                if r.status_code == 200:
                    st.session_state["current_room"] = r.json()
                    st.rerun()
                    return
                st.error(f"Failed to kick user")

    if room["algorithm"] == "FEDGLM" and room["is_processing"]:
        fedglm_status = room["algorithm_info"]
        if fedglm_status is None:
            raise Exception("FEDGLM STATUS CANNOT BE NONE IF ROOM IS PROCESSING")
        provide_fedglm_data(room, fedglm_status)

    return


# ,------.                       ,--.  ,--.      ,--.   ,--.,--.                       ,--.,--.                 ,--.  ,--.
# |  .--. ' ,---.  ,---. ,--.,--.|  |,-'  '-.     \  `.'  / `--' ,---. ,--.,--. ,--,--.|  |`--',-----. ,--,--.,-'  '-.`--' ,---. ,--,--,
# |  '--'.'| .-. :(  .-' |  ||  ||  |'-.  .-'      \     /  ,--.(  .-' |  ||  |' ,-.  ||  |,--.`-.  / ' ,-.  |'-.  .-',--.| .-. ||      \
# |  |\  \ \   --..-'  `)'  ''  '|  |  |  |         \   /   |  |.-'  `)'  ''  '\ '-'  ||  ||  | /  `-.\ '-'  |  |  |  |  |' '-' '|  ||  |
# `--' '--' `----'`----'  `----' `--'  `--'          `-'    `--'`----'  `----'  `--`--'`--'`--'`-----' `--`--'  `--'  `--' `---' `--''--'

arrow_type_lookup = {1: "odot", 2: "normal", 3: "none"}


def data2graph(data, labels):
    graph = graphviz.Digraph(format="png")
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            arrhead = data[i][j]
            arrtail = data[j][i]
            if data[i][j] == 1:
                graph.edge(
                    labels[i],
                    labels[j],
                    arrowtail=arrow_type_lookup[arrtail],
                    arrowhead=arrow_type_lookup[arrhead],
                )
            elif data[i][j] == 2:
                graph.edge(
                    labels[i],
                    labels[j],
                    arrowtail=arrow_type_lookup[arrtail],
                    arrowhead=arrow_type_lookup[arrhead],
                )
            elif data[i][j] == 3:
                graph.edge(
                    labels[i],
                    labels[j],
                    arrowtail=arrow_type_lookup[arrtail],
                    arrowhead=arrow_type_lookup[arrhead],
                )

    return graph


def step_view_results():
    room = st.session_state["current_room"]
    result_graphs = [
        data2graph(d, l) for d, l in zip(room["result"], room["result_labels"])
    ]
    if room["private_result"] is not None:
        t1, t2 = st.tabs(["Combined Results", "Private Result"])
        private_result_graph = data2graph(
            room["private_result"], room["private_labels"]
        )

        with t2:
            _, col1, _ = st.columns((1, 1, 1))
            col1.download_button(
                label="Download PAG",
                data=private_result_graph.pipe(format="png"),
                file_name="federated-private-pag.png",
                mime="image/png",
                use_container_width=True,
            )
            col1.graphviz_chart(private_result_graph)
    else:
        t1 = st.container()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in [
            (f"federated-pag-{i}.png", g.pipe(format="png"))
            for i, g in enumerate(result_graphs)
        ]:
            zip_file.writestr(file_name, data)

    with t1:
        _, col1, _ = st.columns((1, 2, 1))
        col1.download_button(
            label="Download all PAGs",
            data=zip_buffer,
            file_name=f"federated-pags.zip",
            mime="application/x-zip",
            use_container_width=True,
        )
        cols = st.columns((1, 1, 1))
        for i, g in enumerate(result_graphs):
            cols[i % 3].download_button(
                label="Download PAG",
                data=g.pipe(format="png"),
                file_name=f"federated-pag-{i}.png",
                mime="image/png",
                use_container_width=True,
            )
            cols[i % 3].graphviz_chart(g)
            cols[i % 3].write("---")

    # use visualization library to show images of pags as well
    return


# ,--.   ,--.        ,--.            ,------.                            ,---.   ,--.                         ,--.
# |   `.'   | ,--,--.`--',--,--,     |  .--. ' ,--,--. ,---.  ,---.     '   .-',-'  '-.,--.--.,--.,--. ,---.,-'  '-.,--.,--.,--.--. ,---.
# |  |'.'|  |' ,-.  |,--.|      \    |  '--' |' ,-.  || .-. || .-. :    `.  `-.'-.  .-'|  .--'|  ||  || .--''-.  .-'|  ||  ||  .--'| .-. :
# |  |   |  |\ '-'  ||  ||  ||  |    |  | --' \ '-'  |' '-' '\   --.    .-'    | |  |  |  |   '  ''  '\ `--.  |  |  '  ''  '|  |   \   --.
# `--'   `--' `--`--'`--'`--''--'    `--'      `--`--'.`-  /  `----'    `-----'  `--'  `--'    `----'  `---'  `--'   `----' `--'    `----'


def main():
    st.write("# Welcome to {Some App}")

    col1, col2, _ = st.columns((1, 1, 3))
    col1.write(
        "<sup>View our paper [here](https://www.google.com)</sup>",
        unsafe_allow_html=True,
    )
    col2.write(
        "<sup>View our GitHub [here](https://www.google.com)</sup>",
        unsafe_allow_html=True,
    )

    if check_server_connection():
        st.info(
            "Server connection established"
            + (
                ""
                if st.session_state["selected_algorithm"] is None
                else f" - running {st.session_state['selected_algorithm']} method"
            )
            + (
                ""
                if st.session_state["username"] is None
                else f" - checked in as: {st.session_state['username']}"
            )
        )

    refresh_failure = False
    # refresh current room
    if st.session_state["current_room"] is not None:
        try:
            r = post_to_server(
                url=f"{st.session_state['server_url']}/rooms/{st.session_state['current_room']['name']}",
                payload={
                    "id": st.session_state["server_provided_user_id"],
                    "username": st.session_state["username"],
                },
            )
            if r is None:
                return
            if r.status_code == 200:
                st.session_state["current_room"] = r.json()
            else:
                st.session_state["current_room"] = None
        except:
            refresh_failure = True

    step = stx.stepper_bar(
        steps=[
            "Upload Data",
            "Server Check-In",
            "Process Data",
            "Join Room",
            "View Result",
        ],
        lock_sequence=False,
    )

    if refresh_failure:
        st.error("An error occured trying to update current room data")
        return

    if step > 1 and st.session_state["is_connected_to_server"] == False:
        st.warning(
            "Please ensure you have a connection to the server before continuing"
        )
        return

    if step == 0:
        step_upload_data()
    elif step == 1:
        if st.session_state["uploaded_data"] is None:
            st.info("Please upload a file before continuing")
            return
        step_check_in_to_server()
    elif step == 2:
        if st.session_state["username"] is None:
            st.info("Please check in with the server before continuing")
            return
        if st.session_state["selected_algorithm"] != env.Algorithm.META_ANALYSIS:
            st.info("""
                    This step is only required for Meta Analysis.
                    Please continue to the next step.
                    """)
            return
        step_process_data()
    elif step == 3:
        if (
            st.session_state["selected_algorithm"] == env.Algorithm.META_ANALYSIS
            and st.session_state["server_has_received_data"] is False
        ):
            st.info("Please send your data to the server before continuing")
            return
        if st.session_state["current_room"] is None:
            step_join_rooms()
        else:
            step_show_room_details()
    elif step == 4:
        # TODO: verify that data has provided results
        if (
            st.session_state["current_room"] is None
            or st.session_state["current_room"]["result"] is None
        ):
            st.info("Please run the algorithm before attempting to view results")
            return
        st.session_state["do_autorefresh"] = False
        step_view_results()
    else:
        st.error("An error occured. Please reload the page")

    if step > 2 and st.session_state["do_autorefresh"]:
        st_autorefresh(interval=3000, limit=100, key="autorefresh")


main()
