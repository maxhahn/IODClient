import streamlit as st
import extra_streamlit_components as stx
from streamlit_extras.dataframe_explorer import dataframe_explorer

import datetime
import requests
import os
import shutil
import sys
import random
import threading

import base64
import pickle
import graphviz

import pandas as pd
import polars as pl

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from collections import OrderedDict

# Add the parents parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import env
import fedci

if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'server_url' not in st.session_state:
    if 'LITESTAR_CONTAINER_NAME' in os.environ and 'LITESTAR_PORT' in os.environ:
        st.session_state['server_url'] = f'http://{os.environ["LITESTAR_CONTAINER_NAME"]}:{os.environ["LITESTAR_PORT"]}'
    else:
        st.session_state['server_url'] = 'http://127.0.0.1:8000'
if 'server_provided_user_id' not in st.session_state:
    st.session_state['server_provided_user_id'] = None
if 'selected_algorithm' not in st.session_state:
    st.session_state['selected_algorithm'] = env.Algorithm.META_ANALYSIS

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'data_filename' not in st.session_state:
    st.session_state['data_filename'] = None
if 'data_labels' not in st.session_state:
    st.session_state['data_labels'] = None
if 'data_pvals' not in st.session_state:
    st.session_state['data_pvals'] = None

if 'last_health_check' not in st.session_state:
    st.session_state['last_health_check'] = None
if 'is_connected_to_server' not in st.session_state:
    st.session_state['is_connected_to_server'] = None


if '_alpha_value' not in st.session_state:
    st.session_state['_alpha_value'] = 0.05
if '_local_alpha_value' not in st.session_state:
    st.session_state['_local_alpha_value'] = 0.05

if '_max_conditioning_set' not in st.session_state:
    st.session_state['_max_conditioning_set'] = None
if 'max_conditioning_set' not in st.session_state:
    st.session_state['max_conditioning_set'] = 500

if '_local_max_conditioning_set' not in st.session_state:
    st.session_state['_local_max_conditioning_set'] = 1

if '_rpc_port' not in st.session_state:
    st.session_state['_rpc_port'] = None
if 'rpc_port' not in st.session_state:
    st.session_state['rpc_port'] = random.randrange(18850,18900)
if '_rpc_hostname' not in st.session_state:
    st.session_state['_rpc_hostname'] = None
if 'rpc_hostname' not in st.session_state:
    st.session_state['rpc_hostname'] = '127.0.0.1'

if 'current_room' not in st.session_state:
    st.session_state['current_room'] = None
if 'fedci_client' not in st.session_state:
    st.session_state['fedci_client'] = None

if 'do_autorefresh' not in st.session_state:
    st.session_state['do_autorefresh'] = True

if 'server_has_received_data' not in st.session_state:
    st.session_state['server_has_received_data'] = False

# Store previous data uploads
client_base_dir = './IOD'
upload_dir = f'{client_base_dir}/uploaded_files'
ci_result_dir = f'{client_base_dir}/ci'

# Init upload files dir
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir, exist_ok=True)
if not os.path.exists(ci_result_dir):
    os.makedirs(ci_result_dir, exist_ok=True)

# always update
st.session_state['data_on_disk'] = os.listdir(upload_dir)


###
# Helper
###

def post_to_server(url, payload):
    #r = requests.post(url=url, json=payload)
    try:
        r = requests.post(url=url, json=payload)
        return r
    except:
        st.session_state['last_health_check'] = None
        st.error('There are problems with the server connection')
    return None

def run_local_fci():
    df = st.session_state['data_pvals']
    labels = st.session_state['data_labels']
    alpha = st.session_state['local_alpha_value']

    with (ro.default_converter + pandas2ri.converter).context():
        ro.r['source']('./scripts/ci_functions.r')
        iod_on_ci_data_f = ro.globalenv['iod_on_ci_data']

        suff_stat = [
            ('citestResults', ro.conversion.get_conversion().py2rpy(df)),
            ('all_labels', ro.StrVector(labels)),
        ]

        suff_stat = OrderedDict(suff_stat)
        suff_stat = ro.ListVector(suff_stat)
        label_list = [ro.StrVector(labels)]

        result = iod_on_ci_data_f(label_list, suff_stat, alpha)

        # R script problems, out of memory and error with string conversions. Why?
        pag = [x[1].tolist() for x in result['Gi_PAG_list'].items()][0]
        pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()][0]

        if any([a!=b for a,b in zip(pag_labels, st.session_state['data_labels'])]):
            st.error('''A major issue with the data labels occured.
                        Please reload the page.
                        If this error persists, contact an administrator.
                        ''')

        return pag

arrow_type_lookup = {
        1: 'odot',
        2: 'normal',
        3: 'none'
    }
def data2graph(data, labels):
    graph = graphviz.Digraph(format='png')
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            arrhead = data[i][j]
            arrtail = data[j][i]
            if data[i][j] == 1:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
            elif data[i][j] == 2:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
            elif data[i][j] == 3:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])

    return graph

###
# Funcs
###

def app_header():
    st.write('# Welcome to {Some App}')

    col1, col2, _ = st.columns((1,1,3))
    col1.write('<sup>View our paper [here](https://www.google.com)</sup>', unsafe_allow_html=True)
    col2.write('<sup>View our GitHub [here](https://www.google.com)</sup>', unsafe_allow_html=True)
    return

def check_server_connection():
    curr_time = datetime.datetime.now()
    if st.session_state['last_health_check'] is not None and (curr_time - st.session_state['last_health_check']).total_seconds() < 60:
        return True
    try:
        #with st.spinner('Checking connection to server...'):
        r = requests.get(url = f'{st.session_state["server_url"]}/health-check')

        if r.status_code != 200:
            st.error('There are problems with the server connection')
            st.session_state['is_connected_to_server'] = False
            return False
    except:
        st.error('There are problems with the server connection')
        st.session_state['is_connected_to_server'] = False
        return False

    st.session_state['last_health_check'] = curr_time
    st.session_state['is_connected_to_server'] = True
    return True

def step_check_in_to_server():
    if st.session_state['server_provided_user_id'] is not None:
        st.info('Server check-in completed')

    col1, col2 = st.columns((6,1))
    col1.write('Please enter the server URL:')
    server_url = col1.text_input('Server URL:', placeholder=st.session_state['server_url'], label_visibility='collapsed')

    col2.write('Connect!')
    if col2.button(':link:', help='Connect to URL', use_container_width=True) and server_url is not None and server_url != '':
        if st.session_state['server_provided_user_id'] is not None:
            st.warning('''You are already checked in with a server.
                        In order to leave the server, refresh this page.''')
        else:
            if server_url.endswith('/'):
                server_url = server_url[:-1]
            st.session_state['server_url'] = server_url
            st.session_state['last_health_check'] = None
            st.rerun()
            return

    st.write('---')

    container = st.container()

    col1, col2, col3 = st.columns((4,2,1))
    col1.write('Please enter a username:')
    username = col1.text_input('Username:', placeholder=st.session_state['username'], label_visibility='collapsed')

    col2.write('Select algorithm')
    # TODO: ensure selected algorithm is default select
    algo_type = col2.selectbox('Algorithm', [e.name for e in env.Algorithm], label_visibility='collapsed')

    col3.write('Submit!')
    if st.session_state['server_provided_user_id'] is None:
        button_text = 'Submit check-in request'
        request_url = f'{st.session_state["server_url"]}/user/check-in'

        request_params = {
            'username': username,
            'algorithm': algo_type
            }
    else:
        button_text = 'Update user'
        request_url = f'{st.session_state["server_url"]}/user/update'
        request_params = {
            'id': st.session_state['server_provided_user_id'],
            'algorithm': algo_type,
            'username': st.session_state['username'],
            'new_username': username
            }

    button = col3.button(':arrow_heading_up:', help=button_text, use_container_width=True)
    if button:
        st.session_state['server_has_received_data'] = False
        r = post_to_server(request_url, request_params)
        if r is None:
            return
        if r.status_code != 200:
            st.error('Failed to check in with the server. Please reload the page')
            return

        r = r.json()
        st.session_state['server_provided_user_id'] = r['id']
        st.session_state['username'] = r['username']
        st.session_state['selected_algorithm'] = r['algorithm']
        st.rerun()

    with st.expander('Algorithm description'):
        if algo_type == env.Algorithm.FEDCI.name:
            st.markdown("""
                The FedCI Algorithm uses a federated, iterative procedure to determine independence between the provided sets of variables.  """ """
                When this algorithm is started, a RPC server is launched on the client-side which will communicate with the server to do calculations.

                This algorithm shares:
                - Parametrization of linear models (sent from the server)
                - Quality of fit metrics
                - Fisher Information matrices and Score Vectors
                - Expression levels of ordinal and nominal variables

                In most cases, this should not be privacy relevant data.  """ """
                Nevertheless, beware of the aforementioned information exchanges.
                """)
        elif algo_type == env.Algorithm.META_ANALYSIS.name:
            st.markdown("""
                The Meta Analysis algorithm uses locally computed p-values and sends them to the server, which aggregates these from all contributing clients.  """ """
                All client-side computation is performed up-front and the results will be transmitted to the server once.

                The algorithm shares:
                - p-values of CI tests between all provided variables

                In most cases, this should not be privacy relevant data.  """ """
                Nevertheless, beware of the aforementioned information exchanges.
                """)
        else:
            st.warning(f'Unknown algorithm >>{algo_type}<< selected')
    return

def step_upload_data():
    uploaded_file = st.file_uploader('Data Upload', type=['csv', 'parquet'])

    no_files_uploaded_yet = len(st.session_state['data_on_disk']) == 0
    st.write('Select previously uploaded data:')
    col1, col2 = st.columns((6,1))
    previously_uploaded_file = col1.selectbox('Select previously uploaded data:',
                                            st.session_state['data_on_disk'],
                                            index=None,
                                            disabled=no_files_uploaded_yet,
                                            label_visibility='collapsed',
                                            help='No files uploaded so far' if no_files_uploaded_yet else 'Please select the file you want to work with')

    if col2.button(':wastebasket:', help='Remove all local data. THIS INCLUDES PROCESSED DATA AND PAGs.', use_container_width=True):
        shutil.rmtree(client_base_dir)
        st.rerun()
        return

    if uploaded_file is None and previously_uploaded_file is None:
        return

    # read uploaded file into session state
    if uploaded_file is not None:
        filename = uploaded_file.name
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            filename = filename[:-4] + '.parquet'
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            raise Exception('Cannot handle files of the given type')
        df.to_parquet(f'{upload_dir}/{filename}', index=False)
    elif previously_uploaded_file is not None:
        filename = previously_uploaded_file
        df = pd.read_parquet(f'{upload_dir}/{previously_uploaded_file}')

    st.session_state['data'] = df
    st.session_state['data_filename'] = filename
    # Reset when new file is uploaded
    st.session_state['data_labels'] = list(df.columns)
    st.session_state['data_pvals'] = None

    client_port = 16016
    client = fedci.ProxyClient(pl.from_pandas(df))
    client_thread = threading.Thread(target=client.start, args=(16016,), daemon=True)
    client_thread.start()

    #st.session_state['fedglm_client'] = client
    #st.session_state['fedglm_client_thread'] = client_thread
    #st.session_state['fedglm_client_port'] = client_port

    if st.session_state['data'] is not None:
        with st.expander('View Data'):
            df = st.session_state['data']
            st.dataframe(dataframe_explorer(df), use_container_width=True)
    return

def step_process_data():
    _, col1, col2, _, col3, _ = st.columns((1,3,3,1,6,1))

    fileid = os.path.splitext(st.session_state['data_filename'])[0]
    filename = f"citestResults_{st.session_state['data_filename']}"

    #c1, c2 = st.columns((1,1))
    if col2.button(':wastebasket:',
                help='Delete current progress, so that data can be reprocessed from scratch',
                disabled=not os.path.exists(f'{ci_result_dir}/{filename}'), use_container_width=True):
        os.remove(f'{ci_result_dir}/{filename}')
        st.session_state['data_pvals'] = None
        st.rerun()
        return


    if col1.button("Process Data!", use_container_width=True):
        max_conditioning_set = st.session_state['local_max_conditioning_set']

        with st.spinner('Data is being processed...'):
            # Read data from state
            df = st.session_state['data']

            # Call R function
            with (ro.default_converter + pandas2ri.converter).context():
                # load local-ci script
                ro.r['source']('./scripts/ci_functions.r')

                # load function from R script
                run_ci_test_f = ro.globalenv['run_ci_test']

                #converting it into r object for passing into r function
                df_r = ro.conversion.get_conversion().py2rpy(df)
                #Invoking the R function and getting the result
                result = run_ci_test_f(df_r, max_conditioning_set, ci_result_dir+"/", fileid)
                #Converting it back to a pandas dataframe.
                df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
                labels = list(result['labels'])

                if any([a!=b for a,b in zip(labels, st.session_state['data_labels'])]):
                    st.error('''A major issue with the data labels occured.
                                Please reload the page.
                                If this error persists, contact an administrator.
                                ''')


        st.session_state['data_pvals'] = df_pvals

        st.rerun()

    if col3.button('Submit Data!', help='Submit Data to Server', disabled=st.session_state['data_pvals'] is None, use_container_width=True):
        df_pvals = st.session_state['data_pvals']
        labels = st.session_state['data_labels']
        base64_df = base64.b64encode(pickle.dumps(df_pvals)).decode('utf-8')
        # send data and labels
        r = post_to_server(url = f'{st.session_state["server_url"]}/user/submit-data', payload={'id': st.session_state['server_provided_user_id'],
                                                                    'username': st.session_state['username'],
                                                                    'data': base64_df,
                                                                    'data_labels': labels
                                                                    })
        if r is None:
            return
        if r.status_code != 200:
            st.error('An error occured when submitting the data')
            return
        st.session_state['server_has_received_data'] = True

    def change_cond_set_value():
        st.session_state['_local_max_conditioning_set'] = st.session_state['local_max_conditioning_set']
    _, col1, _ = st.columns((1,6,1))
    col1.number_input('Select the maximum conditiong set size:',
                      value=st.session_state['_local_max_conditioning_set'],
                      min_value=0,
                      step=1,
                      key='local_max_conditioning_set',
                      on_change=change_cond_set_value)



    if st.session_state['server_has_received_data'] == True:
        st.success('The server has received your data')

    if st.session_state['data_pvals'] is not None and st.session_state['server_has_received_data'] == False:
        st.warning('''Any submitted data can be accessed by the server.
                                 Any participants in the same room will be able to access the data labels.
                                 Once you join a room, your data can be used in the processing rooms data.
                                 Be sure no sensitive data is submitted!''')

    if st.session_state['data_pvals'] is not None:

        tab1, tab2 = st.tabs(['Processed Data', 'Generated PAG'])

        df_pvals = st.session_state['data_pvals'].copy()
        labels = st.session_state['data_labels']

        with tab1:
            df_pvals['X'] = df_pvals['X'].apply(lambda x: labels[int(x)-1])
            df_pvals['Y'] = df_pvals['Y'].apply(lambda x: labels[int(x)-1])
            df_pvals['S'] = df_pvals['S'].apply(lambda x: ','.join(sorted([labels[int(xi)-1] for xi in x.split(',') if xi != ''])))
            st.dataframe(dataframe_explorer(df_pvals), use_container_width=True)
        with tab2:
            def change_local_alpha_value():
                st.session_state['_local_alpha_value'] = st.session_state['local_alpha_value']
            st.number_input('Select alpha value:',
                            value=st.session_state['_local_alpha_value'],
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            key='local_alpha_value',
                            format='%.2f',
                            on_change=change_local_alpha_value)

            with st.spinner('Preparing PAG...'):
                pag_mat = run_local_fci()
                pag = data2graph(pag_mat, st.session_state['data_labels'])

            _, col1, _ = st.columns((1,1,1))
            col1.download_button(
                label="Download PAG",
                data=pag.pipe(format='png'),
                file_name=f"local-pag-{fileid}.png",
                mime="image/png", use_container_width=True
            )
            col1.graphviz_chart(pag)

    return

def step_setup_rpc():
    col1x, col2x, _ = st.columns((4,2,1))
    col1, col2, col3 = st.columns((4,2,1))

    def change_hostname():
        st.session_state['_rpc_port'] = st.session_state['rpc_port']
    col1x.write('Hostname:')
    hostname = col1.text_input('Hostname:', placeholder=st.session_state['rpc_hostname'], label_visibility='collapsed', key='rpc_hostname', on_change=change_hostname)

    def change_port():
        st.session_state['_rpc_port'] = st.session_state['rpc_port']
    col2x.write('Port:')
    port = col2.number_input('Port:', value=st.session_state['rpc_port'], step=1, label_visibility='collapsed', key='rpc_port', on_change=change_port)

    if col3.button(':arrow_heading_up:', help='Send RPC details', use_container_width=True):
        r = post_to_server(
            url = f'{st.session_state["server_url"]}/user/submit-rpc-info',
            payload={
                'id': st.session_state['server_provided_user_id'],
                'username': st.session_state['username'],
                'data_labels': sorted(st.session_state['data_labels']),
                'hostname': hostname,
                'port': port
                }
            )
        st.session_state['server_has_received_data'] = True

    if st.session_state['server_has_received_data'] == True:
        st.success('The server has received your data')
    return


@st.dialog("Secure your room!")
def room_creation_password_dialog(room_name):
    st.session_state['do_autorefresh'] = False
    st.write(f'### Creating room {room_name}...')
    st.write('''You may give your room a password.
             The password will be stored on the server in plain text, so beware!
             ''')
    password = st.text_input("Please enter your password here. Leave empty for no password.")
    _, col1 = st.columns((6,1))
    if col1.button(':arrow_right:', help='Continue with chosen password', use_container_width=True):
        if len(password) == 0:
            password = None

        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/create', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username'],
                                                                'room_name': room_name,
                                                                'algorithm': st.session_state['selected_algorithm'],
                                                                'password': password})

        if r is None:
            return
        if r.status_code == 200:
            st.session_state['do_autorefresh'] = True
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error('An error occured trying to create the room')
    return

@st.dialog("This room is secured!")
def room_join_password_dialog(room_name):
    st.session_state['do_autorefresh'] = False
    st.write(f'### Joining room {room_name}...')
    st.write('''This room is protected by a password.''')
    password = st.text_input("Please enter the password:")
    _, col1 = st.columns((6,1))
    if col1.button(':arrow_right:', help='Continue with chosen password', use_container_width=True):
        if len(password) == 0:
            password = None

        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room_name}/join', payload={
            'id': st.session_state['server_provided_user_id'],
            'username': st.session_state['username'],
            'password': password
            })

        if r is None:
            return
        if r.status_code == 200:
            st.session_state['do_autorefresh'] = True
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error('''An error occured trying to join the room.
                 The password might be incorrect.''')
    return

def step_join_rooms():
    # enter new room name and join or create it
    info_placeholder = st.empty()
    st.write('Please enter a room name:')
    col1, col2, col3, col4 = st.columns((7,1,1,1))

    room_name = col1.text_input('Please chose a room name', label_visibility='collapsed')
    if room_name is None or len(room_name)==0:
        room_name = f"{st.session_state['username']}'s Room"

    if col2.button(':arrow_right:', help='Join room', disabled=room_name is None):
        room_join_password_dialog(room_name)

    if col3.button(':tada:', help='Create room', disabled=room_name is None):
        room_creation_password_dialog(room_name)

    if col4.button(':arrows_counterclockwise:', help='Refresh the room'):
        st.session_state['do_autorefresh'] = True
        st.rerun()
        return

    # Get room list
    r = post_to_server(url = f'{st.session_state["server_url"]}/rooms', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
    if r is None:
        return
    if r.status_code != 200:
        st.error('An error occured trying to fetch room data')
        return

    rooms = r.json()

    if len(rooms) == 0:
        info_placeholder.info('There are no rooms yet, but you may create your own!')

    col_structure = (3,3,1)
    room_fields = ['Name', 'Owner', 'Join']

    cols = st.columns(col_structure)
    for col, field_name in zip(cols, room_fields):
        col.write(field_name)
    for i, room in enumerate(rooms):
        col1, col2, col3 = st.columns(col_structure)
        col1.write(f"{room['name']} {'*(protected)*' if room['is_protected'] else ''}")
        col2.write(room['owner_name'])
        if col3.button(':arrow_right:', help='Room is locked' if room['is_locked'] else 'Join', disabled=room['is_locked'], key=f'join-button-{i}'):
            if room['is_protected']:
                room_join_password_dialog(room['name'])
            else:
                r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/join', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username'],
                                                                'password': None})

                if r is None:
                    return
                if r.status_code == 200:
                    st.session_state['do_autorefresh'] = True
                    st.session_state['current_room'] = r.json()
                    st.rerun()
                    return
                st.error('An error occured trying to join the room.')
    return


def step_show_room_details():
    room = st.session_state['current_room']
    if room['is_processing']:
        st.write(f"## Room: {room['name']} <sup>(in progress)</sup>", unsafe_allow_html=True)
    elif room['is_finished']:
        st.write(f"## Room: {room['name']} <sup>(finished)</sup>", unsafe_allow_html=True)
        st.session_state['do_autorefresh'] = False
    else:
        st.write(f"## Room: {room['name']} <sup>({'hidden' if room['is_hidden'] else 'public'}) ({'locked' if room['is_locked'] else 'open'}) {'(protected)' if room['is_protected'] else ''}</sup>", unsafe_allow_html=True)

    st.write(f"<sup>Room protocol: {room['algorithm']}<sup>", unsafe_allow_html=True)

    _, col1, col2, col3, col4, col5, _ = st.columns((1,1,1,1,1,1,1))

    if col1.button(':arrows_counterclockwise:', help='Refresh the room', use_container_width=True):
        st.session_state['do_autorefresh'] = True
        st.rerun()
        return

    if room['is_locked']:
        lock_button_text = ':lock:'
        lock_button_help_text = 'Unlock the room'
    else:
        lock_button_text = ':unlock:'
        lock_button_help_text = 'Lock the room'

    if col2.button(lock_button_text, help=lock_button_help_text, disabled=st.session_state['username']!=room['owner_name'], use_container_width=True):
        lock_endpoint = 'unlock' if room['is_locked'] else 'lock'
        r = post_to_server(
            url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/{lock_endpoint}',
            payload={
                'id': st.session_state['server_provided_user_id'],
                'username': st.session_state['username']
            })
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to {lock_endpoint} the room')

    if room['is_hidden']:
        hide_button_text = ':face_in_clouds:'
        hide_button_help_text = 'Reveal the room'
    else:
        hide_button_text = ':eyes:'
        hide_button_help_text = 'Hide the room'

    if col3.button(hide_button_text, help=hide_button_help_text, disabled=st.session_state['username']!=room['owner_name'], use_container_width=True):
        hide_endpoint = 'reveal' if room['is_hidden'] else 'hide'
        r = post_to_server(
            url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/{hide_endpoint}',
            payload={
                'id': st.session_state['server_provided_user_id'],
                'username': st.session_state['username']
            })
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to {hide_endpoint} the room')

    if col4.button(':arrow_left:', help='Leave the room', use_container_width=True):
        r = post_to_server(
            url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/leave',
            payload={
                'id': st.session_state['server_provided_user_id'],
                'username': st.session_state['username']
            })
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['do_autorefresh'] = True
            st.session_state['current_room'] = None
            st.rerun()
            return
        st.error(f'An error occured while trying to leave the room')

    if col5.button(':fire:', help='Run IOD on participant data', disabled=st.session_state['username']!=room['owner_name'], use_container_width=True):
        r = post_to_server(
            url = f'{st.session_state["server_url"]}/run/{room["name"]}',
            payload={
                'id': st.session_state['server_provided_user_id'],
                'username': st.session_state['username'],
                'alpha': round(st.session_state['alpha_value'],2),
                'max_conditioning_set': st.session_state['max_conditioning_set']
            })
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to run IOD')

    def change_alpha_value():
        st.session_state['_alpha_value'] = st.session_state['alpha_value']

    # Run config
    if room['algorithm'] == env.Algorithm.FEDCI.name:
        _, col1, col2, _ = st.columns((1,3,3,1))
    else:
        _, col1, _ = st.columns((1,5,1))
    col1.number_input('Select alpha value:',
                      value=st.session_state['_alpha_value'],
                      min_value=0.0,
                      max_value=1.0,
                      step=0.01,
                      key='alpha_value',
                      format='%.2f',
                      on_change=change_alpha_value)
    if room['algorithm'] == env.Algorithm.FEDCI.name:
        def change_max_conditioning_set():
            st.session_state['_max_conditioning_set'] = st.session_state['max_conditioning_set']
        col2.number_input('Select max conditioning set size:',
                          value=st.session_state['max_conditioning_set'],
                          min_value=1,
                          max_value=999,
                          step=1,
                          key='max_conditioning_set',
                          on_change=change_max_conditioning_set)
    st.empty()

    col_structure = (1,3,3,1)
    room_fields = ['№', 'Name', 'Provided Labels', 'Action']
    cols = st.columns(col_structure)
    for col, field_name in zip(cols, room_fields):
        col.write(field_name)

    for i, user in enumerate(room['users']):
        col1, col2, col3, col4 = st.columns(col_structure)
        col1.write(i)

        user_str = f"{user}"
        if user == st.session_state['username']:
            user_str += " *(you)*"
        if user == room['owner_name']:
            user_str += " *(owner)*"
        col2.write(user_str)

        with col3.expander('Show'):
            for label in sorted(room['user_provided_labels'][user]):
                st.markdown(f'- {label}')

        if user != st.session_state['username']:
            if col4.button(':x:', help='Kick', disabled=st.session_state['username'] != room['owner_name'], key=f'kick-button-{i}', use_container_width=True):
                r = post_to_server(
                    url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/kick/{user}',
                    payload={
                        'id': st.session_state['server_provided_user_id'],
                        'username': st.session_state['username']
                    })
                if r is None:
                    return
                if r.status_code == 200:
                    st.session_state['current_room'] = r.json()
                    st.rerun()
                    return
                st.error(f'Failed to kick user')
    return

def main():
    app_header()

    if check_server_connection():
        st.info('Server connection established' + ('' if st.session_state['selected_algorithm'] is None else f" - running {st.session_state['selected_algorithm']} method") + ('' if st.session_state['username'] is None else f" - checked in as: {st.session_state['username']}"))

    if st.session_state['selected_algorithm'] == env.Algorithm.META_ANALYSIS:
        process_step = 'Process Data'
    elif st.session_state['selected_algorithm'] == env.Algorithm.FEDCI:
        process_step = 'RPC Setup'

        if st.session_state['fedci_client'] is None and st.session_state['current_room'] is not None:
            print(f'Launching RPC Client on port: {st.session_state["rpc_port"]}')
            client = fedci.ProxyClient(pl.from_pandas(st.session_state['data']))
            client_thread = threading.Thread(target=client.start, args=(st.session_state['rpc_port'],), daemon=True)
            client_thread.start()
            st.session_state['fedci_client'] = client
            st.session_state['fedci_client_thread'] = client_thread
        elif st.session_state['fedci_client'] is not None and st.session_state['current_room'] is None:
            print('Closing RPC Client')
            st.session_state['fedci_client'].close()
            st.session_state['fedci_client'] = None
            st.session_state['fedci_client_thread'] = None
    else:
        raise Exception(f'Unknown algorithm selected: {st.session_state["selected_algorithm"]}')

    step = stx.stepper_bar(steps=["Server Check-In", "Upload Data", process_step, "Join Room", "View Result"], lock_sequence=False)

    if step == 0:
        step_check_in_to_server()
    elif step == 1:
        if st.session_state['server_provided_user_id'] is None:
            st.info("Please ensure you have checked in with the server before continuing")
            return
        step_upload_data()
    elif step == 2:
        if st.session_state['data'] is None:
            st.info("Please upload data before continuing")
            return
        if st.session_state['selected_algorithm'] == env.Algorithm.META_ANALYSIS:
            step_process_data()
        elif st.session_state['selected_algorithm'] == env.Algorithm.FEDCI:
            step_setup_rpc() # TODO
        else:
            raise Exception(f'Unknown algorithm selected: {st.session_state["selected_algorithm"]}')
    elif step == 3:
        if st.session_state['current_room'] is None:
            step_join_rooms()
        else:
            step_show_room_details()




main()
