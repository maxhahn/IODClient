#!/bin/bash

if [ "$MODE" = "CLIENT" ]; then
    streamlit run streamlit/app.py --server.enableXsrfProtection false -server.port 8501
elif [ "$MODE" = "SERVER" ]; then
    litestar --app-dir litestar run
elif [ "$MODE" = "HYBRID" ]; then
    streamlit run streamlit/app.py --server.enableXsrfProtection false --server.fileWatcherType none --server.port 8501 & litestar --app-dir litestar run --debug
else
    echo 'Please choose on of >CLIENT<, >SERVER<, or >HYBRID< as the MODE environment variable'
    exit 1
fi
