# Dockerfile
FROM condaforge/miniforge3:latest

RUN conda config --add channels bioconda

RUN mamba install -y r-base=4.4.0
RUN mamba install -y r-energy r-rsvg

RUN mamba install \
    -c conda-forge -c bioconda \
    r-essentials \
    r-devtools \
    r-remotes \
    r-biocmanager \
    graphviz \
    gmp \
    mpfr \
    libxml2 \
    openssl \
    pkg-config \
    -y

RUN mamba install -c conda-forge \
    r-ggm \
    r-sfsmisc \
    r-abind \
    r-corpcor \
    r-rmpfr \
    r-v8 \
    r-doparallel \
    r-survival \
    r-mass \
    r-ordinal \
    r-geepack \
    r-coxme \
    r-rfast \
    r-bigmemory \
    r-hmisc \
    r-dagitty \
    -y

RUN mamba install -c bioconda -c conda-forge \
    bioconductor-graph \
    bioconductor-rgraphviz \
    bioconductor-rbgl \
    -y

RUN mamba install -y r-pscl r-dot r-matrixcalc r-doFuture r-lme4 r-rje r-gtools r-visnetwork r-quantreg

RUN mamba install -y python-devtools gcc

RUN R -e "install.packages('relations', repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('pcalg', repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('Rfast2', repos='http://cran.rstudio.com/')"
#R -e "install.packages('dagitty', repos='http://cran.rstudio.com/')"

RUN R -e "remotes::install_github('IyarLin/simMixedDAG')"
RUN R -e "remotes::install_github('cran/BFF')"

COPY imports /r-imports


RUN R CMD INSTALL /r-imports/MXM_1.5.5.tar.gz
RUN R CMD INSTALL /r-imports/FCI.Utils_1.0.tar.gz
RUN R CMD INSTALL /r-imports/rIOD_1.0.tar.gz

# ,------.            ,--.  ,--.                        ,------.               ,--.
# |  .--. ',--. ,--.,-'  '-.|  ,---.  ,---. ,--,--,     |  .--. ' ,--,--. ,---.|  |,-. ,--,--. ,---.  ,---.  ,---.
# |  '--' | \  '  / '-.  .-'|  .-.  || .-. ||      \    |  '--' |' ,-.  || .--'|     /' ,-.  || .-. || .-. :(  .-'
# |  | --'   \   '    |  |  |  | |  |' '-' '|  ||  |    |  | --' \ '-'  |\ `--.|  \  \\ '-'  |' '-' '\   --..-'  `)
# `--'     .-'  /     `--'  `--' `--' `---' `--''--'    `--'      `--`--' `---'`--'`--'`--`--'.`-  /  `----'`----'
#          `---'                                                                              `---'


# More Python packages
RUN pip install pandas polars graphviz rpy2 litestar[standard] streamlit extra-streamlit-components streamlit-extras streamlit-autorefresh
RUN pip install statsmodels scipy rpyc

EXPOSE 8501
EXPOSE 8000

COPY ./app /app
WORKDIR /app

COPY fedci /fedci

# make startup script executable
RUN chmod +x startup.sh
# Draws config from env vars
CMD ./startup.sh
