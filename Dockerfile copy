# Dockerfile
FROM docker.io/rocker/r-ver:4.4.0

# System dependencies for R packages and rpy2
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libgraphviz-dev \
    python3-pip \
    python3-dev \
    build-essential \
    graphviz \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    libicu-dev \
    libtirpc-dev \
    r-base-dev \
    librsvg2-dev \
    libcairo2-dev \
    libgmp-dev \
    libmpfr-dev \
    libgsl-dev \
    cmake \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install BiocManager and its dependencies first
RUN R -e "install.packages(c('BiocManager', 'devtools'), repos='http://cran.rstudio.com/')"

# Install Bioconductor packages
RUN R -e "BiocManager::install(c('graph', 'Rgraphviz', 'RBGL'), ask=FALSE)"

# Install base dependencies first
RUN R -e "\
    install.packages(c('ggm', 'sfsmisc', 'gmp', 'abind', 'corpcor', 'Rmpfr', 'V8'), \
    repos='http://cran.rstudio.com/', \
    dependencies=TRUE)"

# Install pcalg and its dependencies
RUN R -e "\
    install.packages('pcalg', \
    repos='http://cran.rstudio.com/', \
    dependencies=TRUE)"

# Install rsvg and MXM with dependencies
RUN R -e "\
    install.packages(c('rsvg', 'MXM'), \
    repos='http://cran.rstudio.com/', \
    dependencies=TRUE)"

# Install remaining R packages from requirements.r
COPY requirements.r /tmp/requirements.r
RUN R -e "packages <- readLines('/tmp/requirements.r'); \
    installed_packages <- installed.packages()[,'Package']; \
    packages <- setdiff(packages, installed_packages); \
    if(length(packages) > 0) { \
    install.packages(packages, \
    repos='http://cran.rstudio.com/', \
    dependencies=TRUE) \
    }"

COPY install_mxm.r /tmp/install_mxm.r
RUN Rscript /tmp/install_mxm.r

# Install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy project data
COPY ./imports /r-imports
WORKDIR /r-imports

# Install FCI Utils
RUN R CMD INSTALL /r-imports/FCI.Utils_1.0.tar.gz
# Install rIOD
RUN R CMD INSTALL /r-imports/rIOD_1.0.tar.gz

WORKDIR /
RUN rm -r /r-imports

# ,------.            ,--.  ,--.                        ,------.               ,--.
# |  .--. ',--. ,--.,-'  '-.|  ,---.  ,---. ,--,--,     |  .--. ' ,--,--. ,---.|  |,-. ,--,--. ,---.  ,---.  ,---.
# |  '--' | \  '  / '-.  .-'|  .-.  || .-. ||      \    |  '--' |' ,-.  || .--'|     /' ,-.  || .-. || .-. :(  .-'
# |  | --'   \   '    |  |  |  | |  |' '-' '|  ||  |    |  | --' \ '-'  |\ `--.|  \  \\ '-'  |' '-' '\   --..-'  `)
# `--'     .-'  /     `--'  `--' `--' `---' `--''--'    `--'      `--`--' `---'`--'`--'`--`--'.`-  /  `----'`----'
#          `---'                                                                              `---'

COPY ./app /app
WORKDIR /app

# More Python packages
RUN pip install pandas polars graphviz rpy2 litestar[standard] streamlit extra-streamlit-components streamlit-extras streamlit-autorefresh
RUN pip install statsmodels scipy rpyc

# make startup script executable
RUN chmod +x startup.sh
# Draws config from env vars
CMD ./startup.sh
