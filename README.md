# About this Project

This projects aims to create an accessible platform and user interface to obtain causal knowledge from distributed datasets.
Different parties can share key insights about their data without compromising their privacy.
Results come in the form of causal graphs (PAGs).

There are two main algorithm this paper supports:
* rIOD
* FedGLM

## General

This application is made up of two main components.
There is a streamlit UI and a litestar server.

Via the streamlit UI, one can connect to an existing server instance and run distributed/federated algorithms with peers connected to the same server.

It is designed to be easily self-hostable, and is fully contained within a docker container.

**Beware**: As of now, this client-server architecture communicates via http (_not_ https!).
As such a malicious agent may spoof your identity or steal data that is transmitted over the network.

When hosting this application, you may use a reverse-proxy and your own SSL certificates to enable https.

## rIOD

This project implements the IOD algorithm created by [Tillman and Spirtes](http://proceedings.mlr.press/v15/tillman11a.html).

When running rIOD, (conditional) independence tests are performed and the resulting p-values are transmitted to the server.
On the server-side, these p-values are aggregated to give insights about the independences in the distributed dataset.

rIOD only supports numerical (float) features.
As per the IOD algorithm, not all participating parties have to have identical features in their dataset.

Without code modifications, this application will only ever transmit p-values and PAG adjacency matrices over the network.
The original dataset is kept completely private and is never transmitted over the network.
(This can be easily checked inside the source code)

## FedGLM

This project implemens an algorithm (titled FedGLM for now) which utilizes federated learning to create linear models, which are then used for likelihood-ratio tests in order to obtain independence information about the dataset.

FedGLM supports numerical (float), categorical (string), and ordinal (int) features.
Similar to IOD, here as well, not all participating parties have to have the exact same feature set.

This algorithm requires the transmission of the expression levels of ordinal and categorical variables.
Additionaly, when the algorithm is running, linear model coefficients are transmitted, as well as matrices that do not contain recreatable information about the data (see algorithm 2 in [Cellamare et al.](https://www.mdpi.com/1999-4893/15/7/243))
As such, data privacy is preserved.

## Setup

First, install docker or an alternative like podman.

Pull the image:
`docker pull docker.io/maximilianhahn/fedci-iod-app:0.0.1`

Then run the container:
`docker run --rm -it -p 8501:8501 -e MODE=HYBRID docker.io/maximilianhahn/fedci-iod-app:0.0.1`


Modes `CLIENT` and `HYBRID` start streamlit server on port `8501`, which is exposed via the `-p` flag to localhost, therefore allowing access from the local browser under `localhost:8501`.  
Modes `SERVER` and `HYBRID` start a litestar server, running the backend and managing clients. The server runs on port `8000` by default.  

Network connectivity between separate containers, one running as a server and others running as clients requires proper setup and potential docker networking.  
Since no https is supported, you may consider utilizing reverse proxies to encrypt transferred data.

## Known Issues

- Since R requires to be run on the main thread in a lot of circumstances, there can be issues with R code execution on client site, since streamlit delegates threading. This is fixed for fedCI, but meta-analysis may face problems.
- The app can become unstable after long periods of running and multiple user actions, therefore restarts may be required.
- Streamlit components sometimes lose their state, for example, showing the wrong step indicator, although the page loads correctly.
