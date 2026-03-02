# About this Project

This projects aims to create an accessible platform and user interface to obtain causal knowledge from distributed datasets.
Different parties can share key insights about their data without compromising their privacy.
Results come in the form of causal graphs (PAGs).

There are three main software components our paper presents:
* fedCI: A framework for federated CI testing, available as `python-fedci` on pypi
* rIOD: A public R implementation of the IOD algorithm TODO: ref to repo and original paper
* fedCI-IOD: A WebApp to try out fedCI and IOD, which is contained in this repository

## General

This WebApp is made up of two main components.  
There are a streamlit UI and a litestar server.

Via the streamlit UI, one can connect to an existing server instance and run distributed/federated algorithms with peers connected to the same server.  
It is designed to be easily self-hostable, and is fully contained within a docker container.

The litestar server manages users, sessions, testing and the combined IOD.
There is no proper account management to save user results long-term, therefore a user should download their results as soon as possible to persist them.

**Beware**: 
This WebApp is not production-grade and serves as an exploratory pool to showcase the usefulness of colaborative causal discovery.  
Please read about known issues in the respective section below.

## rIOD

This project implements the IOD algorithm created by [Tillman and Spirtes](http://proceedings.mlr.press/v15/tillman11a.html).

When running rIOD, (conditional) independence tests are performed and the resulting p-values are transmitted to the server.
On the server-side, these p-values are aggregated to give insights about the independences in the distributed dataset.

rIOD only supports numerical (float) features.
As per the IOD algorithm, not all participating parties have to have identical features in their dataset.

Without code modifications, this application will only ever transmit p-values and PAG adjacency matrices over the network.
The original dataset is kept completely private and is never transmitted over the network.
(This can be easily checked inside the source code)

## fedCI

As an alternative to meta-analysis, as done by Tillman and Spirtes, we have proposed an improved federated CI test, which is based on generalized linear models and likelihood-ratio tests.

FedCI supports numerical (float), categorical (string), and ordinal (int) features.
Similar to IOD, here as well, not all participating parties have to have the exact same feature set.

FedCI fits GLMs through federated Fisher scoring, therefore only aggregate statistics are sent over the network to update model parameters.  
Additional security is provided through additive masking, which lets clients mask their contributions by collectively adding noise which sums to zero, thereby being removed when summing the local statistics, giving the correct final result without revealing local contributions.

## Setup

First, install docker or an alternative like podman.

Pull the image:
```
docker pull docker.io/maximilianhahn/fedci-iod-app:0.0.1
```

Then run the container:
```
docker run --rm -it -p 8501:8501 -e MODE=HYBRID docker.io/maximilianhahn/fedci-iod-app:0.0.1
```


Modes `CLIENT` and `HYBRID` start streamlit server on port `8501`, which is exposed via the `-p` flag to localhost, therefore allowing access from the local browser under `localhost:8501`.  
Modes `SERVER` and `HYBRID` start a litestar server, running the backend and managing clients. The server runs on port `8000` by default.  

Network connectivity between separate containers, one running as a server and others running as clients requires proper setup and potential docker networking.  
Since no https is supported, you may consider utilizing reverse proxies to encrypt transferred data.  
Also expose port range 16016:16096 for RPCs.

## Known Issues

- Since R requires to be run on the main thread in a lot of circumstances, there can be issues with R code execution on client site, since streamlit delegates threading. This is fixed for fedCI, but meta-analysis may face problems.
- The app can become unstable after long periods of running and multiple user actions, therefore restarts may be required.
- Streamlit components sometimes lose their state, for example, showing the wrong step indicator, although the page loads correctly.
- The server state is fully in-memory, therefore a reboot will reset the server completely



podman run --rm -it -p 8502:8501 -e MODE=CLIENT --name c1 --network fedci-net maximilianhahn/fedci-iod-app:0.0.1
podman run --rm -it -p 8503:8501 -e MODE=CLIENT --name c2 --network fedci-net maximilianhahn/fedci-iod-app:0.0.1

podman run --rm -it -p 8503:8501 -e MODE=CLIENT --name c3 --network --entrypoint /bin/bash fedci-net maximilianhahn/fedci-iod-app:0.0.1

podman run --rm -it -p 8001:8000 -e MODE=SERVER --name s --network fedci-net maximilianhahn/fedci-iod-app:0.0.1
