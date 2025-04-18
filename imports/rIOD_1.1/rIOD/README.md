## A privacy-preserving implementation in R of the IOD algorithm

### Overview

This package provides an implementation of the IOD algorithm by Tillman and Spirtes [2011] (http://proceedings.mlr.press/v15/tillman11a/tillman11a.pdf).

The IOD learns equivalence classes of acyclic models  
with latent and selection variables from multiple datasets with overlapping variables. 
It outputs a list of PAGs including the true PAG, if the combined statistics are faithful. 

### Reference Manual

Documentation of the methods provided by ... can be found at: ...

### Example

An example is provided at ...

### Installation

First, install R (>= 3.5.0) and the following packages:
```r
install.packages(c("FCI.Utils", "pcalg", "igraph", "RBGL", "graph", "doFuture", "gtools", "MXM", "pscl", "DOT", "rsvg", "doSNOW"), dependencies=TRUE)
```
You can download the latest tar.gz file with the source code of the IOD R package, available at <https://github.com/adele/IOD/releases/latest>, and install it with the following command, where `path_to_file` represents the full path and file name of the tar.gz file:

``` r
install.packages(path_to_file, repos=NULL, type="source", dependencies=TRUE)
```

Or you can install the development version directly from GitHub. Make sure you have the devtools R package installed. If not, install it with `install.packages("devtools", dependencies=TRUE)`.

``` r
devtools::install_github("adele/rIOD", dependencies=TRUE)
```

Note: if you are asked to update packages, then press "a" for all.

All releases are available at <https://github.com/adele/rIOD/releases/>. If you want a specific version of the IOD R package, for example, v1.0, you can install it directly from the URL:

``` r
install.packages("https://github.com/adele/rIOD/releases/download/v1.0/rIOD_1.0.tar.gz", repos=NULL, method="libcurl", dependencies=TRUE)
```
