% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mixedCITest.R
\name{mixedCITest}
\alias{mixedCITest}
\title{Gets the p-value for the conditional independence of X and Y given \eqn{S \cup C},
where C is a fixed set of covariates.}
\usage{
mixedCITest(x, y, S, suffStat)
}
\arguments{
\item{suffStat}{list with the following entries:
dataset: data.frame with all variables that are nodes in the graph
covs: data.frame with a fixed set of covariates that will be
      part of the set S in all conditional independence tests
citestResults: pre-computed conditional independence tests
               in a dataset with columns X, Y, S, and pvalue
symmetric: boolean indicating whether both I(X,Y;S) and I(Y,X;S)
           should be computed
retall: boolean indicating whether only a p-value (retall=FALSE)
        or all computed statistics should be returned (retall=TRUE).
comb_p_method: if "tsagris18" or NULL, then pmm = min(2* min(p1, p2), max(p1, p2)).
               If "min", then pmm = min(p1, p2)
verbose}
}
\description{
Gets the p-value for the conditional independence of X and Y given \eqn{S \cup C},
where C is a fixed set of covariates.
}
