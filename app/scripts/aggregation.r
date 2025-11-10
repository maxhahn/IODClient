library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)

n_cores <- 8
plan("multicore", workers = n_cores, gc=TRUE)

aggregate_ci_results <- function(ci_data, alpha) {
    citestResultsList <- list()
    index <- 1
    for (cur_dat in ci_data) {
      citestResultsList[[index]] <- cur_dat
      index <- index + 1
    }

    suffStat <- list()
    suffStat$citestResultsList <- citestResultsList
    print('HEYYYY')
    print(is.character(citestResultsList[[1]]$labels))
    print(suffStat)
    # call IOD.
    #alpha <- 0.05
    iod_out <- IOD(suffStat, alpha)

    index <- 1
    iod_out$G_PAG_Label_List <- list()
    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)
      index <- index + 1
    }

    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_List) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }
    iod_out
}
