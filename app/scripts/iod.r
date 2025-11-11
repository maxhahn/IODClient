library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)


labelList <- list()

aggregate_ci_results <- function(labelList_, ci_data, alpha, procedure="original") {
    labelList <<- labelList_

    suffStat <- list()
    suffStat$citestResultsList <- ci_data
    suffStat$labelList <- labelList

    labelList <- suffStat$labelList

    # call IOD.
    #alpha <- 0.05
    iod_out <- IOD(labelList, suffStat, alpha, procedure=procedure)

    index <- 1
    iod_out$G_PAG_Label_List <- list()
    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)
      index <- index + 1
    }
    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_list) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }

    iod_out
}

#labelList <- list()
iod_on_ci_data <- function(labelList_, suffStat, alpha, procedure="original") {
    labelList <<- labelList_

    #suffStat$labelList <- labelList
    iod_out <- IOD(labelList, suffStat, alpha, procedure=procedure)

    index <- 1
    iod_out$G_PAG_Label_List <- list()
    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)
      index <- index + 1
    }

    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_list) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }

    iod_out
}
