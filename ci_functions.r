library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)

#n_cores <- 8
#plan("multicore", workers = n_cores, gc=TRUE)
#options(error = function() traceback(3))
get_data <- function(true_pag_amat, num_samples, variable_levels, mode) {
  #print(true_pag_amat)
  #print(type(true_pag_amat))
  #print(class(true_pag_amat))  # Should be "matrix"
  #print(dim(true_pag_amat))    # Should show (5,5) in your case
  #print(attributes(true_pag_amat))  # Should only show "dim"

  #adag <- dagitty::canonicalize(getMAG(true_pag_amat)$magg)$g
  #print(adag)
  #print(class(adag))  # Should be "dagitty"
  #print(str(adag))    # See internal structure


  f.args <- list()
  cols <- colnames(true_pag_amat)
  for (vari in 1:length(cols)) {
      var_name <- colnames(true_pag_amat)[vari]
      f.args[[var_name]] <- list(levels = variable_levels[[vari]])
    }

  dat_out <- FCI.Utils::generateDatasetFromPAG(apag = true_pag_amat,
    N=num_samples,
    type = mode,#"continuous",#type = "mixed",
    f.args = f.args
  )
  dat_out
}

run_ci_test <- function(data, max_cond_set_cardinality, filedir, filename) {
  labels <- colnames(data)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = data,
                                 vars_names = labels,
                                 covs_names = c())
  citestResults <- getAllCITestResults(data,
                                      indepTest,
                                      suffStat,
                                      m.max=max_cond_set_cardinality,
                                      saveFiles=FALSE,
                                      fileid=filename,
                                      citestResults_folder=filedir)
  result <- list(citestResults=citestResults, labels=labels)
  result
}


labelList <- list()

aggregate_ci_results <- function(true_pag_amat, true_pag_cols, labelList_, ci_data, alpha, procedure) {
    labelList <<- labelList_

    colnames(true_pag_amat) <- true_pag_cols
    rownames(true_pag_amat) <- colnames(true_pag_amat)

    suffStat <- list()
    suffStat$citestResultsList <- ci_data
    suffStat$labelList <- labelList

    # call IOD.
    #alpha <- 0.05
    iod_out <- IOD(labelList, suffStat, alpha, procedure=procedure)
    index <- 1
    iod_out$G_PAG_Label_List <- list()
    iod_out$G_PAG_SHD <- list()
    iod_out$G_PAG_FDR <- list()
    iod_out$G_PAG_FOR <- list()
    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)

      posneg_metrics <- getPAGPosNegMetrics(true_pag_amat, gpag)
      iod_out$G_PAG_SHD[[index]] <- shd_PAG(true_pag_amat, gpag)
      iod_out$G_PAG_FDR[[index]] <- posneg_metrics$false_discovery_rate
      iod_out$G_PAG_FOR[[index]] <- posneg_metrics$false_omission_rate

      index <- index + 1
    }
    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_List) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }

    iod_out$found_correct_pag = containsTheTrueGraph(trueAdjM = true_pag_amat, iod_out$G_PAG_List)

    iod_out
}

iod_on_ci_data <- function(true_pag_amat, true_pag_cols, labelList_, suffStat, alpha, procedure) {
    labelList <<- labelList_

    colnames(true_pag_amat) <- true_pag_cols
    rownames(true_pag_amat) <- colnames(true_pag_amat)

    suffStat$labelList <- labelList
    iod_out <- IOD(labelList, suffStat, alpha, procedure)

    index <- 1
    iod_out$G_PAG_Label_List <- list()
    iod_out$G_PAG_SHD <- list()
    iod_out$G_PAG_FDR <- list()
    iod_out$G_PAG_FOR <- list()

    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)

      posneg_metrics <- getPAGPosNegMetrics(true_pag_amat, gpag)
      iod_out$G_PAG_SHD[[index]] <- shd_PAG(true_pag_amat, gpag)
      iod_out$G_PAG_FDR[[index]] <- posneg_metrics$false_discovery_rate
      iod_out$G_PAG_FOR[[index]] <- posneg_metrics$false_omission_rate

      index <- index + 1
    }

    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_List) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }

    iod_out$found_correct_pag = containsTheTrueGraph(trueAdjM = true_pag_amat, iod_out$G_PAG_List)

    iod_out
}

load_pags <- function() {
    load("100randomPAGs.RData")
    #c(truePAGs, subsetsList)
    #tuple <- list(A, B)
    return(list(truePAGs = truePAGs, subsetsList = subsetsList))
}
