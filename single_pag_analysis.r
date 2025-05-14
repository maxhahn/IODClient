library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)
library(dagitty)


labelList <- list()

aggregate_ci_results <- function(labelList_, ci_data, alpha, procedure) {
  true_pag_amat <- t(matrix(c(0,0,2,2,0,
                             0,0,2,0,0,
                             2,1,0,2,2,
                             2,0,3,0,2,
                             0,0,3,3,0), 5, 5))
  true_pag_cols <- c("A", "B", "C", "D", "E")
  colnames(true_pag_amat) <- true_pag_cols
  rownames(true_pag_amat) <- true_pag_cols


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

  #print(true_pag_amat)
  #print(iod_out$G_PAG_List)

  is_in_list <- FALSE
  col_order = colnames(true_pag_amat)
  for (adj_matrix in iod_out$G_PAG_List) {
  #make sure that trueAdjM and theadj_matrix have the same sequence of vars
  #new_order <- colnames(adj_matrix)
  adj_matrix = adj_matrix[col_order, col_order]
  #trueAdjM <- trueAdjM[new_order, new_order]
  if (all(adj_matrix == true_pag_amat)) {
      is_in_list <- TRUE
      break
    }
  }

  iod_out$found_correct_pag = is_in_list

  #iod_out$found_correct_pag = containsTheTrueGraph(trueAdjM = true_pag_amat, iod_out$G_PAG_List)

  #print(iod_out$found_correct_pag)

  iod_out
}

iod_on_ci_data <- function(labelList_, suffStat, alpha, procedure) {
  true_pag_amat <- t(matrix(c(0,0,2,2,0,
                             0,0,2,0,0,
                             2,1,0,2,2,
                             2,0,3,0,2,
                             0,0,3,3,0), 5, 5))
  true_pag_cols <- c("A", "B", "C", "D", "E")
  colnames(true_pag_amat) <- true_pag_cols
  rownames(true_pag_amat) <- true_pag_cols

  labelList <<- labelList_

  colnames(true_pag_amat) <- true_pag_cols
  rownames(true_pag_amat) <- colnames(true_pag_amat)

  suffStat$labelList <- labelList
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

  #print(true_pag_amat)
  #print(iod_out$G_PAG_List)

  is_in_list <- FALSE
  col_order = colnames(true_pag_amat)
  for (adj_matrix in iod_out$G_PAG_List) {
  #make sure that trueAdjM and theadj_matrix have the same sequence of vars
  #new_order <- colnames(adj_matrix)
  adj_matrix = adj_matrix[col_order, col_order]
  #trueAdjM <- trueAdjM[new_order, new_order]
  #print(true_pag_amat)
  #print(adj_matrix)
  #print('---')
  if (all(adj_matrix == true_pag_amat)) {
      is_in_list <- TRUE
      break
    }
  }

  iod_out$found_correct_pag = is_in_list

  #iod_out$found_correct_pag = containsTheTrueGraph(trueAdjM = true_pag_amat, iod_out$G_PAG_List)

  #print(iod_out$found_correct_pag)
  iod_out
}

run_fci <- function(dataset, obs_vars, alpha=0.05) {

  indepTest <- mixedCITest

  suffStat <- getMixedCISuffStat(dat = dataset,
                                   vars_names = obs_vars,
                                   covs_names = c())

  citestResults <- getAllCITestResults(dataset, indepTest, suffStat)


  estimated_pag <- pcalg::fci(suffStat,
                                indepTest = indepTest,
                                labels=obs_vars, alpha = alpha,
                                verbose = TRUE)

  #renderAG(estimated_pag@amat)

  estimated_pag@amat
}

run_ci_test <- function(data, max_cond_set_cardinality, filedir, filename) {
  data[] <- lapply(data, function(col) {
      if (is.integer(col)) {
          factor(col, levels = sort(unique(col)), ordered = TRUE)
      } else {
          col
      }
  })
  labels <- colnames(data)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = data,
                                 vars_names = labels,
                                 covs_names = c())
  suffStat$verbose <- TRUE
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
