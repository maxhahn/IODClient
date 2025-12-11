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

get_data <- function(true_pag_amat, num_samples, vars, variable_levels, mode, coef_thresh, seed=NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # 1. Map provided levels to variable names
  var_levels  <- list()
  colnames(true_pag_amat) <- vars
  rownames(true_pag_amat) <- colnames(true_pag_amat)
  cols <- colnames(true_pag_amat)

  for (vari in 1:length(cols)) {
      var_name <- colnames(true_pag_amat)[vari]
      var_levels[[var_name]] <- variable_levels[[vari]]
  }

  # 2. Convert PAG adjacency matrix to canonical DAG
  # dagitty::canonicalize adds Latent nodes (L1, L2, etc.) not present in original data
  adag <- dagitty::canonicalize(getMAG(true_pag_amat)$magg)$g
  dag_cols <- names(adag)

  f.args <- list()

  # --- PASS 1: Initialize Levels for ALL variables first ---
  for (var_name in dag_cols) {
    f.args[[var_name]] <- list()

    # Check if we have levels for this var (Latents L1, L2 etc will be null -> default to 1)
    lvl <- var_levels[[var_name]]
    f.args[[var_name]]$levels <- if (is.null(lvl)) 1 else lvl
  }

  # --- PASS 2: Generate Betas using known levels ---
  for (var_name in dag_cols) {

    # Determine parents using the canonical graph
    parent_names <- parents(adag, var_name)
    betas <- list()

    if (length(parent_names) > 0) {
      for (parent_name in parent_names) {

        # Now we can safely look up the parent's levels because Pass 1 is complete
        k <- f.args[[parent_name]]$levels

        # Logic: If parent is categorical (levels > 2), we need (k-1) coefficients
        # If parent is binary (2) or continuous (1), we need 1 coefficient.
        if (k > 2) {
          # Multinomial parent: generate matrix of (k-1) rows x 1 column
          coefs <- runif(k - 1, min = -1, max = 1)
          while (any(abs(coefs) < coef_thresh)) {
            coefs <- runif(k - 1, min = -1, max = 1)
          }
          beta_matrix <- matrix(coefs, nrow = k - 1, ncol = 1)
          rownames(beta_matrix) <- paste0("L", 2:k)
          colnames(beta_matrix) <- parent_name
          betas[[parent_name]] <- beta_matrix
        } else {
          # Binary/Continuous parent: single scalar
          coef <- runif(1, min = -1, max = 1)
          while (abs(coef) < coef_thresh) {
            coef <- runif(1, min = -1, max = 1)
          }
          betas[[parent_name]] <- coef
        }
      }
    }

    f.args[[var_name]]$betas <- betas
  }

  # Generate data
  dat_out <- FCI.Utils::generateDataset(
    adag = adag,
    N = num_samples,
    type = mode,
    coef_thresh = 0.001,
    f.args = f.args
  )

  return(dat_out)
}

get_datax <- function(true_pag_amat, num_samples, vars, variable_levels, mode, coef_thresh, seed=NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  var_levels  <- list()
  colnames(true_pag_amat) <- vars
  rownames(true_pag_amat) <- colnames(true_pag_amat)
  cols <- colnames(true_pag_amat)
  for (vari in 1:length(cols)) {
      var_name <- colnames(true_pag_amat)[vari]
      print(vari)
      print(colnames(true_pag_amat)[vari])
      print(variable_levels[[vari]])
      var_levels[[var_name]] <- variable_levels[[vari]]
  }

  # Convert PAG adjacency matrix to canonical DAG
  adag <- dagitty::canonicalize(getMAG(true_pag_amat)$magg)$g
  print(adag)
  f.args <- list()
  cols <- names(adag)

  for (var_name in cols) {
    f.args[[var_name]] <- list()
    var_level = var_levels[[var_name]]
    var_level = if (is.null(var_level)) 1 else var_level
    f.args[[var_name]]$levels <- var_level


    # Determine parents
    parent_names <- parents(adag, var_name)

    # Initialize betas
    betas <- list()

    if (length(parent_names) > 0) {
      for (parent_name in parent_names) {
        #k <- f.args[[var_name]]$levels
        k <- f.args[[parent_name]]$levels
        k = if (is.null(k)) 1 else k

        if (k > 2) {
          # Multinomial case: generate matrix of (k-1) rows × 1 column
          coefs <- runif(k - 1, min = -1, max = 1)
          while (any(abs(coefs) < coef_thresh)) {
            coefs <- runif(k - 1, min = -1, max = 1)
          }
          beta_matrix <- matrix(coefs, nrow = k - 1, ncol = 1)
          rownames(beta_matrix) <- paste0("L", 2:k)  # optional
          colnames(beta_matrix) <- parent_name
          betas[[parent_name]] <- beta_matrix
        } else {
          # Binary or continuous case: single scalar
          coef <- runif(1, min = -1, max = 1)
          while (abs(coef) < coef_thresh) {
            coef <- runif(1, min = -1, max = 1)
          }
          betas[[parent_name]] <- coef
        }
      }
    }

    # Always set betas, even if empty
    f.args[[var_name]]$betas <- betas
  }
  print(f.args)
  print(paste("Variable levels:", paste(names(var_levels), "=", var_levels, collapse=", ")))
  # Generate data
  dat_out <- FCI.Utils::generateDataset(
    adag = adag,
    N = num_samples,
    type = mode,
    coef_thresh = 0.001,
    f.args = f.args
  )

  return(dat_out)
}


get_data_for_single_pag <- function(num_samples, variable_levels, mode, coef_thresh, seed) {
  true.amat.pag <- t(matrix(c(0,0,2,2,0,3,
                             0,0,2,0,0,3,
                             2,1,0,2,2,3,
                             2,0,3,0,2,3,
                             0,0,3,3,0,0,
                             2,2,2,2,2,0), 6, 6))
  vars <- c("A", "B", "C", "D", "E", "CLIENT")
  colnames(true.amat.pag) <- vars
  rownames(true.amat.pag) <- colnames(true.amat.pag)

  return(get_data(true.amat.pag, num_samples, vars, variable_levels, mode, coef_thresh, seed))
}


msep <- function(true_pag_amat, vars, x, y, s) {
    colnames(true_pag_amat) <- vars
    rownames(true_pag_amat) <- colnames(true_pag_amat)
    result <- isMSeparated(true_pag_amat, x, y, s)
    result
}

msep_for_single_pag <- function(x, y, s) {
    true.amat.pag <- t(matrix(c(0,0,2,2,0,
                              0,0,2,0,0,
                              2,1,0,2,2,
                              2,0,3,0,2,
                              0,0,3,3,0), 5, 5))
    colnames(true.amat.pag) <- c("A", "B", "C", "D", "E")
    rownames(true.amat.pag) <- colnames(true.amat.pag)
    result <- isMSeparated(true.amat.pag, x, y, s)
    result
}

get_slide_pag <- function(x, y, s) {
    true.amat.pag <- t(matrix(c(0,0,2,2,0,
                              0,0,2,0,0,
                              2,1,0,2,2,
                              2,0,3,0,2,
                              0,0,3,3,0), 5, 5))
    colnames(true.amat.pag) <- c("A", "B", "C", "D", "E")
    rownames(true.amat.pag) <- colnames(true.amat.pag)
    true.amat.pag
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

run_ci_test2 <- function(x,y,s,data) {
    data[] <- lapply(data, function(col) {
        if (is.integer(col)) {
            factor(col, levels = sort(unique(col)), ordered = TRUE)
        } else {
            col
        }
    })
    r <- MXM::ci.mm(x,y,s,data)
    r
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

iod_on_ci_data <- function(true_pag_amat, true_pag_cols, labelList_, suffStat, alpha, procedure) {
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

load_pags <- function() {
    load("100randomPAGs.RData")
    #c(truePAGs, subsetsList)
    #tuple <- list(A, B)
    return(list(truePAGs = truePAGs, subsetsList = subsetsList))
}
