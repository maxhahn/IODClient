createFakeData <- function(true.pag.amat, setsToConsider) {
  data <- list()  # Create an empty list to store data frames
  for (i in 1:length(setsToConsider)) {
    cur_vars <- setsToConsider[[i]]
    if (length(cur_vars) > 0) {
      data[[i]] <- data.frame(matrix(nrow = 0, ncol = length(cur_vars)))
      colnames(data[[i]]) <- unlist(cur_vars)
    }
  }
  return(data)
}

# computing p-values for each dataset separately
# and saving then in a table, citestResults
# citest_type = {"mixedCI", "oracleCI"}
getCIResultsList <- function(citest_type = "mixedCI",
                             data=NULL, setsToConsider=NULL,
                             true.pag.amat=NULL) {
  citestResultsList <- list()
  labelList <- list()
  covs_names <- c() # leave it empty for now

  #TODO check if oracleCI and true.pag.amat != NULL or
  # if not oracle, then data != NULL

  if (citest_type == "oracleCI") {
    indepTest <- dagittyCIOracle2
    true.magg <- getMAG(true.pag.amat)$magg
    suffStat <- list(g=true.magg, labels=colnames(true.pag.amat))

    if (is.null(setsToConsider)) {
      vars <- colnames(true.pag.amat)
      n <- length(vars)
      allcombs <- mycombn(vars,(n-2))
      length_allcombs <- length(allcombs)/(n-2)
      setsToConsider <- list(allcombs[,1], allcombs[,length_allcombs*6/10],
                             allcombs[,length_allcombs*3/10], allcombs[,length_allcombs])
    }

    data <- createFakeData(true.pag.amat, setsToConsider)
    all_obs_vars <- colnames(getMAG(true.pag.amat)$amat.mag)
    #all_obs_vars <- unique(unlist(lapply(data, colnames)))
    fake_full_data <- data.frame(matrix(nrow = 0, ncol = length(all_obs_vars)))
    colnames(fake_full_data) <- all_obs_vars
    allCITests <- getAllCITestResults(fake_full_data, indepTest, suffStat)

    for (i in 1:length(data)) {
      cur_dat <-  data[[i]]
      cur_labels <- colnames(cur_dat)
      suffStat$cur_labels <- cur_labels
      citestResults <- extractValidCITestResults(allCITests, all_obs_vars, cur_labels)
      citestResultsList[[i]] <- citestResults
      labelList[[i]] <- cur_labels
    }
  } else {
    for (i in 1:length(data)) {
      cur_dat <-  data[[i]]
      cur_labels <- colnames(cur_dat)
      print(cur_labels)
      print(cur_dat)
      if (citest_type != "oracleCI") {
        cur_dat <-  data[[i]]
        cur_labels <- colnames(cur_dat)
        indepTest <- mixedCITest
        suffStat <- getMixedCISuffStat(dat = cur_dat,
                                       vars_names = cur_labels,
                                       covs_names = covs_names)
      }
      suffStat$cur_labels <- cur_labels
      citestResults <- getAllCITestResults(cur_dat, indepTest, suffStat)
      citestResultsList[[i]] <- citestResults
      labelList[[i]] <- cur_labels
      #alpha <- 0.05
      #subset(citestResultsList[[i]]$citestResults, pvalue > alpha)
    }
  }
  return(list(citestResultsList=citestResultsList, labelList=labelList))
}

getRandomMAG <- function(n_nodes, dir_edges_prob = 0.4, bidir_edges_prob = 0.2) {
  done = FALSE
  while(!done) {
    amat.mag <- matrix(0, nrow = n_nodes, ncol=n_nodes)
    colnames(amat.mag) <- rownames(amat.mag) <- LETTERS[seq( from = 1, to = n_nodes)]

    edges <- combn(1:n_nodes, 2)
    n_edges <- dim(edges)[2]
    dir_edges <- sample(1:n_edges, floor(n_edges * dir_edges_prob), replace = FALSE)
    for (i in dir_edges) {
      amat.mag[edges[1,i], edges[2,i]] <- 2
      amat.mag[edges[2,i], edges[1,i]] <- 3
    }

    bidir_edges <- sample((1:n_edges)[-dir_edges], floor(n_edges * bidir_edges_prob), replace = FALSE)
    for (i in bidir_edges) {
      amat.mag[edges[1,i], edges[2,i]] <- 2
      amat.mag[edges[2,i], edges[1,i]] <- 2
    }

    if (isAncestralGraph(amat.mag)) {
      done = TRUE
    }
  }
  return(amat.mag)
}


generateUniqueRandomPAGsSubsets <- function(n_graphs = 100, n_nodes = 5,
                                     n_subsets = 2, size_subsets = n_nodes-1,
                                     dir_edges_prob = 0.4, bidir_edges_prob = 0.2,
                                     verbose=FALSE) {
  truePAGs <- list()
  subsets <- list()
  stats <- c()

  while (length(truePAGs) < n_graphs) {
    amat.mag <- getRandomMAG(n_nodes, dir_edges_prob = 0.4, bidir_edges_prob = 0.2)
    labels <- colnames(amat.mag)
    mec <- MAGtoMEC(amat.mag, verbose=verbose)
    if ((length(mec$CK) > 0 && length(which(mec$CK$ord >= 1)) > 0)) { ## || (length(mec$NCK) > 0)) {

      #if (verbose) {
        cat("PAG", length(truePAGs), "with nCK1:", length(which(mec$CK$ord >= 1)), "and nNCK", nrow(mec$NCK), "\n")
      #}

      stats <- rbind(stats, c(nCK1 = length(which(mec$CK$ord >= 1)), nNCK = nrow(mec$NCK)))

      #renderAG(amat.mag)
      amag <- pcalg::pcalg2dagitty(amat.mag, colnames(amat.mag), type="mag")
      truePAG <- getTruePAG(amag)
      amat.pag <- truePAG@amat
      #renderAG(amat.pag)
      truePAGs[[length(truePAGs) + 1]] <- amat.pag
      truePAGs <- unique(truePAGs)

      # get subsets of the variables forming the discrimating paths or, alternatively
      # random subsets such that the overlapping includes at least n-2 variables.
      ordered_triplets <- rbind(mec$NCK, mec$CK)
      high_ordered_triplets <- subset(ordered_triplets, ord >= 1 & ord+3 <= size_subsets)
      cur_subsets <- list()
      for (i in 1:nrow(high_ordered_triplets)) {
        cur_subsets <- c(cur_subsets, list(labels[getSepVector(high_ordered_triplets$path[i])]))
        if (length(cur_subsets) == n_subsets)
          break
      }

      if (length(cur_subsets) < n_subsets) {
        combs <- t(mycombn(labels, length(labels)-1))

        valid_ids <- apply(combs, 1, function(x) { sapply(cur_subsets, function(y) {
          a = length(which(x %in% y)); a < size_subsets & a >= size_subsets -2 })})
        if (!is.null(dim(valid_ids))) {
          valid_ids <- which(apply(valid_ids, 2, all))
        }

        n_add_subsets <- length((length(cur_subsets)+1):n_subsets)

        if (length(valid_ids) < n_add_subsets) {
          valid_ids <- !apply(combs, 1, function(x) {
            sapply(cur_subsets, function(y) { all(x %in% y) })})
          valid_ids <- which(apply(valid_ids, 2, all))
        }

        combs <- as.data.frame(t(combs[valid_ids,]))

        subsets_ids <- sample(1:ncol(combs), n_add_subsets, replace = F)
        cur_subsets <- c(cur_subsets, as.list(combs[, subsets_ids, drop=FALSE]))
        names(cur_subsets) <- NULL
      }

      subsets[[length(truePAGs)]] <- cur_subsets
    }
  }
  return(list(pags=truePAGs, subsets=subsets))
}

generateDatasetsSuffStats <- function(truePAGs, subsetsList, Nvec, data_type="continuous") {
  if (length(subsetsList[[1]]) != length(Nvec)) {
    stop("Nvec must have the same length as each entry of the subsetsList")
  }

  all_datasets <- list()
  all_suffstats <- list()

  out_list <- foreach (i = 1:length(truePAGs),
                       .verbose=TRUE, .options.future = list(seed = TRUE)) %dofuture% {
    cur_pag <- truePAGs[[i]]
    adag <- dagitty::canonicalize(FCI.Utils::getMAG(cur_pag)$magg)$g

    cur_subsets <- subsetsList[[i]]

    allvars <- colnames(cur_pag)

    cur_datasets <- list()
    for (j in 1:length(Nvec)) {
      cur_full_dat <- FCI.Utils::generateDataset(adag = adag, N = Nvec[j],
                                                 type=data_type)$dat
      cur_datasets[[j]] <- cur_full_dat[cur_subsets[[j]]]
    }
    #lapply(cur_datasets, head)

    suffStat <- list()
    citestResultsList_out <- getCIResultsList(citest_type="mixedCI",
                                                   data=cur_datasets)
    suffStat$citestResultsList <- citestResultsList_out$citestResultsList
    suffStat$labelList <- citestResultsList_out$labelList

    cur_unfaithf_scs <- list()


    list(datasets=cur_datasets, suffStat=suffStat)
  }
  return(out_list)
}


getRandomGraph <- function(n, e_prob, l_prob, force_nedges=TRUE,
                           min_nedges=NULL, max_nedges=NULL)  {
  if (force_nedges) {
    if (is.null(min_nedges)) {
      min_nedges <- floor((n*(n-1)/2)*(e_prob * 1))
    }

    if (is.null(max_nedges)) {
      max_nedges <- ceiling((n*(n-1)/2)*(e_prob * 1))
    }
  } else {
    min_nedges <- 0
    max_nedges <- n*(n-1)/2
  }

  nedges <- 0
  vars <- c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
  while (!is.null(nedges) && (nedges < min_nedges || nedges > max_nedges)) {
    n_lat <- ceiling(n * l_prob)
    n_tot <- n + n_lat
    dag <- pcalg::randomDAG(n_tot, e_prob, V = vars[1:n_tot])

    #V = paste0("V", sprintf(paste0("%0", nchar(n), "d"), seq(from = 1, to = n)))
    amat.dag <- (as(dag, "matrix") > 0) * 1
    adag <- pcalg::pcalg2dagitty(amat.dag, colnames(amat.dag), type="dag")
    lat <- sample(colnames(amat.dag), n_lat)

    dagitty::latents(adag) <- lat

    magg <- dagitty::toMAG(adag)
    nedges <- nrow(edges(magg))
  }

  return(adag)
}


################################################################################
# outputs a list of PAGs that are in the first list but not in the second
getDiffPAGList <- function(pag_List1, pag_List2) {
  diff_pag_list <- list()

  i = 1
  for (cur_pag in pag_List1) {
    if (!any(sapply(pag_List2, function(x) {
      isTRUE(all.equal(x, cur_pag))
    }))) {
      diff_pag_list[[i]] <- cur_pag
      i = i+1
    }
  }

  return(diff_pag_list)
}

##################################################################################
procedeIODWithGraphs <- function(graphs, subsets, output_folder=NULL, fileid = "graph",
                                 suffStats = NULL) {

  if (!is.null(output_folder) && !file.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }

  foreach (i = 1:length(graphs), .verbose=TRUE) %dofuture% {
    cat("Test:", i)

    trueAdjM <- graphs[[i]]
    allvars <- colnames(trueAdjM)
    setsToConsider <- subsets[[i]]

    if (is.null(suffStats)) {
      suffStat <- list()
      citestResultsList_out <-  getCIResultsList(citest_type="oracleCI",
                                                      setsToConsider = setsToConsider,
                                                      true.pag.amat = trueAdjM)
      suffStat$citestResultsList <- citestResultsList_out$citestResultsList
      suffStat$labelList <- citestResultsList_out$labelList
    } else {
      suffStat <- suffStats[[i]]
    }

    iod_out_Pags <-list()
    iod_out_Gi <- list()
    labelList <- suffStat$labelList

    start.time <- Sys.time()
    res_orig <- IOD(labelList, suffStat)
    end.time <- Sys.time()
    res_orig$runtime <- end.time - start.time
    res_orig$hasTruePAG <- containsTheTrueGraph(trueAdjM, res_orig$G_PAG_List)
    res_orig$margIncons <-  allPAGsincludeAllInvariancesFromGis(res_orig$Gi_PAG_list, res_orig$G_PAG_List) # marginal inconsistencies
    res_orig$listbefhasTruePAG <- containsTheTrueGraph(trueAdjM, res_orig$G_PAG_List_before)

    start.time <- Sys.time()
    res_cwo <- IOD(labelList, suffStat, procedure = "orderedcolls")
    end.time <- Sys.time()
    res_cwo$runtime <- end.time - start.time
    res_cwo$hasTruePAG <- containsTheTrueGraph(trueAdjM, res_cwo$G_PAG_List)
    res_cwo$margIncons <-  allPAGsincludeAllInvariancesFromGis(res_cwo$Gi_PAG_list, res_cwo$G_PAG_List) # marginal inconsistencies
    res_cwo$listbefhasTruePAG <- containsTheTrueGraph(trueAdjM, res_cwo$G_PAG_List_before)

    start.time <- Sys.time()
    res_ncwo <- IOD(labelList, suffStat, procedure = "orderedtriplets")
    end.time <- Sys.time()
    res_ncwo$runtime <- end.time - start.time
    res_ncwo$hasTruePAG <- containsTheTrueGraph(trueAdjM, res_ncwo$G_PAG_List)
    res_ncwo$margIncons <-  allPAGsincludeAllInvariancesFromGis(res_ncwo$Gi_PAG_list, res_ncwo$G_PAG_List) # marginal inconsistencies
    res_ncwo$listbefhasTruePAG <- containsTheTrueGraph(trueAdjM, res_ncwo$G_PAG_List_before)



    res <- list(orig=res_orig, cwo=res_cwo, ncwo=res_ncwo, trueAdjM=trueAdjM, setsToConsider=setsToConsider)

    if (!is.null(output_folder)) {
      save(res, file=paste0(output_folder, fileid, "_", sprintf("%03d",i) ,
                          "_cwo", res_ncwo$nCK1, "_ncwo", res_ncwo$nNCK,
                          "_len", res_orig$len_before, "_", res_cwo$len_before, "_", res_ncwo$len_before,
                          "_n", length(res_orig$G_PAG_List), "_", length(res_cwo$G_PAG_List), "_", length(res_ncwo$G_PAG_List),
                          ".RData"))
    }
  }
}

getStatistics <- function(graphs, results_files, output_folder=NULL, fileid=NULL) {
  results <- c()
  i <- 1

  for (cur_file in results_files[1:length(results_files)]) {
    load(cur_file)

    truePag <- graphs[[i]]
    i <- i + 1
    shd_org <-  lapply(res$orig$G_PAG_List, function(pag) shd_PAG(pag, truePag))
    shd_cwo <-  lapply(res$cwo$G_PAG_List, function(pag) shd_PAG(pag, truePag))
    shd_ncwo <-  lapply(res$ncwo$G_PAG_List, function(pag) shd_PAG(pag, truePag))

    posneg_org <-  lapply(res$orig$G_PAG_List, function(pag) getPAGPosNegMetrics(pag, truePag))
    posneg_cwo <-  lapply(res$cwo$G_PAG_List, function(pag) getPAGPosNegMetrics(pag, truePag))
    posneg_ncwo <-  lapply(res$ncwo$G_PAG_List, function(pag) getPAGPosNegMetrics(pag, truePag))

    posneg_org_fdr <- lapply(posneg_org, function(x) x$false_discovery_rate)
    posneg_cwo_fdr <- lapply(posneg_cwo, function(x) x$false_discovery_rate)
    posneg_ncwo_fdr <-  lapply(posneg_ncwo, function(x) x$false_discovery_rate)

    posneg_org_for <- lapply(posneg_org, function(x) x$false_omission_rate)
    posneg_cwo_for <- lapply(posneg_cwo, function(x) x$false_omission_rate)
    posneg_ncwo_for <-  lapply(posneg_ncwo, function(x) x$false_omission_rate)


    results <- rbind(results,
                     c(nCK1= res$ncwo$nCK1,
                     nNCK = res$ncwo$nNCK,
                     orig.lenbef = res$orig$len_before,
                     cwo.lenbef = res$cwo$len_before,
                     ncwo.lenbef = res$ncwo$len_before,
                     orig.runtime = res$orig$runtime,
                     cwo.runtime = res$cwo$runtime,
                     ncwo.runtime = res$ncwo$runtime,
                     orig.npags = length(res$orig$G_PAG_List),
                     cwo.npags = length(res$cwo$G_PAG_List),
                     ncwo.npags = length(res$ncwo$G_PAG_List),
                     orig.true = res$orig$hasTruePAG,
                     cwo.true = res$cwo$hasTruePAG,
                     ncwo.true = res$ncwo$hasTruePAG,
                     orig.true.bf = res$orig$listbefhasTruePAG,
                     cwo.true.bf = res$cwo$listbefhasTruePAG,
                     ncwo.true.bf = res$ncwo$listbefhasTruePAG,
                     orig.margIncons = length(res$orig$margIncons),
                     cwo.margIncons = length(res$cwo$margIncons),
                     ncow.margIncons = length(res$ncwo$margIncons),




                     shd_org = list(shd_org),
                     shd_cwo = list(shd_cwo),
                     shd_ncwo = list(shd_ncwo),
                     posneg_org_fdr = list(posneg_org_fdr),
                     posneg_cwo_fdr = list(posneg_cwo_fdr),
                     posneg_ncwo_fdr = list(posneg_ncwo_fdr),
                     posneg_org_for = list(posneg_org_for),
                     posneg_cwo_for = list(posneg_cwo_for),
                     posneg_ncwo_for = list(posneg_ncwo_for)))
  }
  results <- as.data.frame(results)

  if (!is.null(output_folder)) {
    save(results, file=paste0(output_folder, "results_", fileid, ".RData"))
  }
}

mycombn <- function(x, m) {
  if (length(x) == 1) {
    return(combn(list(x),m))
  } else {
    return(combn(x,m))
  }
}

dagittyCIOracle2 <- function(x, y, S, suffStat) {
  g <- suffStat$g
  labels <- suffStat$labels
  if (dagitty::dseparated(g, labels[x], labels[y], labels[S])) {
    return(1)
  } else {
    return(0)
  }
}

allPAGsincludeAllInvariancesFromGis <- function(listGi, listPags){
  tracker <- list()
  i <- 1
  for (pag in listPags) {
    for (gi in listGi) {
      apag <- getAncestralMatrix(pag)
      agi <- getAncestralMatrix(gi)
      relevant_vars <- colnames(agi)
      for (var_col in  relevant_vars) {
        for (var_row in  relevant_vars) {
          if (var_col != var_row) {
            value_subset <- agi[var_col,var_row] # relation in Gi
            if (!(apag[var_col,var_row] == value_subset) && ## the relations in G and Gi do not match
                value_subset != 2) { ## and it is not due to an undetermination (circle) in Gi
              tracker[[i]] <- list(pag,gi)
              i <- i+1
            }
          }
        }
      }
    }
  }
  return(unique(tracker))
}
