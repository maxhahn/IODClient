#' #' @importFrom FCI.Utils extractValidCITestResults
#' #' @importFrom stats pchisq
#' #' @noRd
#' iodCITestOLD <- function(x, y, S, suffStat) {
#'   xname <- suffStat$cur_labels[x]
#'   yname <- suffStat$cur_labels[y]
#'   snames <- suffStat$cur_labels[S]
#'
#'   citestResultsList <- suffStat$citestResultsList
#'   k <- 0
#'   n_datasets <- length(citestResultsList)
#'   p <- rep(1, n_datasets)
#'
#'   for (i in 1:n_datasets) {
#'     labels_i <- citestResultsList[[i]]$labels
#'     required_labels <- c(xname, yname, snames)
#'     if (all(required_labels %in% labels_i)) {
#'
#'       citestResults_i <- citestResultsList[[i]]$citestResults
#'       required_results <- extractValidCITestResults(citestResults_i,
#'                                                     labels_i, required_labels)
#'       colnames(required_results) <- c("ord", "X", "Y", "S", "pvalue")
#'       required_results[, c(1:3,5)] <- lapply(required_results[, c(1:3,5)], as.numeric)
#'
#'       x <- which(required_labels == xname) # should be always 1
#'       y <- which(required_labels == yname) # should be always 2
#'       S <- which(required_labels %in% snames) # should be always 3:length(required_labels)
#'       sStr <- getSepString(S)
#'       resultsxys <- c()
#'       if (!is.null(required_results)) {
#'         # the test should be the symmetric for X,Y|S and Y,X|S
#'         resultsxys <- subset(required_results, X == x & Y == y & S == sStr)
#'         resultsyxs <- subset(required_results, X == y & Y == x & S == sStr)
#'         resultsxys <- rbind(resultsxys, resultsyxs)
#'       }
#'       p[i] <- resultsxys[1, "pvalue"]
#'       k <- k+1
#'     } else {
#'       p[i] <- 1
#'     }
#'   }
#'   test_statistic <- -2 * sum(log(p))
#'   df <- 2*k
#'
#'   p_value <- pchisq(test_statistic, df, lower.tail = FALSE) # H0: Independency
#'
#'   return(p_value)
#' }

# suffStat$all_labels
# suffStat$citestResults

# suffStat$citestResultsList


#' @importFrom FCI.Utils extractValidCITestResults
#' @importFrom stats pchisq
#' @noRd
iodCITest <- function(x, y, S, suffStat) {
  xname <- suffStat$cur_labels[x]
  yname <- suffStat$cur_labels[y]
  snames <- suffStat$cur_labels[S]

  if (!is.null(suffStat$citestResults) && !is.null(suffStat$all_labels)) {
    xind <- which(suffStat$all_labels == xname)
    yind <- which(suffStat$all_labels == yname)
    Sinds <- which(suffStat$all_labels %in% snames)


    SxyStr <- getSepString(sort(Sinds))
    resultsxys <- c()
    if (!is.null(suffStat$citestResults)) {
      resultsxys <- subset(suffStat$citestResults, X == xind & Y == yind & S == SxyStr)
      resultsyxs <- subset(suffStat$citestResults, X == yind & Y == xind & S == SxyStr)
      resultsxys <- rbind(resultsxys, resultsyxs)
    }

    if (!is.null(resultsxys) && nrow(resultsxys) > 0) {
      #   cat("Returning pre-computed p-value for  X=", x, "; Y=", y,
      #       "given S={", paste0(S, collapse = ","), "}\n")

      # the test is expected to be the symmetric for X,Y|S and Y,X|S
      p_value <- resultsxys[1, "pvalue"]
    } else {
      p_value = 1
    }
  } else if (!is.null(suffStat$citestResultsList) && !is.null(suffStat$labelList)) {
    k <- 0
    citestResultsList <- suffStat$citestResultsList
    n_datasets <- length(suffStat$citestResultsList)
    p <- rep(1, n_datasets)

    for (i in 1:n_datasets) {
      labels_i <- suffStat$labelList[[i]]
      required_labels <- c(xname, yname, snames)
      if (all(required_labels %in% labels_i)) {

        citestResults_i <- citestResultsList[[i]]
        required_results <- extractValidCITestResults(citestResults_i,
                                                      labels_i, required_labels)
        colnames(required_results) <- c("ord", "X", "Y", "S", "pvalue")
        required_results[, c(1:3,5)] <- lapply(required_results[, c(1:3,5)], as.numeric)

        x <- which(required_labels == xname) # should be always 1
        y <- which(required_labels == yname) # should be always 2
        S <- which(required_labels %in% snames) # should be always 3:length(required_labels)
        sStr <- getSepString(S)
        resultsxys <- c()
        if (!is.null(required_results)) {
          # the test should be the symmetric for X,Y|S and Y,X|S
          resultsxys <- subset(required_results, X == x & Y == y & S == sStr)
          resultsyxs <- subset(required_results, X == y & Y == x & S == sStr)
          resultsxys <- rbind(resultsxys, resultsyxs)
        }
        p[i] <- resultsxys[1, "pvalue"]
        k <- k+1
      } else {
        p[i] <- 1
      }
    }
    test_statistic <- -2 * sum(log(p))
    df <- 2*k

    p_value <- pchisq(test_statistic, df, lower.tail = FALSE) # H0: Independency
  } else {
    stop("Error: suffStat does not include the expected information.")
  }

  return(p_value)
}


# Initialize G with edges between all nodes
#' @noRd
initG <-function(labelList) {
  labelsG <- collectLabelsG(labelList)
  #print(labelsG)
  p <- length(labelsG)
  #print(p)
  G <- matrix(1, nrow = p, ncol = p)
  G <- G - diag(p)
  colnames(G) <- rownames(G) <- labelsG
  return(G)
}

#' @noRd
collectLabelsG <- function(labelList) {
  n_datasets <- length(labelList)
  labelsG <- list()
  for (i in 1:n_datasets){
    labelsG[[i]] <- labelList[[i]]
  }
  return(unique(unlist(labelsG)))
}

#adjust G with information from skeleton alg
#' @noRd
remEdgesFromG <- function(sepset, G, cur_labels) {
  n_sepset <- length(sepset)
  for (i in 1:n_sepset){
    for (j in 1:n_sepset){
      if (!is.null(sepset[[i]][[j]])){
        labelX <- as.character(cur_labels[i])
        labelY <- as.character(cur_labels[j])
        G[labelX, labelY] <- 0
        G[labelY, labelX] <- 0
      }
    }
  }
  return(G)
}

#' @importFrom FCI.Utils getAdjNodes
#' @noRd
adjPairsOneOccurrence <- function(G) {
  labelsG <- colnames(G)
  neighbours <- list()
  G_copy <- G
  #renderAG(G)
  index <- 1

  for (j in 1:length(labelsG)) {
    cur_adj <- getAdjNodes(G_copy, labelsG[j])
    if(length(cur_adj) > 0){
      for(adj in cur_adj){
        neighbours[[index]] <- c(labelsG[j],adj)
        index <- index + 1
        #only one occurrence
        G_copy[adj,labelsG[j]] <- 0
        G_copy[labelsG[j],adj] <- 0
      }
    }
  }
  return(neighbours)
}

#' @importFrom FCI.Utils getAdjNodes
#' @noRd
setOfPossSep <- function(G, p, pdsep.max= Inf, m.max = Inf) {
  labelsG <- colnames(G)
  possSepG <- lapply(seq_len(p), function(.) vector("list", p))
  #this outputs all PossSep(x,G) which are the nodes where a path exists from x
  #that they are possible separating
  allPdsep <- lapply(1:p, qreach, amat = G)
  for(x in seq_len(p)) {
    cur_adj_x <- getAdjNodes(G,x)
    for(y in seq_len(p)){
      cur_adj_y <- getAdjNodes(G,y)
      if(y %in% allPdsep[[x]]){

        X <- labelsG[x]
        Y <- labelsG[y]

        allPdsepLabels_x <- labelsG[allPdsep[[x]]]
        allPdsepLabels_y <- labelsG[allPdsep[[y]]]

        pdsep_x <- setdiff(allPdsepLabels_x, list(Y, cur_adj_x))
        pdsep_y <- setdiff(allPdsepLabels_y, list(X, cur_adj_y))

        #this gets the paths between x and y for which the condition is true
        unionOfSetXY <-  intersect(pdsep_x,  pdsep_y)

        possSepG[[x]][[y]] <- unionOfSetXY
      }
    }
  }
  return(possSepG)
}

#' @noRd
induceSubgraph <- function(G, edges){
  H <- G
  for (edge in edges) {
    H[edge[1], edge[2]] <- H[edge[2], edge[1]] <- 0
  }

  return(H)
}

#' @noRd
all_combinations <- function(lst) {
  result <- list(list()) # also empty set
  for (i in 1:length(lst)) {
    result <- c(result, combn(lst, i, simplify = FALSE))
  }
  return(result)
}


#https://rdrr.io/cran/pcalg/src/R/pcalg.R
#' @noRd
minDiscrPath <- function(pag, a,b,c, verbose = FALSE){
  ## Purpose: find a minimal discriminating path for a,b,c.
  ## If a path exists this is the output, otherwise NA
  ## ----------------------------------------------------------------------
  ## Arguments: - pag: adjacency matrix
  ##            - a,b,c: node positions under interest
  ## ----------------------------------------------------------------------
  ## Author: Diego Colombo, Date: 25 Jan 2011; speedup: Martin Maechler

  p <- as.numeric(dim(pag)[1])
  visited <- rep(FALSE, p)
  visited[c(a,b,c)] <- TRUE # {a,b,c} "visited"
  ## find all neighbours of a  not visited yet
  indD <- which(pag[a,] != 0 & pag[,a] == 2 & !visited) ## d *-> a
  if (length(indD) > 0) {
    path.list <- updateList(a, indD, NULL)
    while (length(path.list) > 0) {
      ## next element in the queue
      mpath <- path.list[[1]]
      m <- length(mpath)
      d <- mpath[m]
      if (pag[c,d] == 0 & pag[d,c] == 0)
        ## minimal discriminating path found :
        return( c(rev(mpath), b,c) )

      ## else :
      pred <- mpath[m-1]
      path.list[[1]] <- NULL


      ## d is connected to c -----> search iteratively
      if (pag[d,c] == 2 && pag[c,d] == 3 && pag[pred,d] == 2) {
        visited[d] <- TRUE
        ## find all neighbours of d not visited yet
        indR <- which(pag[d,] != 0 & pag[,d] == 2 & !visited) ## r *-> d
        if (length(indR) > 0)
          ## update the queues
          path.list <- updateList(mpath[-1], indR, path.list)
      }
    } ## {while}
  }
  ## nothing found:  return
  NA
} ## {minDiscrPath}

#' @noRd
updateList <- function(path, set, old.list) {
  ## Purpose: update the list of all paths in the iterative functions
  ## minDiscrPath, minUncovCircPath and minUncovPdPath
  ## ----------------------------------------------------------------------
  ## Arguments: - path: the path under investigation
  ##            - set: (integer) index set of variables to be added to path
  ##            - old.list: the list to update
  ## ----------------------------------------------------------------------
  ## Author: Diego Colombo, Date: 21 Oct 2011; Without for() by Martin Maechler
  c(old.list, lapply(set, function(s) c(path,s)))
}

#Note: There is no sepset for G, so we create both graphs
#maybe document what we put in the sepset of G
#' @noRd
newRule4 <- function(pag, p, verbose=FALSE) {
  applied <- FALSE
  #orig_pag_objs <- list(pag, sepset)
  #out <- list(orig_pag_objs)
  out_pags <- list(pag)
  seq_p <- seq_len(p)

  # #initialize sepset
  # sepset <- lapply(seq_p, function(.) vector("list", p))
  jci ='0' #no knowledge

  ind <- which((pag != 0 & t(pag) == 1), arr.ind = TRUE)## b o-* c
  while (length(ind) > 0) {
    b <- ind[1, 1]
    c <- ind[1, 2] # pag[b,c] != 0, pag[c,b] == 1
    ind <- ind[-1,, drop = FALSE]
    ## find all a s.t. a -> c and a <-* b
    indA <- which((pag[b, ] == 2 & pag[, b] != 0) &
                    (pag[c, ] == 3 & pag[, c] == 2))
    # pag[b,a] == 2, pag[a,b] != 0
    # pag[c,a] == 3, pag[a,c] == 2
    ## chose one a s.t. the initial triangle structure exists and the edge hasn't been oriented yet
    while (length(indA) > 0 && pag[c,b] == 1) {
      a <- indA[1]
      indA <- indA[-1]
      ## path is the initial triangle
      ## abc <- c(a, b, c)
      ## Done is TRUE if either we found a minimal path or no path exists for this triangle
      Done <- FALSE
      ### MM: FIXME?? Isn't  Done  set to TRUE in *any* case inside the following
      ### while(.), the very first time already ??????????
      while (!Done && pag[a,b] != 0 && pag[a,c] != 0 && pag[b,c] != 0) {
        ## find a minimal discriminating path for a,b,c
        md.path <- minDiscrPath(pag, a,b,c, verbose = verbose)
        ## if the path doesn't exists, we are done with this triangle
        if ((N.md <- length(md.path)) == 1) {
          Done <- TRUE
        } else {
          applied <- TRUE
          pag1 <- pag2 <- pag
          ## a path exists
          ## if b is in sepset

          #NOTE: we dont know about the sepset
          #create both graphs
          #b could be v from the lecture

          # if ((b %in% sepset[[md.path[1]]][[md.path[N.md]]]) ||
          #   (b %in% sepset[[md.path[N.md]]][[md.path[1]]])) {
          #   if (verbose)
          #     cat("\nRule 4",
          #           "\nThere is a discriminating path between",
          #           md.path[1], "and", c, "for", b, ",and", b, "is in Sepset of",
          #           c, "and", md.path[1], ". Orient:", b, "->", c, "\n")

          #FOUND DISCRIMINATING PATH, DO THE THINGS MENTIONED IN THE PAPER

          # TODO for Alina: we should think about updating the sepsets in the entire code
          #sepset1 <- sepset
          #sepset1[[md.path[1]]][[md.path[N.md]]] <- b
          pag1[b, c] <- 2
          pag1[c, b] <- 3
          # }
          # else {
          #   ## if b is not in sepset
          #   if (verbose)
          #     cat("\nRule 4",
          #         "\nThere is a discriminating path between",
          #         md.path[1], "and", c, "for", b, ",and", b, "is not in Sepset of",
          #         c, "and", md.path[1], ". Orient:", a, "<->", b, "<->",
          #         c, "\n")

          #FOUND DISCRIMINATING PATH, DO THE THINGS MENTIONED IN THE PAPER
          pag2[b,c] <- pag2[c,b] <- 2
          if( pag2[a,b] == 3 ) { # contradiction with earlier orientation!
            if( verbose )
              cat('\nContradiction in Rule 4b!\n')
            if( jci == "0" ) {
              pag2[a,b] <- 2 # backwards compatibility
            }
          } else { # no contradiction
            pag2[a,b] <- 2
          }
          out_pags <- list(pag1, pag2)
        }
        Done <- TRUE
      }
    }
  }
  return(out_pags)
}

#' @noRd
computePossImm <- function(PossImm) {
  new_PossImm <- list()
  for(triplet in PossImm){

    if(!is.null(triplet)){
      flipped_triplet <- c(triplet[3],triplet[2],triplet[1])
      if(!any(sapply(new_PossImm, function(x) identical(x, triplet))) &&
         !any(sapply(new_PossImm, function(x) identical(x, flipped_triplet)))){
        new_PossImm[[length(new_PossImm)+1]] <- triplet
      }
    }
  }
  return(new_PossImm)
}

#' @importFrom pcalg find.unsh.triple
#' @noRd
colliderOrientation <- function(amat, G, sepset, verbose=FALSE) {

  labels_Gi <- colnames(amat)

  unshieldedTr <- find.unsh.triple(amat,check=TRUE)
  if (length(unshieldedTr$unshVect > 0)) {
    for (j in 1:ncol(unshieldedTr[[1]])) {
      unshTripl <-  unshieldedTr[[1]][,j] # (unshieldedTr[1]$unshTripl)
      start_node <- unshTripl[1]  #rownames(amat)[unshTripl[1]]
      middle_node <- unshTripl[2] #rownames(amat)[unshTripl[2]]
      end_node <- unshTripl[3]    #rownames(amat)[unshTripl[3]]

      cur_sep <- sepset[[start_node]][[end_node]]
      if (!(is.null(cur_sep))) {
        # check if middle_node belongs to sepset[[start_node]][[[end_node]]]
        # and proceed only if not.
        if (!(middle_node %in% cur_sep)) {
          #used Coding for type amat.pag from pcalg page 12
          if (G[labels_Gi[start_node],  labels_Gi[middle_node]] != 0) { #AND EDGE EXISTS IN G,
            if (verbose) {
              cat("(1) Adding an arrowhead using dataset with variables ", paste0(labels_Gi, collapse=","), "\n")
            }
            G[labels_Gi[start_node],  labels_Gi[middle_node]] <- 2 # start_node o--> middle_node
          }
          if (G[labels_Gi[end_node],  labels_Gi[middle_node]] != 0) { #AND EDGE EXISTS IN G
            if (verbose) {
              cat("(2) Adding an arrowhead using dataset with variables ", paste0(labels_Gi, collapse=","), "\n")
            }
            G[labels_Gi[end_node],  labels_Gi[middle_node]] <- 2
          }
        }
      }
    }
  }
  return(G)
}

#' @noRd
getSubsets <- function(nodes) {
  return(powerSet(nodes))
}

# This is Algorithm 2, note that it applies the skeleton + pdsep + collider orientation
# in each dataset using the iodCITest.
# We can run this only requiring the suffStat of the iodCITest
# cur_labels does not need to be defined in the input
#' @noRd
#' @importFrom pcalg pdsep skeleton
#' @importFrom future.apply future_lapply
#' @importFrom FCI.Utils hasViolation
initialSkeleton <- function(labelList, suffStat, alpha, procedure, verbose=FALSE) {
  G <- initG(labelList)
  n_datasets <- length(labelList)
  listGi <- list()
  sepsetList <- list()
  #listGiRaw <- list()

  skeleton_list <- foreach (i = 1:n_datasets, .verbose=verbose) %dofuture% {
    cur_labels <- labelList[[i]]
    suffStat$cur_labels <- cur_labels
    #skeleton
    skel.fit <- skeleton(suffStat = suffStat,
                         indepTest = iodCITest,
                         method = "stable",
                         alpha = alpha, labels = cur_labels,
                         verbose = verbose, NAdelete = FALSE)
    # skeleton removes edges from independent nodes in Gi
    # orients Colliders of order 0

    sepset <- skel.fit@sepset
    sepset <- fixSepsetList(sepset)

    amat.Gi <-  as(skel.fit@graph, "matrix")
    # renderAG(amat.Gi, add_index = TRUE)
    # formatSepset(sepset)

    # We don't need to remove here, as it is going to be removed later...
    # G <- remEdgesFromG(sepset, G, cur_labels) # here the edges that are removed
    # before in Gi are removed from G

    p <- length(cur_labels)
    pdsepRes <- pdsep(skel=skel.fit@graph, suffStat, indepTest = iodCITest,
                      p = p, sepset = sepset, alpha = alpha,
                      pMax = skel.fit@pMax,
                      m.max = Inf, pdsep.max = Inf,
                      NAdelete = FALSE,
                      verbose = verbose)
    sepset <- pdsepRes$sepset # sepset from the final skeleton
    # formatSepset(sepset)
    sepset_i <- sepset

    G_i <- pdsepRes$G #pdsepRes$G is the final skeleton of Gi

    return(list(sepset_i=sepset_i, G_i=G_i))
  }

  sepsetList <- lapply(skeleton_list, function(x) { x$sepset_i })
  # lapply(sepsetList, formatSepset)
  listGi <- lapply(skeleton_list, function(x) { x$G_i })
  # lapply(listGi, renderAG)

  IP <- list()
  index <- 1
  for (i in 1:length(sepsetList)) {
    sepset <- sepsetList[[i]]
    cur_labels <- labelList[[i]]
    # removing from G the edges removed from Gi after calling pdsep
    G <- remEdgesFromG(sepset, G, cur_labels)

    # adding the remaining edges to IP
    n_sepset <- length(sepset)
    for (k in 1:(n_sepset-1)) {
      for (j in (k+1):n_sepset) {
        #if the Sepset is NULL there is no Separator and the edge is not removed
        if (is.null(sepset[[k]][[j]])) {
          labelX <- as.character(cur_labels[k])
          labelY <- as.character(cur_labels[j])
          IP[index] <- list(c(labelX, labelY, cur_labels))
          index <- index + 1
        }
      }
    }

    if (procedure == "original") {
      G <- colliderOrientation(listGi[[i]], G, sepset)
    }
  }


  pairGiSepsetList <- mapply(list, listGi=listGi, sepsetList=sepsetList, SIMPLIFY=F)

  listGi <- future.apply::future_lapply(pairGiSepsetList,
                                        function(x) { udag2pag(x$listGi, rules = rep(TRUE, 10),
                                                               orientCollider = TRUE, sepset = x$sepsetList) })

  # lapply(listGi, renderAG)

  # get the PAG for Gi
  # transfer all colliders (and maybe all non-colliders) with order to G
  nCK1List <- NA
  nNCKList <- NA
  possImmfromTriplets <-list()
  if (procedure == "orderedcolls"|| procedure == "orderedtriplets") {

    # TODO: check if we need to save the oriented Gi
    for (i in 1:length(listGi)) {
      hasViol <- hasViolation(listGi[[i]], sepsetList[[i]])
      if (hasViol) {
        listGi[[i]] <- (listGi[[i]] != 0) * 1
      }
    }

    G_out <- tripletsWithOrderOrientation(listGi, G, procedure, verbose=verbose, sepsetList = sepsetList)
    G <- G_out$G
    nCK1List <- G_out$nCK1
    nNCKList <- G_out$nNCK
    sepsetList <- G_out$sepsetList
    possImmfromTriplets <- G_out$possImmfromTriplets
  }

  return(list(G=G, IP=IP, sepsetList=sepsetList, listGi=listGi, nCK1List=nCK1List, nNCKList=nNCKList, possImmfromTriplets= possImmfromTriplets))
}

# Here we either include colliders with order and non colliders with order or only
# colliders with order
#' @noRd
#' @importFrom FCI.Utils getMAG MAGtoMEC
tripletsWithOrderOrientation <- function(listGi, G, procedure, verbose=FALSE, sepsetList) {
  nCK1_list = list() # number of colliders with order >= 1
  nNCK_list = list() # number of non-colliders with order >= 0
  possImmfromTriplets <- list()

  mec_list <- foreach (i = 1:length(listGi), .verbose=verbose) %dofuture% {
    amat.pag <- listGi[[i]]
    mag_out <- getMAG(amat.pag)
    magg <- mag_out$magg
    amat.mag <- mag_out$amat.mag
    mec <- MAGtoMEC(amat.mag, verbose=verbose)
  }

  conflicting_pairs <- findConflictingPairs(mec_list, listGi) # NOTE: This should be 1)
  #The first var is the collider

  if(length(conflicting_pairs) > 0){
    updateSepsetList <- updateSepsetList(conflicting_pairs = conflicting_pairs, sepsetList = sepsetList, mec_list = mec_list) # NOTE: This should be 2)
    sepsetList <- updateSepsetList$sepsetList
    possImmfromTriplets <- updateSepsetList$possImmfromTriplets
  }

  for (i in 1:length(listGi)) {
    mec <- mec_list[[i]]
    nCK1_list[[i]] <- NA

    # TODO Jul 19: check if this is correct:
    if (length(mec$CK) > 0) {
      listColliders <- mec$CK[, c("X", "Z", "Y", "ord")] # colliders with order
      number_rows <- dim(listColliders)[[1]]

      # TODO Jul 19: # this is the number of colliders of order higher than 0.
      nCK1_list[[i]] = length(which(listColliders$ord > 0))
      vars_Gi <- colnames(listGi[[i]])
      for (j in 1:number_rows){
        cur_collider <- listColliders[j,]
        cur_indices <- unlist(cur_collider)[1:3]
        vars_collider <- vars_Gi[cur_indices]

        X0 <- mec$CK[j, "X0"]
        Y0 <- mec$CK[j, "Y0"]

        # this if is probably not necessary anymore but it is not harming and save
        if(length(sepsetList[[i]]) >= X0 & length(sepsetList[[i]][[X0]]) >= Y0){
          correspondingSepset <- sepsetList[[i]][[X0]][[Y0]]
        }else{
          correspondingSepset <- NULL
        }

        if(!is.null(correspondingSepset)){
          G <- orientCwo(vars_collider, G, listGi = listGi)
        }
      }
    }
  }

  if (procedure =="orderedtriplets") {

    for (i in 1:length(listGi)) {
      mec <- mec_list[[i]]
      nNCK_list[[i]] <- NA

      # TODO Jul 19: check if this is correct:
      if (length(mec$NCK) > 0) {
        listnonColliders <- mec$NCK[, c("X", "Z", "Y")] # non-colliders with order
        number_rows <- dim(listnonColliders)[[1]]

        # TODO Jul 19: # this is the number of colliders of order higher than 0.
        nNCK_list[[i]] = number_rows
        for (j in 1:number_rows){
          cur_ncollider <- listnonColliders[j,]
          vars_Gi <- colnames(listGi[[i]])
          cur_indices <- unlist(cur_ncollider)
          vars_noncollider <- vars_Gi[cur_indices]
          X0 <- mec$NCK[j, "X0"]
          Y0 <- mec$NCK[j, "Y0"]

          # this if is probably not necessary anymore but it is not harming and save
          if(length(sepsetList[[i]]) >= X0 & length(sepsetList[[i]][[X0]]) >= Y0){
            correspondingSepset <- sepsetList[[i]][[X0]][[Y0]]
          }else{
            correspondingSepset <- NULL
          }

          if(!is.null(correspondingSepset)){
            G <- orientnonColls(vars_noncollider, G, listGi[[i]], listGi)
          }
        }
      }
    }
  }

  return(list(G=G, nCK1=nCK1_list, nNCK=nNCK_list, sepsetList = sepsetList, possImmfromTriplets = possImmfromTriplets))
}

#' @noRd
updateSepsetList <- function(conflicting_pairs = conflicting_pairs, sepsetList = sepsetList, mec_list = mec_list){

  possImmfromTriplets <- list()

  for(i in 1:length(conflicting_pairs)){

    cur_edges <- conflicting_pairs[[i]]$edges
    cur_datasets_having_zero <- conflicting_pairs[[i]]$datasets_having_zero
    cur_datasets_having_one <- conflicting_pairs[[i]]$datasets_having_one

    # the datasets having the definite non ancestors
    for(index_zero in cur_datasets_having_zero){

      cur_mec <- mec_list[[index_zero]]
      index_collider <- which(colnames(cur_mec$skel) == cur_edges[[1]])

      if(any(cur_mec$CK[,"Z"] == index_collider)){ # There is any CK having this Collider
        mecs_of_interest <- cur_mec$CK[cur_mec$CK[,"Z"]== index_collider,]
        number_rows <- dim(mecs_of_interest)[[1]]

        for (j in 1:number_rows){
          X0 <- cur_mec$CK[[j,"X0"]]
          Y0 <-  cur_mec$CK[[j,"Y0"]]
          sepsetList[[index_zero]][[X0]][Y0] <- list(NULL)
          sepsetList[[index_zero]][[Y0]][X0] <- list(NULL)
          col_names <- colnames(cur_mec$skel)

          possImmfromTriplets[[length(possImmfromTriplets)+1]] <- col_names[unlist(mecs_of_interest[j,c("X", "Z", "Y")])]
        }

      }

    }

    # the datasets having the definite ancestors
    for(index_one in cur_datasets_having_one){
      cur_mec <- mec_list[[index_one]]
      index_noncollider <- which(colnames(cur_mec$skel) == cur_edges[[1]])

      if(any(cur_mec$CK[,"Z"] == index_noncollider)){ # There is any CK having this Collider
        mecs_of_interest <- cur_mec$NCK[cur_mec$NCK[,"Z"]== index_noncollider,]
        number_rows <- dim(mecs_of_interest)[[1]]

        for (j in 1:number_rows){
          X0 <- cur_mec$NCK[[j,"X0"]]
          Y0 <-  cur_mec$NCK[[j,"Y0"]]
          sepsetList[[index_zero]][[X0]][Y0] <- NULL
          sepsetList[[index_zero]][[Y0]][X0] <- NULL
          col_names <- colnames(cur_mec$skel)

          possImmfromTriplets[[length(possImmfromTriplets)+1]] <- col_names[unlist(mecs_of_interest[j,c("X", "Z", "Y")])]
        }

      }

    }

  }

  return(list(sepsetList = sepsetList, possImmfromTriplets = possImmfromTriplets))
}

#' @noRd
findConflictingPairs <- function(mec_list, listGi){
  conflicts <-list()

  listGi_ancestral <- lapply(listGi, getAncestralMatrix)

  #collider is first
  alledges <- getalledges(mec_list)

  for (edge in alledges) {
    list_orientations <- list()
    for(Gi in listGi_ancestral){
      #  = 0 implies that i is a definite non-ancestor of j
      #  = 1 implies that i is a definite ancestor of j
      #  = 2 implies that i is a possible ancestor of j
      # only check if Gi contains the edges

      if(edge[[1]] %in% colnames(Gi) && edge[[2]] %in% colnames(Gi)) {
        list_orientations[length(list_orientations)+1] <- Gi[edge[[1]],edge[[2]]]
        #i <- i+1
      }
    }
    if(0 %in% list_orientations && 1 %in% list_orientations) {

      datasets_having_zero <- which(list_orientations == 0)
      datasets_having_one <- which(list_orientations == 1)


      conflicts[[length(conflicts)+1]] <- list(edges = edge,
                                               datasets_having_zero = datasets_having_zero,
                                               datasets_having_one = datasets_having_one)
    }
  }
  return(conflicts)
}

#' @noRd
getalledges <- function(mec_list){

  cur_mec <- list()
  edges <- list()

  for(i in 1:length(mec_list)){
    cur_mec <- mec_list[[i]]
    coln<- colnames(cur_mec$skel)

    number_rows <- dim(cur_mec$CK)[[1]]
    if(length(number_rows) > 0 && number_rows > 0){
      for (j in 1:length(number_rows)) {
        collZX <- coln[unlist(cur_mec$CK[j,c("Z","X")])]
        collZY <- coln[unlist(cur_mec$CK[j,c("Z","Y")])]
        edges[[length(edges)+1]] <- collZX
        edges[[length(edges)+1]] <- collZY
      }
    }

    number_rows <- dim(cur_mec$NCK)[[1]]
    if(length(number_rows) > 0 && number_rows > 0){
      for (j in 1:length(number_rows)) {
        collZX <- coln[unlist(cur_mec$NCK[j,c("Z","X")])]
        collZY <- coln[unlist(cur_mec$NCK[j,c("Z","Y")])]
        edges[[length(edges)+1]] <- collZX
        edges[[length(edges)+1]] <- collZY
      }
    }
  }
  return(unique(edges))
}

# use ancestrality matrix here
# True if there are contradictory orientations
# list_orientations: an empty list or a list with one element: 1, for testing
# if edge[[2]] can be a definite ancestor of edge[[1]], or 0, for testing
# if the edge[[2]]  can be a definite non-ancestor oc edge[[1]].
#' @noRd
checkForContradictoryOrientations <- function(edge, listGi, list_orientations=list()) {
  i <- length(list_orientations) + 1

  listGi_ancestral <- lapply(listGi, getAncestralMatrix)
  for(Gi in listGi_ancestral){
    #  = 0 implies that i is a definite non-ancestor of j
    #  = 1 implies that i is a definite ancestor of j
    #  = 2 implies that i is a possible ancestor of j
    # only check if Gi contains the edges
    if(edge[[1]] %in% colnames(Gi) && edge[[2]] %in% colnames(Gi)) {
      list_orientations[[i]] <- Gi[edge[[1]],edge[[2]]]
      i <- i+1
    }
  }

  if(0 %in% list_orientations && 1 %in% list_orientations) {
    return(TRUE)
  }

  return(FALSE)
}

#' @noRd
orientCwo <- function(vars_collider, G, Gi, listGi){
  #if(!checkForContradictoryOrientations(edge = list(vars_collider[[2]], vars_collider[[1]]), listGi = listGi)) {
  if(G[vars_collider[[1]],  vars_collider[[2]]] != 0) {
    G[vars_collider[[1]], vars_collider[[2]]] <- 2 # start_node o--> middle_node
  }
  #}
  #if(!checkForContradictoryOrientations(edge =  list(vars_collider[[2]], vars_collider[[3]]), listGi = listGi)) {
  if (G[vars_collider[[3]],  vars_collider[[2]]] != 0) { #AND EDGE EXISTS IN G
    G[vars_collider[[3]],  vars_collider[[2]]] <- 2
  }
  #}
  return(G)
}

#' @noRd
orientnonColls <- function(vars_noncollider, G, Gi, listGi) {
  # transfer the tails if they exist
  #if(!checkForContradictoryOrientations(edge = list(vars_noncollider[[1]],  vars_noncollider[[2]]), listGi = listGi)) {
  if(Gi[vars_noncollider[[1]],  vars_noncollider[[2]]] == 3 && (G[vars_noncollider[[1]], vars_noncollider[[2]]] != 0)) { #check if Gi has this edge and if the edge exists in G...
    G[vars_noncollider[[1]], vars_noncollider[[2]]] <- 3 # start_node *--- middle_node
  }
  #}

  #if(!checkForContradictoryOrientations(edge = list(vars_noncollider[[3]],  vars_noncollider[[2]]), listGi = listGi)) {
  if (Gi[vars_noncollider[[3]],  vars_noncollider[[2]]] == 3 && (G[vars_noncollider[[3]], vars_noncollider[[2]]] != 0)) { #check if Gi has this edge
    G[vars_noncollider[[3]],  vars_noncollider[[2]]] <- 3 # middle_node --* end_node
  }
  #}

  # Here, all arrowheads that are not conflicting across the Gi's
  # have been already transferred to G.
  # It is possible, therefore, that a ncwo has only circles in the middle node in Gi
  # but in G, it has an arrowhead on it.
  # In this case, we have to complete the other edge mark as a tail, thus
  # preventing such a triplet to be considered a possible immorality later on.
  if (Gi[vars_noncollider[[1]],  vars_noncollider[[2]]] == 1 && Gi[vars_noncollider[[3]],  vars_noncollider[[2]]] == 1) {
    # There is circle-circle definite non-collider in Gi

    # Triplet in G has an arrowhead on the first edge of the triplet
    if (G[vars_noncollider[[1]],  vars_noncollider[[2]]] == 2 && G[vars_noncollider[[3]],  vars_noncollider[[2]]] == 1) {
      # This would imply Gi[vars_noncollider[[3]],  vars_noncollider[[2]]] == 3
      # Checking if such definite ancestrallity would violate orientations in the others G_i
      if(!checkForContradictoryOrientations(edge = list(vars_noncollider[[3]],  vars_noncollider[[2]]),
                                            listGi = listGi, list_orientations = list(1))) {
        # Transfering such definite ancestrallity to G:
        G[vars_noncollider[[3]],  vars_noncollider[[2]]] <- 3
      }
    }

    # Triplet in G has an arrowhead on the second edge of the triplet
    if (G[vars_noncollider[[3]],  vars_noncollider[[2]]] == 2 && G[vars_noncollider[[1]],  vars_noncollider[[2]]] == 1) {
      # This would imply Gi[vars_noncollider[[1]],  vars_noncollider[[2]]] == 3
      # Checking if such definite ancestrallity would violate orientations in the others G_i
      if(!checkForContradictoryOrientations(edge = list(vars_noncollider[[1]],  vars_noncollider[[2]]),
                                            listGi = listGi, list_orientations = list(1))) {
        # Transfering such definite ancestrallity to G:
        G[vars_noncollider[[1]],  vars_noncollider[[2]]] <- 3
      }
    }
  }

  return(G)
}

#' @noRd
getPossImm <- function(H, n_datasets,suffStat,sepsetList, labelsG){
  PossImm <- list()
  for (z in colnames(H)) {
    adj_z <- which(H[z, ] != 0)
    for (x in adj_z) {
      for (y in adj_z) {
        if (x != y) {
          #Can X,Z,Y be made an immorality? -> if H[X,Y]=0 (unshielded, symmetric)
          if (H[x,y] == 0 &&
              H[x,z] != 3 && H[y,z] != 3) {
            conditionsforAllVi <- list()
            for (i in 1:n_datasets) {

              conditionsforAllVi[i] <- FALSE
              cur_labels <- labelList[[i]]
              sepsetGi <- sepsetList[[i]]
              #check if x,y in GI
              # Labels of G are the same as labels of H

              x_label <- labelsG[x]
              y_label <- labelsG[y]

              #check if Sepset is undefined, i.e. X,Y are not both observed in Gi
              if(!(x_label %in% cur_labels) | !(y_label %in% cur_labels)){
                conditionsforAllVi[i] <- TRUE
              }else{
                # check if Z is not in Vi
                if(!(z %in% cur_labels)){
                  conditionsforAllVi[i] <- TRUE # NOTE: this captures some conflicts of triplets that are both cwo and ncw across different Gi's
                }
                else{
                  indx <- which(cur_labels == x_label)
                  indy <- which(cur_labels == y_label)
                  #Sepset is NULL -> undef in paper
                  if(length(sepsetGi[[indx]]) >= indy){
                    lengthlist <- length(sepsetGi[[indx]][[indy]])
                  }else{
                    lengthlist <- 0
                  }
                  if(!(lengthlist > 0)) {
                    conditionsforAllVi[i] <- TRUE
                  }
                }
              }
            }
            if(all(unlist(conditionsforAllVi))) {
              if(H[x_label,z] != 2 || H[y_label,z] != 2) { # this should prevent getting possImm that are inside of H
                PossImm[[length(PossImm)+1]] <- c(x_label,z,y_label)
              }
            }
          }
        }
      }
    }
  }
  PossImm <- computePossImm(PossImm) #removes flipped occurencies
  return(PossImm)
}

#' @importFrom FCI.Utils getAdjNodes
#' @noRd
getRemEdges <- function(existingEdges, G, possSepList, labelList) {
  n_datasets <- length(labelList)

  RemEdges <- list()
  for (pair in existingEdges) {
    X <- pair[1]
    Y <- pair[2]
    labelsG <- colnames(G)
    index_X <- which(labelsG == X)
    index_Y <- which(labelsG == Y)

    AdjX <- getAdjNodes(G, X)
    AdjY <- getAdjNodes(G, Y)

    cur_possSep <- c(possSepList[[index_X]][[index_Y]])

    if (length(cur_possSep) > 0) {
      set1 <- unique(unlist(c(pair, AdjX, cur_possSep)))
      set2 <- unique(unlist(c(pair, AdjY, cur_possSep)))
    }
    else{
      set1 <- unique(unlist(c(pair, AdjX))) #currPossSep can be char(0)
      set2 <- unique(unlist(c(pair, AdjY)))
    }

    flagRemEdge <- TRUE
    for (i in 1:n_datasets) {
      cur_labels <- labelList[[i]]

      #setdiff outputs elements that are in vector1 and not in vector2
      if (length(setdiff(set1, cur_labels)) == 0) {
        flagRemEdge <- FALSE
      }
      if (length(setdiff(set2, cur_labels)) == 0) {
        # enters here onl< if the possSep was already entirely observed in some Vi
        flagRemEdge <- FALSE
      }
    }

    if (flagRemEdge) {
      RemEdges[[length(RemEdges)+1]] <- pair
    }
  }
  return(RemEdges)
}

#' @noRd
#' @importFrom pcalg udag2pag
applyRulesOnHt <- function(listAllHt){

  rules <- rep(TRUE,10)
  rules[4] <- FALSE
  orientCollider = FALSE
  sepset <- list()

  done_flags <- rep(FALSE, length(listAllHt))
  graphs_done <- list()
  G_PAG <- list()

  while (!all(done_flags)) {

    listPagsLength <- length(listAllHt)

    for (j in 1:listPagsLength) {
      if (!done_flags[j]) {
        new_graph <- udag2pag(listAllHt[[j]], rules = rules, orientCollider = FALSE, sepset = list())
        has_rule4_change <- newRule4(new_graph, length(colnames(new_graph)))

        if (length(has_rule4_change) == 2) {
          done_flags[j] <- TRUE
          listAllHt[[length(listAllHt)+1]] <- has_rule4_change[[1]]
          listAllHt[[length(listAllHt)+1]] <- has_rule4_change[[2]]
          done_flags[length(done_flags)+1] <- FALSE
          done_flags[length(done_flags)+1] <- FALSE
        }
        else {
          graphs_done[[length(graphs_done)+1]] <- new_graph
          done_flags[j] <- TRUE
        }
      }
    }
    G_PAG <- c(G_PAG, graphs_done)
  }
  return(G_PAG)
}

#' @noRd
checkIfInvariancesfromGiAreInPAG <- function(listGi, G_PAG){
  tracker <- c()
  apag <- getAncestralMatrix(G_PAG)
  for (gi in listGi) {
    agi <- getAncestralMatrix(gi)
    relevant_vars <- colnames(agi)
    for(var_col in  relevant_vars){
      for(var_row in  relevant_vars){
        if(var_col != var_row){
          value_subset <- agi[var_col,var_row]
          if (!(apag[var_col, var_row] == value_subset) && !(apag[var_col, var_row] == 2 || value_subset == 2)) {
            tracker <- c(tracker, FALSE)
          } else {
            tracker <- c(tracker, TRUE)
          }
        }
      }
    }
  }
  return(all(tracker))
}

#' @importFrom FCI.Utils getTruePAG getMAG
#' @importFrom ggm isAG makeMG
#' @noRd
hasOnlyValidMAGs <- function(pagAdjM, verbose = FALSE) {

  # Checking whether there are cycles or almost cycles
  isAGret <- tryCatch({
    amag <- getMAG(pagAdjM)
    if (!is.null(amag$amat.mag)) {
      ug_mag <- (amag$amat.mag == 3 & t(amag$amat.mag == 3)) * 1
      bg_mag <- (amag$amat.mag == 2 & t(amag$amat.mag == 2)) * 1
      dg_mag <- (amag$amat.mag == 2 & t(amag$amat.mag == 3)) * 1
      mag_ggm <- ggm::makeMG(dg_mag, ug_mag, bg_mag)
      ggm::isAG(mag_ggm)
    } else {
      FALSE
    }
  },
  error=function(cond) {
    print(cond)
    return(FALSE)
  },
  warning=function(cond) {
    print(cond)
    return(FALSE)
  })

  if (!isAGret) {
    if (verbose) {
      cat(paste("MAG is not ancestral.\n"))
    }
    return(FALSE)
  } else {
    # Here we check whether a MAG in the PAG is maximal
    # We do this by getting the PAG P of a MAG in pagAdjM and comparing
    # whether P and pagAdjM have the same skeleton.
    # We do not check if P and pagAdjM are identical (with the same orientations)
    # because the returning PAGs may be more informative than the ones recovered
    # only using conditional independencies.

    recPAG <- getTruePAG(amag$magg, verbose = verbose)
    # renderAG(recPAG@amat)
    recPAG_skel <- (recPAG@amat[colnames(pagAdjM), colnames(pagAdjM)] != 0) * 1
    pagAdjM_skel <- (pagAdjM != 0) * 1
    if (any(recPAG_skel - pagAdjM_skel != 0)) {
      if (verbose) {
        cat(paste("MAG is not maximal.\n"))
      }
      return(FALSE)
    }
    return(TRUE)
  }
}

#' @importFrom FCI.Utils isMSeparated
#' @noRd
validatePossPags <- function(G_PAG, sepsetList, labelList, IP, method, listGi, verbose=FALSE){
  violates_list <-
    foreach (J = G_PAG, .verbose=verbose) %dofuture% {
      #for (J in G_PAG) {
      violates <- FALSE
      #(ii)
      if (!violates) {
        for (n in 1:length(sepsetList)) {
          for (i in 1:length(sepsetList[[n]])) {
            for (j in 1:length(sepsetList[[n]][[i]])) {
              labels <- labelList[[n]]

              if(length(sepsetList[[n]]) > 0  && length(sepsetList[[n]][[i]]) > 0 && length(sepsetList[[n]][[i]][[j]]) > 0 ){
                if(length(sepsetList[[n]][[i]]) >= j){
                    Sij <- sepsetList[[n]][[i]][[j]]
                }
                else{
                    Sij <- NULL
                }
              } else {
                  Sij <- NULL
              }


              if(!is.null(Sij) & i!=j){
                xname <- labels[i]
                yname <- labels[j]
                snames <- labels[unlist(Sij)]

                if (verbose) {
                  cat(paste("Checking if {", paste0(labels[unlist(Sij)], collapse={","}),
                            "} m-separates", labels[i], "and", labels[j],"\n"))
                }
                msep <- isMSeparated(J, xname, yname, snames, verbose=verbose)
                violates <- violates || !msep # because we want the m-separation

              }
            }
          }
        }

        if(!violates) {
          #(iii)
          # IP saves the labels
          for(xyS in IP) {

            X <- xyS[1]
            Y <- xyS[2]
            Vi <- xyS[3:length(xyS)]

            nodes <- setdiff(Vi, list(X,Y))
            subsets <- getSubsets(nodes) # this returns list of all possible V'

            for(subset in subsets){
              if(!violates){
                msep <- isMSeparated(J, X, Y, subset,
                                     verbose=verbose)
                violates <- violates || msep # because we want the m-connection
                if(method == "consistence" & !violates){
                  if(!checkIfInvariancesfromGiAreInPAG(listGi, J)){
                    violates <- TRUE
                  }
                }
              }
            }
          }
        }

        #(i)
        if (!violates) {
          # Here we check whether the PAG is valid by checking whether
          # the canonical MAG is ancestral
          validPAG <- hasOnlyValidMAGs(J) # returns a boolean
          if (! validPAG) {
            violates <- TRUE
          }
        }
      }
      violates
    }
  #return(G_PAG_List)
  return(unlist(violates_list))
}

#' @importFrom dagitty dseparated
#' @noRd
dagittyCIOracle2 <- function(x, y, S, suffStat) {
  g <- suffStat$g
  labels <- suffStat$labels
  if (dagitty::dseparated(g, labels[x], labels[y], labels[S])) {
    return(1)
  } else {
    return(0)
  }
}

# returns a matrix in which [i,j]
#  = 0 implies that i is a definite non-ancestor of j
#  = 1 implies that i is a definite ancestor of j
#  = 2 implies that i is a possible ancestor of j
#' @noRd
getAncestralMatrix <- function(amat.pag) {
  defTrueAncM <- getAncestralMatrixHelper(amat.pag, definite = T)
  possTrueAncM <- getAncestralMatrixHelper(amat.pag, definite = F)
  return((possTrueAncM - defTrueAncM) + possTrueAncM)
}

# returns a matrix in which [i,j] = 1 implies that
# i is a possible/definite ancestor of j
#' @importFrom pcalg possAn
#' @noRd
getAncestralMatrixHelper <- function(amat.pag, definite=TRUE) {
  ancM <- 0 * amat.pag
  if (definite) {
    # Removing every edge with a circle
    to_rm <- which(amat.pag == 1, arr.ind = T)
    to_rm2 <- to_rm[,c(2,1)]
    amat.pag[rbind(to_rm, to_rm2)] <- 0
  }
  labels <- colnames(ancM)
  for (vj in 1:length(labels)) {
    possAncVj <- pcalg::possAn(amat.pag, x=vj, type="pag", ds=FALSE)
    possAncVj <- setdiff(possAncVj, vj)
    if (length(possAncVj) > 0) {
      ancM[possAncVj, vj] <- 1
    }
  }
  return(ancM)
}
