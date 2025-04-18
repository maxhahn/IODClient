#' @title Learning PAGs from multiple datasets with overlapping variables by using the IOD
#'
#' @description This package implements the Integration of overlapping datasets algorithm by Tillman et. al. (2011) and
#' also offers 2 additional procedures which both improve the performance.
#'
#' The Algorithm can be used to obtain a list of PAGs from datasets with overlapping variables.
#' The PAGs in the list all reflect the observed data.
#' The IOD assumes faithfulness, so if this assumption does not hold, the results can be bad containing PAGs with wrong (in)dependencies or an empty list.
#'
#' @param suffStat list of lists each including the nth CI-test results and the labels of the dataset
#' @param alpha significance level on which the combined statistics in the IOD reject H0 (independence of variables). The default value is 0.05.
#' @param method the default value is "standard". If "standard" is chosen, it won't have an additional check. If "consistence" is chosen,
#' an additional validation check is done, which will exclude any PAGs from the list, if they violate marginal consistency. This could be helpful, when dealing with unfaithfulness.
#' @param procedure there are 3 procedures you can choose from: "original", "orderedcolls" and "orderedtriplets".
#' "original" performs the regular IOD. "orderedcolls" includes colliders of any order, while "orderedtriplets" considers triplets of any order.
#' "orderedcolls" or "orderedtriplets" can have an output in unfaithful cases where the  "original" version could not find any, because in cases where
#' datasets have conflicting orientations we consider both.
#' @param verbose the default value is FALSE. If it is set to TRUE, there is additional output in the console about the steps and calculations that are done.
#'
#' @return Returns a list with the following elements:
#' \describe{
#' \item{G_PAG_List}{The output list of PAGs given by adjacency matrices which reflect the given statistics (\code{G_PAG_List}).}
#' \item{Gi_PAG_list}{A list of the fully oriented input-datasets (\code{Gi_PAG_list}), which were used to train the PAGs.}
#' \item{len_before}{The length of the output list of PAGs before these were validated (\code{len_before}).}
#' \item{nCK1}{The number of colliders of an order higher than 1 (\code{nCK1}) taken into account.}
#' \item{nNCK1}{The number of non-colliders of any order (\code{nNCK1})taken into account.}
#'}
#'
#'#'
#' @examples
#' # I generate fake data from the true PAG below
#' adag_out <- FCI.Utils::getDAG("pdsep_g")
#' truePAG <- FCI.Utils::getTruePAG(adag_out$dagg)
#' trueAdjM <- truePAG@amat
#'
#' # create datasets or organize yours like this:
#' data <- list()
#' for (i in 1:3) {
#'   adat_out <- FCI.Utils::generateDataset(adag = adag_out$dagg,
#'                                          N=100000, type = "continuous")
#'   cur_full_dat <- adat_out$dat
#'   data[[i]] <-  cur_full_dat[,
#'       sample(1:ncol(cur_full_dat), size = 3), drop=FALSE] #generated datasets
#'}
#' # run the citests separately
#' citestResultsList <- list()
#' labelList <- list()
#' index <- 1
#' for (cur_dat in data) {
#'  #this is how to run CI Tests for a dataset cur_dat
#'  cur_labels <- colnames(cur_dat)
#'  indepTest <- FCI.Utils::mixedCITest
#'  suffStat <- FCI.Utils::getMixedCISuffStat(dat = cur_dat,
#'                                 vars_names = cur_labels,
#'                                  covs_names = c())
#'
#'   citestResults <- FCI.Utils::getAllCITestResults(
#'                                        cur_dat, indepTest, suffStat,
#'                                        m.max=2, saveFiles = TRUE,
#'                                        fileid = sprintf("%04d", index),
#'                                        citestResults_folder="./citests/")
#'   citestResulstList[[index]] <- citestResults
#'   labelList[[index]] <- cur_labels
#'
#'   index <- index + 1
#' }
#'
#' # create the suffstat for the meta-analysis approach
#' suffStat <- list()
#' suffStat$citestResultsList <- citestResultsList
#' suffStat$labelList <- labelList
#'
#' # call IOD.
#' alpha <- 0.05
#' iod_out <- IOD(labelList, suffStat, alpha)
#'
#' # show the output.
#' iod_out$Gi_PAG_list # list of PAGs generated from each dataset
#' lapply(iod_out$Gi_PAG_list, FCI.Utils::renderAG)
#'
#' iod_out$G_PAG_List # list of possible merged PAGs
#'
#' # This is going to render the graphs in the viewer tab
#' lapply(iod_out$G_PAG_List, FCI.Utils::renderAG)
#'
#' @import doFuture
#' @importFrom foreach foreach
#' @importFrom rje powerSet
#' @export IOD
IOD <- function(labelList, suffStat, alpha=0.05, method = "standard", procedure = "original", verbose=FALSE) {
  E = NULL

  initSkeletonOutput <- initialSkeleton(labelList, suffStat, alpha, procedure=procedure, verbose=verbose)
  G <- initSkeletonOutput$G
  # renderAG(G)

  IP <- initSkeletonOutput$IP
  sepsetList <- initSkeletonOutput$sepsetList
  # lapply(sepsetList, formatSepset)

  Gi_list <- initSkeletonOutput$listGi
  # lapply(Gi_list, renderAG)

  nCK1 <- sum(unlist(initSkeletonOutput$nCK1List), na.rm = TRUE)
  nNCK <- sum(unlist(initSkeletonOutput$nNCKList), na.rm = TRUE)

  n_datasets <- length(labelList)

  possImmfromTriplets <- initSkeletonOutput$possImmfromTriplets

  # Algorithm 3:

  p <- length(colnames(G))

  G_copy <- G
  G_copy[which(G_copy == 3)] <- 1

  #all((amat != 0) == (t(amat != 0))) ist nicht TRUE
  possSepList <- setOfPossSep(G_copy,p)

  existingEdges <- adjPairsOneOccurrence(G)

  RemEdges <- getRemEdges(existingEdges,G, possSepList, labelList)


  power_RemEdges <- powerSet(unique(RemEdges))
  # one_edge_list <- which(lapply(power_RemEdges, length) == 1)

  index_possImmList <- 1

  #G_PAG_List <- list()
  #for (E in power_RemEdges) {
  G_PAG_List <- foreach (E = power_RemEdges, .verbose=verbose) %dofuture% {
    H <- induceSubgraph(G,E)
    labelsG <- colnames(G)
    PossImm <- getPossImm(H, n_datasets, suffStat, sepsetList, labelsG)

    if (procedure == "orderedtriplets") {
      savetails <- H
      savetails[which(savetails != 3)] <- 0

      H[which(savetails == 3)] <- 1
    }

    listAllHt <- list()
    if (length(PossImm) > 0) {
      power_possImm <- all_combinations(PossImm)
      for (t in power_possImm) {
        H_t <- H
        # orient Colliders
        for (tau in t) {
          if (!is.null(tau)) {
            H_t[tau[1], tau[2]] <- 2
            H_t[tau[3], tau[2]] <- 2
          }
        }
        listAllHt[[length(listAllHt)+1]] <- H_t
      }
    } else {
      listAllHt[[length(listAllHt)+1]] <- H
    }


    G_PAG <- unique(applyRulesOnHt(unique(listAllHt)))
    temp_G_PAG <- G_PAG
    # TODO: below, just change circles to tails, not arrowhead to tails..
    # if there is an arrowhead on a place that a definite tail is expected,
    # we should remove the PAG inside of this loop
    # -- it is fine to remove these PAGs before
    # getting the G_PAG_List_before.


    if(procedure == "orderedtriplets"){
      #include Tails
      indices <- which(savetails == 3, arr.ind = TRUE)
      # Zeilennamen der Werte, die 3 sind
      row_names <- rownames(savetails)[indices[, 1]]

      # Spaltennamen der Werte, die 3 sind
      col_names <- colnames(savetails)[indices[, 2]]

      if (length(row_names) > 0) {
        temp_G_PAG <- list()
        for (i in 1:length(G_PAG)) {
          exclude_Pag <- FALSE
          #% should not be indices but names
          cur <- G_PAG[[i]]

          for (j in 1:length(row_names)) {
            # do not want to force the arrowheads
            if (cur[row_names[[j]], col_names[[j]]] != 2) {
              cur[row_names[[j]], col_names[[j]]] <- 3
            } else{
              exclude_Pag <- TRUE
            }
            # idk what I did here, maybe a left over from something that passed away
            #if(i == length(G_PAG) && j == length(row_names)){
            #exclude_Pag <- TRUE
            #}
          }

          if (!exclude_Pag) {
            temp_G_PAG[[length(temp_G_PAG)+1]] <- cur
          }
        }
      }
    }
    G_PAG <- temp_G_PAG
    # For each possible G in the power set of graphs you are creating, make sure
    # to update sepset accordingly.
    return(G_PAG)
  }

  # This is counting the number of PAGs before using the violation checks,
  # which takes more time in the computational sense.
  if (length(unique(unlist(G_PAG_List, recursive=F))) != 0) {
    #The if excludes list() and list(list(), list(), list()), list(list(list(), list()), list(list(), list())) ...
    G_PAG_List_before <- unique(unlist(G_PAG_List, recursive=F))
    len_before <- length(G_PAG_List_before)
  } else {
    G_PAG_List_before <- G_PAG_List <- list()
    len_before <- 0
  }

  if (len_before > 0) {
    violation_List <- validatePossPags(G_PAG_List_before, sepsetList, labelList, IP, method, Gi_list)
    G_PAG_List <- G_PAG_List_before[!violation_List]
  }
  return(list(G_PAG_List=G_PAG_List, Gi_PAG_list=Gi_list,
              G_PAG_List_before=G_PAG_List_before, len_before=len_before,
              nCK1=nCK1, nNCK=nNCK))
}
