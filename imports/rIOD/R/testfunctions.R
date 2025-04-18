#' @export containsTheTrueGraph
containsTheTrueGraph <- function(trueAdjM, paglist, verbose=FALSE) {
  is_in_list <- FALSE
  for (adj_matrix in paglist) {
    #make sure that trueAdjM and theadj_matrix have the same sequence of vars
    new_order <- colnames(adj_matrix)
    trueAdjM <- trueAdjM[new_order, new_order]
    if (all(adj_matrix == trueAdjM)) {
      is_in_list <- TRUE
      break
    }
  }
  if(is_in_list) {
    if (verbose)
      print("The List of PAGs includes the true graph")
    return(TRUE)
  } else {
    if (verbose)
      print("The List of PAGs DOESN'T include the true graph")
    return(FALSE)
  }
}
