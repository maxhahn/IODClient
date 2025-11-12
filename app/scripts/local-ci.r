library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)

n_cores <- 1
plan("multicore", workers = n_cores, gc=TRUE)

run_ci_test <- function(data, max_cond_set_cardinality, filedir, filename) {

  data[] <- lapply(data, function(x) if (is.character(x)) as.factor(x) else x)

  labels <- colnames(data)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = data,
                                 vars_names = labels,
                                 covs_names = c())
  citestResults <- getAllCITestResults(data,
                                      indepTest,
                                      suffStat,
                                      m.max=max_cond_set_cardinality,
                                      saveFiles=TRUE,
                                      fileid=filename,
                                      citestResults_folder=filedir)
  result <- list(citestResults=citestResults, labels=labels)
  result
}
