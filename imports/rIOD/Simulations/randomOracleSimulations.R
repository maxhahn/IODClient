rm(list=ls())
##################################################
# TO INSTALL THE PACKAGES BELOW
# install.packages(c("FCI.Utils", "pcalg", "igraph","RBGL","rje",
# "graph", "doFuture", "gtools","dagitty"))
##################################################
library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(dagitty)
library(stringr)
#install
library(rIOD)
source("tests/testfunctions.R")

##################################################


library(doFuture)
library(future.apply)
n_cores <- 8

plan("multicore", workers = n_cores)

output_folder <- "./ResultsOracleRandomGraphs/"
if (!is.null(output_folder) && !file.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

n_tests = 100
n_nodes = 5
pagsfile = paste0(output_folder, n_tests, "randomPAGs.RData")

if (file.exists(pagsfile)) {
  load(pagsfile)
} else {
  pags_subsets <- generateUniqueRandomPAGsSubsets(n_graphs=n_tests)
  truePAGs <- pags_subsets$pags
  subsetsList <- pags_subsets$subsets

  save(truePAGs, subsetsList, file=paste0(output_folder, n_tests, "randomPAGs.RData"))
}

load(pagsfile)

fileid = "randomPAG"
results_filenames <- list.files(pattern = paste0("^", fileid, "_*"), output_folder, full.names = FALSE)

processed_ids <- as.numeric(sapply(results_filenames,
                        function(x) { as.numeric(str_extract(x, "\\d+"))}))

toProcessed_ids <- 1:n_tests
if (length(processed_ids) > 0) {
  toProcessed_ids <- toProcessed_ids[-processed_ids]
}
procedeIODWithGraphs(truePAGs, subsetsList, output_folder, fileid=fileid)

procedeIODWithGraphs(truePAGs[toProcessed_ids], subsetsList[toProcessed_ids], output_folder, fileid=fileid)

results_file <- paste0(output_folder, "results.RData")
if (!file.exists(results_file)) {
  results_files <- list.files(pattern = paste0("^", fileid, "_*"), output_folder, full.names = TRUE)
  graphs <- truePAGs
  getStatistics(graphs, results_files, output_folder, fileid = fileid)
} else {
  load(results_file)
}


