rm(list=ls())
source("IOD.R")
source("IOD_Helper.R")
source("SimulationHelper.R")
source("tests/testfunctions.R")


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
library(future.apply)
library(gtools)
library(dagitty)
library(stringr)

##################################################

n_cores <- 8
# plan("multisession", workers = n_cores)
plan("multicore", workers = n_cores)
# plan("cluster", workers = n_cores)


output_folder <- "./ResultsUnfaithfulRandomGraphs/"
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

NvecList <- list(10000, 10000)
for (NVec in NvecList) {
  datasets_suffstat_file <- paste0(output_folder,
                                   paste0("datasets_suffstats_N", NVec, ".RData"))
  if (!file.exists(datasets_suffstat_file)) {
    datasets_suffstats <- generateDatasetsSuffStats(truePAGs, subsetsList,
                                                    NVec, data_type="continuous")
    save(datasets_suffstats, file=datasets_suffstat_file)
  } else {
    load(datasets_suffstat_file)
  }
  results_file <- paste0(output_folder, "results_N", NVec[1], ".RData")
  if (!file.exists(results_file)) {
    fileid = paste0("randomPAG_N", NVec)
    results_filenames <- list.files(pattern = paste0("^", fileid, "_*"), output_folder, full.names = FALSE)

    processed_ids <- as.numeric(sapply(results_filenames,
                                       function(x) { as.numeric(str_extract(x, "\\d+"))}))

    toProcessed_ids <- 1:n_tests
    if (length(processed_ids) > 0) {
      toProcessed_ids <- toProcessed_ids[-processed_ids]
    }

    suffStats <- lapply(datasets_suffstats, function(x) { x$suffStat })

    procedeIODWithGraphs(truePAGs[toProcessed_ids], subsetsList[toProcessed_ids],
                         output_folder, fileid=fileid, suffStats=suffStats)

    results_files <- list.files(pattern = paste0("^", fileid, "_*"), output_folder, full.names = TRUE)
    graphs <- truePAGs
    getStatistics(graphs, results_files, output_folder, fileid = fileid)
  }
   else {
     load(results_file)
   }
}
