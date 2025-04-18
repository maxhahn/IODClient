library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)
library(matrixcalc)

n_cores <- 8
plan("multicore", workers = n_cores, gc=TRUE)

# create datasets for 3 servers separately

# They are saved in data[[1]], data[[2]], data[[3]]
adag_out <- getDAG("pdsep_g")
truePAG <- getTruePAG(adag_out$dagg)
trueAdjM <- truePAG@amat
renderAG(trueAdjM)

set.seed(12345)
data <- list()
for (i in 1:3) {
  adat_out <- FCI.Utils::generateDataset(adag = adag_out$dagg, N=100000, type = "continuous")
  cur_full_dat <- adat_out$dat
  data[[i]] <-  cur_full_dat[, sample(1:ncol(cur_full_dat), size = 3), drop=FALSE] #generated datasets
}

# run the citests separately
citestResultsList <- list()
labelList <- list()
index <- 1
for (cur_dat in data) {
  #this is how to run CI Tests for a dataset cur_dat
  cur_labels <- colnames(cur_dat)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = cur_dat,
                                 vars_names = cur_labels,
                                 covs_names = c())

  citestResults <- getAllCITestResults(cur_dat, indepTest, suffStat,
                                       m.max=2, saveFiles = TRUE,
                                       fileid = sprintf("%04d", index),
                                       citestResults_folder="./citests/")
  citestResultsList[[index]] <- citestResults
  labelList[[index]] <- cur_labels
  index <- index + 1
}


# Transfer each list(citestResults=citestResults, labels=cur_labels) created locally to the server

# Part of the server
# create the list of the suffstat
suffStat <- list()
suffStat$citestResultsList <- citestResultsList
suffStat$labelList <- labelList

# call IOD.
alpha <- 0.05
iod_out <- IOD(labelList, suffStat, alpha)

# show the output.
iod_out$Gi_PAG_list # list of PAGs generated from each dataset
lapply(iod_out$Gi_PAG_list, renderAG)

iod_out$G_PAG_List # list of possible merged PAGs
lapply(iod_out$G_PAG_List, renderAG)

#function to check if the true pag is inside the pag list
containsTheTrueGraph(trueAdjM = trueAdjM, iod_out$G_PAG_List)

