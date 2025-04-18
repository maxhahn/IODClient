library(FCI.Utils)
library(pcalg)
library(rIOD)
library(doFuture)

n_cores <- 8
plan("multicore", workers = n_cores, gc=TRUE)

# true.amat.pag <-
#   fromJSON("[[0,0,2,2,0],[0,0,2,0,0],[2,1,0,2,2],[2,0,3,0,2],[0,0,3,3,0]]")

true.amat.pag <- t(matrix(c(0,0,2,2,0,
                           0,0,2,0,0,
                           2,1,0,2,2,
                           2,0,3,0,2,
                           0,0,3,3,0), 5, 5))
colnames(true.amat.pag) <- c("A", "B", "C", "D", "E")
rownames(true.amat.pag) <- colnames(true.amat.pag)

renderAG(true.amat.pag)


#######################
# Simulation Datasets #
#######################

aseed <- 5325496 # This seed generates data corresponding the
                 # example from the slide
set.seed(aseed)

#########################
# Simulating for Node 1 #
#########################

N = 5000
obs_vars_1 <- c("A", "C", "D", "E")
dat_out <- FCI.Utils::generateDatasetFromPAG(apag = true.amat.pag,
                                              N=N,
                                              type = "continuous")
dataset_1 <- dat_out$dat[, obs_vars_1]
head(dataset_1)
write.csv(dataset_1, file = "./example/dataset_1.csv", row.names = FALSE)


#########################
# Simulating for Node 2 #
#########################

N = 10000
obs_vars_2 <- c("A", "B", "C", "E")
dat_out <- FCI.Utils::generateDatasetFromPAG(apag = true.amat.pag,
                                             N=N,
                                             type = "continuous")
dataset_2 <- dat_out$dat[, obs_vars_2]
head(dataset_2)
write.csv(dataset_2, file = "./example/dataset_2.csv", row.names = FALSE)

################################
# Run FCI locally in each node #
################################

indepTest <- mixedCITest
alpha <- 0.05

###################
# PAG from Node 1 #
###################


suffStat_1 <- getMixedCISuffStat(dat = dataset_1,
                                 vars_names = obs_vars_1,
                                 covs_names = c())

citestResults_1 <- getAllCITestResults(dataset_1, indepTest, suffStat_1)

estimated_pag_1 <- pcalg::fci(suffStat_1,
                              indepTest = indepTest,
                              labels= obs_vars_1, alpha = alpha,
                              verbose = TRUE)

renderAG(estimated_pag_1@amat)


###################
# PAG from Node 2 #
###################

suffStat_2 <- getMixedCISuffStat(dat = dataset_2,
                                 vars_names = obs_vars_2,
                                 covs_names = c())

citestResults_2 <- getAllCITestResults(dataset_2, indepTest, suffStat_2)


estimated_pag_2 <- pcalg::fci(suffStat_2,
                              indepTest = indepTest,
                              labels= obs_vars_2, alpha = alpha,
                              verbose = TRUE)

renderAG(estimated_pag_2@amat)


###############
# Running IOD #
###############

labelList <- list()
citestResultsList <- list()
citestResultsList[[1]] <- citestResults_1
labelList[[1]] <- obs_vars_1

citestResultsList[[2]] <- citestResults_2
labelList[[2]] <-  obs_vars_2


######################################################################
# Test using citestResultsList of separated p-values for each client #
######################################################################

# Creating a suffStat including citestResultsList and labelList

suffStat <- list()
suffStat$citestResultsList <- citestResultsList
suffStat$labelList <- labelList

# call IOD.
alpha <- 0.05
iod_out <- IOD(labelList, suffStat, alpha)

# list of PAGs generated using combined p-values in each node
iod_out$Gi_PAG_list
lapply(iod_out$Gi_PAG_list, renderAG)

# list of possible merged PAGs
iod_out$G_PAG_List
lapply(iod_out$G_PAG_List, renderAG)

#function to check if the true pag is inside the pag list
containsTheTrueGraph(trueAdjM = true.amat.pag, iod_out$G_PAG_List)


##########################################################
# Test using a citestResults table of combined p-values  #
##########################################################

# Creating the table combined p-values

fisherMetaTest <- function(pvalues) {
  test_statistic <- -2 * sum(log(pvalues))
  df <- 2 * length(pvalues)
  p <- pchisq(test_statistic, df, lower.tail = FALSE) # H0: Independency
  return(p)
}

all_labels <- unique(c(obs_vars_1, obs_vars_2))
citestResults <- extractValidCITestResults(citestResults_1, obs_vars_1, all_labels)
citestResults <- rbind(citestResults,
                       extractValidCITestResults(citestResults_2, obs_vars_2, all_labels))


citestResults_merged <- aggregate(citestResults$pvalue, by = (citestResults[, 1:4]), fisherMetaTest)
colnames(citestResults_merged) <- colnames(citestResults)

# Creating a suffStat including citestResults and all_labels

suffStat <- list()
suffStat$citestResults <- citestResults_merged
suffStat$all_labels <- all_labels
suffStat$verbose <- TRUE
iod_out <- IOD(labelList, suffStat, alpha)


# list of PAGs generated using combined p-values in each node
iod_out$Gi_PAG_list
lapply(iod_out$Gi_PAG_list, renderAG)

# list of possible merged PAGs
iod_out$G_PAG_List
lapply(iod_out$G_PAG_List, renderAG)

#function to check if the true pag is inside the pag list
containsTheTrueGraph(trueAdjM = true.amat.pag, iod_out$G_PAG_List)
