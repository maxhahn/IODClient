library(FCI.Utils)
library(pcalg)
library(rIOD)
library(doFuture)

true.amat.pag <- t(matrix(c(0,0,2,2,0,
                           0,0,2,0,0,
                           2,1,0,2,2,
                           2,0,3,0,2,
                           0,0,3,3,0), 5, 5))

# SANITY CHECK (First of 100 random PAGs)
#true.amat.pag <- t(matrix(c(0,1,0,2,0,
#                           1,0,2,0,0,
#                           0,1,0,2,2,
#                           1,0,2,0,2,
#                           0,0,2,3,0), 5, 5))

colnames(true.amat.pag) <- c("A", "B", "C", "D", "E")
rownames(true.amat.pag) <- colnames(true.amat.pag)

renderAG(true.amat.pag)

#n_cores <- 8
#plan("multicore", workers = n_cores, gc=TRUE)


get_data_four_partitions <- function(true_pag_amat, variable_levels, coef_thresh, seed) {
  if(seed > 0) {
    set.seed(seed)
  }

  obs_vars_1 <- c("A", "C", "D", "E")
  obs_vars_2 <- c("A", "B", "C", "E")

  n1_1 = 2500
  n1_2 = 2500
  n2_1 = 5000
  n2_2 = 5000

  # N1
  d1_1 = get_data(true_pag_amat, n1_1, variable_levels, 'continuous', coef_thresh)
  d1_1 <- d1_1$dat[, obs_vars_1]

  # N2
  d1_2 = get_data(true_pag_amat, n1_2, variable_levels, 'continuous', coef_thresh)
  d1_2 <- d1_2$dat[, obs_vars_1]

  # N3
  d2_1 = get_data(true_pag_amat, n2_1, variable_levels, 'continuous', coef_thresh)
  d2_1 <- d2_1$dat[, obs_vars_1]

  # N4
  d2_2 = get_data(true_pag_amat, n2_2, variable_levels, 'continuous', coef_thresh)
  d2_2 <- d2_2$dat[, obs_vars_1]




}

get_data <- function(true_pag_amat, num_samples, variable_levels, mode, coef_thresh) {

  var_levels  <- list()
  cols <- colnames(true_pag_amat)
  for (vari in 1:length(cols)) {
      var_name <- colnames(true_pag_amat)[vari]
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
