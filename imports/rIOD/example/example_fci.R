rm(list=ls())

library(FCI.Utils)
library(pcalg)
library(jsonlite)
library(dagitty)

library(doFuture)
library(future.apply)

library(doFuture)
n_cores <- 8
plan("multicore", workers = n_cores, gc=TRUE)


# Simulate data following the structure of a dagitty DAG, adag.
# type: either "continuous" or "binary"
# To generate mixed data, discretize certain continuous variables afterward.
generateDatasetFromDAG <- function(adag, N, type="continuous", verbose=FALSE) {
  if (!(type %in% c("continuous", "binary")))  {
    stop("type must be either continuous or binary")
  }

  done <- FALSE
  while (!done) {
    done <- tryCatch(
      {
        if(type == "binary") {
          obs.dat <- dagitty::simulateLogistic(adag, N=N, verbose=FALSE)
          obs.dat <- as.data.frame(sapply(obs.dat, function(col) as.numeric(col)-1))
          lt <- dagitty::localTests(adag, obs.dat, type="cis.chisq")
          TRUE
        } else if (type == "continuous") {
          obs.dat <- dagitty::simulateSEM(adag, N=N)
          lt <- dagitty::localTests(adag, obs.dat, type="cis")
          R <- cor(obs.dat)
          valR <- matrixcalc::is.symmetric.matrix(R) &&
            matrixcalc::is.positive.definite(R, tol=1e-8)
          valR
        }
        TRUE
      }, error=function(cond) {
        message(cond)
        return(FALSE)
      })
  }
  return(list(dat=obs.dat, lt=lt))
}

# n: number of nodes
# e_prob: probability of having an edge between any two nodes
# l_prob: proportion of the nodes to be considered latent (unmeasured variables)
getRandomGraph <- function(n, e_prob, l_prob, force_nedges=TRUE)  {
  if (force_nedges) {
    min_nedges <- floor((n*(n-1)/2)*(e_prob * 1))
  } else {
    min_nedges <- 0
  }
  nedges <- 0
  while (nedges <= min_nedges) {
    dag <- pcalg::randomDAG(n, e_prob, V = paste0("V", sprintf(
      paste0("%0", nchar(n), "d"), seq(from = 1, to = n))))

    amat.dag <- (as(dag, "matrix") > 0) * 1
    adag <- pcalg::pcalg2dagitty(amat.dag, colnames(amat.dag), type="dag")
    lat <- sample(colnames(amat.dag), l_prob * n)

    dagitty::latents(adag) <- lat

    magg <- dagitty::toMAG(adag)
    nedges <- nrow(edges(magg))
  }

  return(adag)
}


get2DiscrPathGraph <- function() {
  allvars <- c("A", "B", "C", "D", "E", "Ubc", "Uce")
  p <- length(allvars)
  amat <- matrix(0, p, p)
  colnames(amat) <- rownames(amat) <- allvars

  amat["A","B"] <- 0; amat["B","A"] <- 1; # A -> B
  amat["B","D"] <- 0; amat["D","B"] <- 1; # B -> D
  amat["B","E"] <- 0; amat["E","B"] <- 1; # B -> E
  amat["C","D"] <- 0; amat["D","C"] <- 1; # C -> D

  # B <-> C
  amat["Ubc","B"] <- 0; amat["B","Ubc"] <- 1; # Ubc -> B
  amat["Ubc","C"] <- 0; amat["C","Ubc"] <- 1; # Ubc -> C

  # C <-> E
  amat["Uce","C"] <- 0; amat["C","Uce"] <- 1; # Uce -> C
  amat["Uce","E"] <- 0; amat["E","Uce"] <- 1; # Uce -> E

  lat <- c("Ubc", "Uce")
  adag <- pcalg::pcalg2dagitty(amat, colnames(amat), type="dag")
  dagitty::latents(adag) <- lat
  dagitty::coordinates(adag) <-
    list( x=c(A=1, B=1, C=1, D=0, E= 2, Ubc=1, Uce=1.5),
          y=c(A=0, B=1, C=2, D=2.5, E=2.5, Ubc=1.5, Uce=2.25) )

  return(list(amat.dagitty = adag, # adjacency matrix as a dagitty object
              amat.pcalg = amat  # adjacency matrix as a matrix using pcalg notation
  ))
}

getFaithfulnessDegree <- function(amat.pag, citestResults, alpha) {
  labels <- colnames(amat.pag)
  exp_indep <- data.frame()
  exp_dep <- data.frame()

  for (i in 1:nrow(citestResults)) {
    cur_row <- citestResults[i, , drop=TRUE]
    snames <- labels[getSepVector(cur_row$S)]
    xname <- labels[cur_row$X]
    yname <- labels[cur_row$Y]

    def_msep <- isMSeparated(amat.pag, xname, yname, snames,
                             verbose=FALSE)
    if (def_msep) {
      exp_indep <- rbind.data.frame(exp_indep, c(cur_row))
    } else {
      exp_dep <- rbind.data.frame(exp_dep, c(cur_row))
    }
  }

  exp_indep <- cbind(exp_indep, faithful=exp_indep$pvalue > alpha)
  exp_dep <- cbind(exp_dep, faithful=exp_dep$pvalue <= alpha)
  indep_faithf_score <- length(which(exp_indep$faithful))/length(exp_indep$faithful)
  dep_faithf_score <- length(which(exp_dep$faithful))/length(exp_dep$faithful)
  dep_faithf_score_ord <- aggregate(exp_dep$faithful, by = list(exp_dep$ord), FUN = "mean")
  dep_faithf_score_ord_vec <- dep_faithf_score_ord$x
  names(dep_faithf_score_ord_vec) <- paste0("dep_faithf_score_ord", dep_faithf_score_ord$Group.1)

  faithful_vec <- c(exp_indep$faithful, exp_dep$faithful)
  faithful_score <- length(which(faithful_vec))/length(faithful_vec)

  return(as.list(c(faithful_score=faithful_score,
                   indep_faithf_score=indep_faithf_score,
                   dep_faithf_score=dep_faithf_score,
                   dep_faithf_score_ord_vec)))
}


###############################
# Setting up true DAG and PAG #
###############################

true_dag_dagitty <- get2DiscrPathGraph()$amat.dagitty
#true_dag_dagitty <- getRandomGraph(6, 0.3, 0.2)
true_pag_amat <- getTruePAG(true_dag_dagitty)@amat

renderAG(true_pag_amat, add_index = TRUE)

true_sepset <- getPAGImpliedSepset(true_pag_amat)
print(formatSepset(true_sepset))

#############################################################################
# Generating the dataset with variables as columns and observations as rows #
#############################################################################

# seed for only continuous variables: 1562332467

#cur_seed <- sample(1:.Machine$integer.max, 1)
cur_seed <- 1229693544
set.seed(cur_seed)

N = 10000 # sample size
type = "continuous" # "binary"

adat_out <- generateDatasetFromDAG(adag = true_dag_dagitty, N=N, type=type)
dat <- adat_out$dat
head(dat)

###############################
# Discretizing some variables #
###############################

discretize <- function(v, levels=2, labels = c(0,1)) {
  return(cut(v, breaks=levels, labels=labels))
}

discr_vars <- sample(colnames(dat), 2)
dat[, discr_vars] <- lapply(dat[, discr_vars], discretize)

##############################
# Running using mixedCITests #
##############################

covs_names = c() # variables that are not part of the graph, but are always conditioned on.
vars_names = colnames(dat) # variables in the dataset that are nodes in the graph
vars_df <- dat[,vars_names, drop=FALSE]

indepTest <- mixedCITest
suffStat <- getMixedCISuffStat(dat = dat,
                               vars_names = vars_names,
                               covs_names = c())

fileid <- NULL #paste0("seed_", cur_seed)
citestResults <- getAllCITestResults(vars_df, indepTest, suffStat,
                                     m.max=Inf, computeProbs = FALSE,
                                     fileid=fileid,
                                     saveFiles = TRUE,
                                     citestResults_folder="./tmp/")

alpha = 0.05
faithfulness_score <- getFaithfulnessDegree(true_pag_amat, citestResults, alpha)
faithfulness_score$faithful_score

suffStat$citestResults <- citestResults

estimated_pag <- pcalg::fci(suffStat,
                     indepTest = indepTest,
                     labels= vars_names, alpha = alpha,
                     verbose = TRUE)

estimated_pag_amat <- estimated_pag@amat

renderAG(estimated_pag_amat) # This may be very wrong under unfaithfulness
faithfulness_score$faithful_score
