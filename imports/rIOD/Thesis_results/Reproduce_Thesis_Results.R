# This file exists to recreate the results of the thesis and to guide through the files in a compact way.

#Build source package and install package

##################################################
#install.packages(c("MXM", "pscl", "DOT", "rsvg"))
#install.packages("doSNOW")
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
library(rIOD)
source("Simulations/SimulationHelper.R")
##################################################

library(doFuture)
n_cores <- 8
plan("multicore", workers = n_cores, gc=TRUE)

######################################################################################
#Chapter 3.2. Improvements due to including colliders of higher order
load("~/IOD_Rpackage/Thesis_results/256graphs.RData")

#Used in the improvements_cwo
trueAdjM <- graphs[[29]]
renderAG(trueAdjM)
true.sepset <- getPAGImpliedSepset(trueAdjM) #citype = "missing.edge"
formatSepset(true.sepset)
labels <- colnames(trueAdjM)

#Oracle with indep from trueAdjM
setsToConsider <- list(list("A", "B", "C", "E"), list("B", "C", "D", "E"))
suffStat<- list()
suffStat$labels <- labels
citestResultsList_out <- getCIResultsList(citest_type="oracleCI",
                                           setsToConsider=setsToConsider,
                                           true.pag.amat = trueAdjM)
suffStat$citestResultsList <- citestResultsList_out$citestResultsList
suffStat$labelList <- citestResultsList_out$labelList

labelList <- suffStat$labelList
alpha = 0.05
#source("R/IOD_Helper.R")
iod_out <- IOD(labelList, suffStat, alpha)
listPags <- iod_out$G_PAG_List # PAGS
listGi <- iod_out$Gi_PAG_list
iod_out$len_before
length(listPags)


#lapply(listPags, renderAG)
#lapply(listGi, renderAG)

source("tests/testfunctions.R")
containsTheTrueGraph(trueAdjM, listPags)

#G_i in thesis
#renderAG(iod_out$Gi_PAG_list[[1]], type="pdf", width=300, height=300)
#renderAG(iod_out$Gi_PAG_list[[2]], type="pdf", width=300, height=300)


iod_out_cwo <- IOD(labelList, suffStat, alpha, procedure = "orderedcolls")
listPags_cwo <- iod_out_cwo$G_PAG_List # PAGS
listGi_cwo  <- iod_out_cwo$Gi_PAG_list
iod_out_cwo$len_before
length(listPags_cwo)

lapply(listPags_cwo, renderAG)

containsTheTrueGraph(trueAdjM, listPags_cwo)


iod_out_ncwo <- IOD(labelList, suffStat, alpha, procedure = "orderedtriples")
listPags_ncwo <- iod_out_ncwo$G_PAG_List # PAGS
listGi_ncwo  <- iod_out_ncwo$Gi_PAG_list
iod_out_ncwo$len_before
length(listPags_ncwo)

lapply(listPags_ncwo, renderAG)
lapply(listGi_ncwo, renderAG)

containsTheTrueGraph(trueAdjM, listPags_ncwo)


# the output lists have the same lengths so if diff is 0 it is the same
#for(i in 1:256){
#  trueAdjM <- graphs[[i]]
#  iod_out <- IOD(suffStat, alpha)
#  listPags <- iod_out$G_PAG_List # PAGS
#  iod_out_ncwo <- IOD(suffStat, alpha, procedure = "orderedtriples")
#  listPags_ncwo <- iod_out_ncwo$G_PAG_List # PAGS
#  print(getDiffPAGList(listPags_ncwo, listPags))
#}
# only empty lists
# Note: in all observations in the final version of the algorithm the final output PAG lists were the same

#################################################################
#iod_out$G_PAG_List_before has 463 PAGs
#iod_out_cwo$G_PAG_List_before: 389
#iod_out_,cwo$G_PAG_List_before: 301

#getDiffPAGList is all in the 1st list that are not in the second list ( |A\B| )
#comparing cwo and org
diff1 <- getDiffPAGList(iod_out_cwo$G_PAG_List_before, iod_out$G_PAG_List_before)#195
diff2 <- getDiffPAGList(iod_out$G_PAG_List_before, iod_out_cwo$G_PAG_List_before)#269

renderAG(diff1[[1]]) #Used in the thesis (in cwo not in org)
renderAG(diff2[[1]]) #Used in the thesis (in org not in cwo)

#The formula below was used to calculate the intersection.
# |A\B| + |B\A| + |A ∩ B| = |A| + |B| - |A ∩ B|
# 195 + 269 + x = 463 + 389 - x
# x = 194
######################################################################################################
#Chapter 3.3. Improvements due to including triples of higher order (this extends the example above)


diff3 <- getDiffPAGList(iod_out$G_PAG_List_before, iod_out_ncwo$G_PAG_List_before) #367
diff4 <- getDiffPAGList(iod_out_ncwo$G_PAG_List_before,iod_out$G_PAG_List_before) #205


renderAG(diff3[[1]])#Used in the thesis (in org not in ncwo)
renderAG(diff4[[1]])#Used in the thesis (in ncwo not in org)

#diff3[[5]]
#diff4[[3]]
#both of the above violate the IP check

#The formula below was used to calculate the intersection.
# |A\B| + |B\A| + |A ∩ B| = |A| + |B| - |A ∩ B|
# 367 + 205 + x = 463 + 301 - x
# x = 96
######################################################################################################
# Chapter 4
# To obtain the results from this chapter the function procedeIODWithGraphs() and getStatistics() were used.
# To get an idea of how this was done, take a look at the files randomUnfaithfulSimulations.R, randomOracleSimulations.R, 256_discrGraphs.R.

# 4.1
# The result variable is an evaluation of all tests (getStatistics).
# A total of 100 tests were conducted (please refer Simultions/randomOracleSimulations.R)

# in the following, the results are analysed
load("~/IOD_Rpackage/Thesis_results/results_100_ oracle.RData")

#all include the true pag
all(results$orig.true == TRUE)
all(results$cwo.true == TRUE)
all(results$ncwo.true == TRUE)

# all approaches produced the same output list the shd and for and fdr are the same in every case
# one could rund the IOD for every example and use
# getDiffPAGList() to investigate the output lists
# I observed only the same PAGS in the output lists under faithfulness which also implies that including
# triples with order also has a PAG list as output and not graphs that are differently classified.


# Histograms and tables (Appendix)
print(results)
print(unlist(results$cwo.lenbef)- unlist(results$orig.lenbef))
print(unlist(results$orig.lenbef) - unlist(results$ncwo.lenbef))
print(unlist(results$cwo.lenbef) - unlist(results$ncwo.lenbef))

print(table(unlist(results$cwo.lenbef)- unlist(results$orig.lenbef)))
print(table(unlist(results$ncwo.lenbef)-unlist(results$orig.lenbef)))
print(table(unlist(results$ncwo.lenbef)-unlist(results$cwo.lenbef)))

rotate_x <- function(data, labels_vec, rot_angle, ylim) {
  plt <- barplot(data, col='steelblue', xaxt="n", ylim=ylim, lwd= 2, cex.axis=1.5)
  text(plt, par("usr")[3], labels = labels_vec, srt = rot_angle, adj = c(1.1,1.1), xpd = TRUE, cex=1.5)
}

summary_len_orig_cwo <- cut(unlist(results$cwo.lenbef)- unlist(results$orig.lenbef), breaks=c(-402,-200, -100,-1, 0))
data <- table(summary_len_orig_cwo)
labels_vec <- sort(unique(summary_len_orig_cwo))
pdf("len_org_cwo.pdf")
rotate_x(data,labels_vec,22, ylim = c(0, 70))
dev.off()


summary_len_orig_ncwo <- cut(unlist(results$ncwo.lenbef)- unlist(results$orig.lenbef), breaks=c(-1015, -200, -100,-1, 0))
data <- table(summary_len_orig_ncwo)
labels_vec <- c("(-1015,-200]", "(-200,-100]", "(-100,-1]", "(-1,0]")
pdf("len_ncwo_org.pdf")
rotate_x(data,labels_vec,22, ylim = c(0, 40))
dev.off()

summary_len_ncwo_cwo <- cut(unlist(results$ncwo.lenbef)- unlist(results$cwo.lenbef), breaks=c(-1015,-200, -100,-1, 0))
data <- table(summary_len_ncwo_cwo)
labels_vec <- c("(-1015,-200]", "(-200,-100]", "(-100,-1]", "(-1,0]")# sort(unique(summary_len_ncwo_cwo))
table(summary_len_ncwo_cwo)
pdf("len_ncwo_cwo.pdf")
rotate_x(data,labels_vec,22, ylim = c(0, 40))
dev.off()

#Inverstigate why org and cwo obtain the same lists before when a discriminating path is in the subset and colliders can be identified
#the first graph is an example, see:
print(unlist(results$cwo.lenbef)- unlist(results$orig.lenbef))
load("~/IOD_Rpackage/Thesis_results/100randomPAGs.RData")
truePAG <- truePAGs[[1]]
subset <- subsetsList[[1]]
setsToConsider <- subset
suffStat<- list()
citestResultsList_out<- getCIResultsList(citest_type="oracleCI",
                                         setsToConsider=setsToConsider,
                                         true.pag.amat = truePAG)
suffStat$citestResultsList <- citestResultsList_out$citestResultsList
suffStat$labelList <- citestResultsList_out$labelList
labelList <- suffStat$labelList

################################################################################################################################
# 4.2
load("~/IOD_Rpackage/Thesis_results/results_randomPAG_N10000.RData")
# To obtain these results, the true graphs from 4.1 were used.
# The result is an evaluation of all tests (getStatistics).
# A total of 100 tests were conducted (please refer Simultions/randomUnfaithfulSimulations.R).
res <- results

# all include the true pag? NO
all(results$orig.true == TRUE)
all(results$cwo.true == TRUE)
all(results$ncwo.true == TRUE)
summary(unlist(results$orig.true))
summary(unlist(results$cwo.true))
summary(unlist(results$ncwo.true))

# If the true graph was found, it was found by all
all(unlist(results$orig.true)==unlist(results$cwo.true))
all(unlist(results$orig.true)==unlist(results$ncwo.true))

#The output list length is not the same
all(unlist(results$orig.npags)==unlist(results$cwo.npags)) #not the same
all(unlist(results$cwo.npags)==unlist(results$ncwo.npags)) #same

ind <- which(unlist(results$orig.npags)!=unlist(results$cwo.npags)) #4  8 33 58

#min(unlist(results$shd_cwo[[4]]))
min(unlist(results$shd_ncwo[[4]]))
min(unlist(results$posneg_ncwo_fdr[[4]]))
#due to these observations the 3rd is the best
min(unlist(results$posneg_ncwo_for[[4]]))

#min(unlist(results$shd_cwo[[8]]))
min(unlist(results$shd_ncwo[[8]]))
min(unlist(results$posneg_ncwo_fdr[[8]]))
min(unlist(results$posneg_ncwo_for[[8]]))

#min(unlist(results$shd_cwo[[33]]))
min(unlist(results$shd_ncwo[[33]]))
min(unlist(results$posneg_ncwo_fdr[[33]]))
min(unlist(results$posneg_ncwo_for[[33]]))

#min(unlist(results$shd_cwo[[58]]))
min(unlist(results$shd_ncwo[[58]]))
min(unlist(results$posneg_ncwo_fdr[[58]]))
min(unlist(results$posneg_ncwo_for[[58]]))

# How often we find nothing
sum(unlist(results$orig.npags)==0)
sum(unlist(results$cwo.npags)==0)
sum(unlist(results$ncwo.npags)==0)

cat(unlist(results$orig.npags[ind]),"\n", unlist(results$cwo.npags[ind]),"\n", unlist(results$ncwo.npags[ind]))
# So the differences occure only when the original version could not find any PAG.
# In this case the approaches return some possible PAGs having marginal inconsisties

ind_org <- which(unlist(results$orig.margIncons) > 0) #12
ind_cwo <- which(unlist(results$cwo.margIncons) > 0) #4 8 12 33 58
ind_ncwo <- which(unlist(results$ncow.margIncons)>0) #4 8 12 33 58

# 12 is not a marginal violation. we detected 12 because detetecting marginal consistency violations is hard, and we did not check it completely correct
# 12 is not actually a violation

#### Metrics in general: compare shd for fdr#####
min_org <- unlist(lapply(lapply(results$shd_org, unlist),min))
min_cwo <- unlist(lapply(lapply(results$shd_cwo, unlist), min))
min_ncwo <- unlist(lapply(lapply(results$shd_ncwo, unlist), min))

min_org[which(min_org ==Inf)] <- 1000
min_cwo[which(min_cwo ==Inf)] <- 1000
min_ncwo[which(min_ncwo ==Inf)] <- 1000

min_cwo - min_org
min_ncwo - min_org

#these are the 4 cases were we changed the IOD regarding the conflicting edges

#the average of the min, and the average of those that have an output list
mean(min_org[min_org< 1000])

mean(min_cwo[min_cwo< 1000])
min_cwo <- min_cwo[-c(4,8,12,33,58)]
mean(min_cwo[min_cwo< 1000])

mean(min_ncwo[min_ncwo< 1000])
min_ncwo <- min_ncwo[-c(4,8,12,33,58)]
mean(min_ncwo[min_ncwo< 1000])

#all shd
pdf("SHD.pdf")
rotate_x(table(min_org[min_org< 1000]), c(0,2:11), 22, ylim=c(0,25))
dev.off()

#

avg_org <- unlist(lapply(lapply(results$shd_org, unlist),mean))
avg_cwo <- unlist(lapply(lapply(results$shd_cwo, unlist), mean))
avg_ncwo <- unlist(lapply(lapply(results$shd_ncwo, unlist), mean))

avg_org[which(is.na(avg_org))] <- 0
avg_cwo[which(is.na(avg_cwo))] <- 0
avg_ncwo[which(is.na(avg_ncwo))] <- 0


avg_cwo - avg_org
avg_ncwo - avg_org

getMeanSHDStats <- function(list_shd,list_fdr, list_for){
  mean_shd_list <- list()
  mean_for_list <- list()
  mean_fdr_list <- list()
  for(i in 1:length(list_shd)){
    if(length(list_shd[[i]]) > 0){
      shd <- unlist(lapply(lapply(list_shd[i], unlist),min))
      index <- which(list_shd[[i]] == shd)
      # there can be multiple min shd
      for_val <- mean(unlist(list_for[[i]])[index])
      fdr_val <- mean(unlist(list_fdr[[i]])[index])

      mean_shd_list[i] <- shd
      mean_for_list[i] <- for_val
      mean_fdr_list[i] <- fdr_val
    }
  }
  return(list(shdMean = mean(unlist(mean_shd_list)), forMean = mean(unlist(mean_for_list)), fdrMean= mean(unlist(mean_fdr_list))))
}

getMeanSHDStats(results$shd_org, results$posneg_org_fdr, results$posneg_org_for)
getMeanSHDStats(results$shd_cwo, results$posneg_cwo_fdr, results$posneg_cwo_for)
getMeanSHDStats(results$shd_ncwo, results$posneg_ncwo_fdr, results$posneg_ncwo_for)
#

max_org <- unlist(lapply(lapply(results$shd_org, unlist),max))
max_cwo <- unlist(lapply(lapply(results$shd_cwo, unlist), max))
max_ncwo <- unlist(lapply(lapply(results$shd_ncwo, unlist), max))

max_org[which(max_org==-Inf)] <- 0
max_cwo[which(max_cwo==-Inf)] <- 0
max_ncwo[which(max_ncwo==-Inf)] <- 0

max_cwo - max_org
max_ncwo - max_org

# all of the output lists have the same shd besides the 4 cases where conflicting edges occured

#minimal
posneg_org_fdr <- unlist(lapply(results$posneg_org_fdr, function(x) ifelse(length(x) > 0, min(unlist(x)), NA)))
posneg_cwo_fdr <- unlist(lapply(results$posneg_cwo_fdr, function(x) ifelse(length(x) > 0, min(unlist(x)), NA)))
posneg_ncwo_fdr <- unlist(lapply(results$posneg_ncwo_fdr,function(x) ifelse(length(x) > 0, min(unlist(x)), NA)))
posneg_org_for <- unlist(lapply(results$posneg_org_for, function(x) ifelse(length(x) > 0, min(unlist(x)), NA)))
posneg_cwo_for <- unlist(lapply(results$posneg_cwo_for, function(x) ifelse(length(x) > 0, min(unlist(x)), NA)))
posneg_ncwo_for <-unlist(lapply(results$posneg_ncwo_for, function(x) ifelse(length(x) > 0, min(unlist(x)), NA)))

# this function checks if there are any differencies in the lists
# If so, the indices are returned
compare_fdr_for <-  function(list1, list2){
  index <- list()
  for(i in 1:length(list1)){
    if(!is.na(list1[i]) && !is.na(list2[i])){
      if(list1[i] != list2[i]){
        index[length(index)+1] <- i
      }
    }
    if(!is.na(list1[i]) && is.na(list2[i])){
      index[length(index)+1] <- i
    }
    if(is.na(list1[i]) && !is.na(list2[i])){
      index[length(index)+1] <- i
    }
  }
  return(index)
}

compare_fdr_for(posneg_org_fdr,posneg_ncwo_fdr)
compare_fdr_for(posneg_org_fdr,posneg_cwo_fdr)
compare_fdr_for(posneg_org_for,posneg_ncwo_for)
compare_fdr_for(posneg_org_for,posneg_cwo_for)

######

which(unlist(res$cwo.lenbef) - unlist(res$orig.lenbef) > 0) #  4 33 58 65
# 4 33 58, here we observed a bigger list before validation but delivered any list
# 65 what happend here

#unlist(res$ncwo.lenbef)
#unlist(res$orig.lenbef)
#res$cwo.lenbef

# Histograms and tables (Appendix)
summary_len_orig_cwo <- cut(unlist(res$cwo.lenbef) - unlist(res$orig.lenbef), breaks=c(-300,-200, -100, -1,0, 248))
data <- table(summary_len_orig_cwo)
labels_vec <- sort(unique(summary_len_orig_cwo))
pdf("len_org_cwo.pdf")
rotate_x(data,labels_vec,22, ylim = c(0, 70))
dev.off()

summary_len_orig_ncwo <- cut(unlist(res$ncwo.lenbef) - unlist(res$orig.lenbef), breaks=c(-560, -200,-100, -1,0, 5))
data <- table(summary_len_orig_ncwo)
labels_vec <- sort(unique(summary_len_orig_ncwo))
pdf("len_ncwo_org.pdf")
rotate_x(data,labels_vec,22, ylim = c(0, 50))

dev.off()

summary_len <- cut(unlist(res$ncwo.lenbef) - unlist(res$cwo.lenbef), breaks=c(-370,-200, -100, -1, 0))
data <- table(summary_len)
labels_vec <- sort(unique(summary_len))
pdf("len_ncwo_cwo.pdf")
rotate_x(data,labels_vec,22, ylim = c(0, 50))
dev.off()

table(unlist(res$cwo.lenbef) - unlist(res$orig.lenbef))
table(unlist(res$ncwo.lenbef) - unlist(res$orig.lenbef))
table(unlist(res$ncwo.lenbef) - unlist(res$cwo.lenbef))

################################
#suffStat <- datasets_suffstats[[1]]$suffStat
#dump(c("suffStat"), "futureWork.R")
source("Thesis_results/futureWork.R")

res <- IOD(labelList = labelList, suffStat = suffStat)
print(suffStat)
lapply(res$Gi_PAG_list, renderAG)
#renderAG(res$Gi_PAG_list[[1]], width = 300, height = 300, type = "pdf")
#renderAG(res$Gi_PAG_list[[2]], width = 300, height = 300, type = "pdf")

load("~/IOD_Rpackage/Thesis_results/100randomPAGs.RData")
#renderAG(truePAGs[[1]], width = 300, height = 300, type = "pdf")

# Subset the rows where pvalue is greater than or equal to 0.05
# "A" "D" "C" "E"
pvals1 <- suffStat[[1]][[1]]$citestResults[suffStat[[1]][[1]]$citestResults$pvalue >= 0.05, ]
#"B" "C" "D" "E"
pvals2 <- suffStat[[1]][[2]]$citestResults[suffStat[[1]][[2]]$citestResults$pvalue >= 0.05, ]

#suffStat$cur_labels <- suffStat$citestResultsList[[1]]$labels
suffStat$cur_labels <- ssuffStat$labelList[[1]]

source("R/IOD_Helper.R")
iodCITest(2,3,4, suffStat)
iodCITest(2,3,{}, suffStat)

#oracle testresults to compare
suffStat<- list()
citestResultsList_out <- getCIResultsList(citest_type="oracleCI",
                                           setsToConsider=list(list("A","D","C","E"),list("B","C","D","E")),
                                           true.pag.amat = truePAGs[[1]])
suffStat$citestResultsList <- citestResultsList_out$citestResultsList
suffStat$labelList <- citestResultsList_out$labelList

