rm(list=ls())
library(FCI.Utils)

types =  c("chain4", "iv", "collfork", "discr1_c", "1be", "4be")

for (type in types) {
  adag_out <- getDAG(type)
  true_pag <- getTruePAG(adag_out$dagg)
  true_adjm <- true_pag@amat
  true_sepset <- getPAGImpliedSepset(true_adjm)
  formatSepset(true_sepset)

  renderAG(true_adjm, output_folder="./",
           fileid = type, type = "pdf", width = 700, height = 500,
           labels=colnames(true_adjm), add_index = FALSE)
}


