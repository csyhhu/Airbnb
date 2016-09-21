myOneHot <- function(num, nb_classes){
  
  row_num <- length(num)
  onehot <- array(0, dim=c(length(num),nb_classes))
  
  for (i in (1 : row_num)) {
    onehot[i, num[i]] <- 1
  }
  
  return (onehot)
}