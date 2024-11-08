#' Get the Property for Each Feature
#'
#' The `FeatureProperty` function retrieves the properties of each feature, allowing users to examine the characteristics of features selected by PreLect.
#'
#' @param X              Matrix or DataFrame. Raw count data with samples as rows and features as columns.
#' @param Y              Character or numeric vector. Labels for the data. If `task` is `coxPH`, please provide event labels.
#' @param PreLect_result PreLect output.  Lambda value, the intensity of regularization. Strongly suggested to be determined by the `LambdaTuning` process.
#' @param task           String. Specifies the type of task: either `classification`, `regression`, `multi-class classification`, or `coxPH` (default is `classification`).
#' @return A DataFrame with tendency, mean relative abundance, variance, and prevalence for each feature.
#' @importFrom matrixStats rowVars
#' @export
#' @examples」
#' 
#' set.seed(42)
#' n_samples <- 10
#' n_features <- 100
#' 
#' X_raw <- matrix(rnbinom(n_features * n_samples, size = 10, mu = 1), nrow = n_features, ncol = n_samples)
#' rownames(X_raw) <- paste0('feat',1:n_features)
#' X_scaled <- t(scale(t(X_raw)))  # feature-wise z-standardization
#' diagnosis <- c('CRC','CRC','health','CRC','health','CRC','health','health','CRC','CRC')
#' diagnosis <- factor(diagnosis, levels=c('health', 'CRC')) # assign the 'health' is control sample
#' pvlvec <- GetPrevalence(X_raw)
#' 
#' result <- PreLect(X_scaled, pvlvec, diagnosis, lambda=1e-4, task="classification")
#'
FeatureProperty <- function(X, Y, PreLect_result, task="classification"){
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  
  if (ncol(X) != length(Y)) {
    stop(paste("Label count mismatch: found", length(Y),
               "labels but data has", ncol(X), "samples."))
  }
  
  if (!task %in% c("classification", "regression", "multi-class classification", "coxPH")) {
    stop("Invalid task parameter. Choose either 'classification', 'regression', 'multi-class classification', or 'coxPH'.")
  }
  
  if(task == 'classification'){
    if(is.null(levels(Y))){
      control_sample <- Y[1]
    } else {
      control_sample <- levels(Y)[1]
    }
    y <- ifelse(Y == control_sample, 0, 1)
  } else if(task == 'regression') {
    y <- Y
    cutoff <- median(y)
  }
  
  x <- X
  for(i in 1:ncol(x)){
    x[,i] <- x[,i]/sum(x[,i])
  }
  meanRA <- rowMeans(x)
  var <- rowVars(X)
  pvl_globl <- get_prevalence(X)
  
  if(task == 'classification'){
    selected <- ifelse(PreLect_result$coef_table$coef != 0, 'Selected', 'others')
    pvl_case <- get_prevalence(X[,Y != control_sample])
    pvl_control <- get_prevalence(X[,Y == control_sample])
    mean_control <- rowMeans(X[,Y == control_sample])
    mean_case    <- rowMeans(X[,Y != control_sample])
    fc <- log2(mean_case) - log2(mean_control)
    out <- data.frame(
      'FeatName' = PreLect_result$coef_table$FeatName,
      'coef' = PreLect_result$coef_table$coef,
      'tendency' = PreLect_result$coef_table$tendency,
      'selected' = selected,
      'meanAbundance' = meanRA,
      'variance' = var,
      'prevalence' = as.numeric(pvl_globl),
      'prevalence_case' = as.numeric(pvl_case),
      'prevalence_control' = as.numeric(pvl_control),
      'logFC' = fc
    )
  }
  
  if(task == 'regression'){
    selected <- ifelse(PreLect_result$coef_table$coef != 0, 'Selected', 'others')
    pvl_low <- get_prevalence(X[,Y < cutoff])
    pvl_high <- get_prevalence(X[,Y >= cutoff])
    mean_low <- rowMeans(X[,Y < cutoff])
    mean_high <- rowMeans(X[,Y >= cutoff])
    fc <- log2(mean_high) - log2(mean_low)
    out <- data.frame(
      'FeatName' = PreLect_result$coef_table$FeatName,
      'coef' = PreLect_result$coef_table$coef,
      'tendency' = PreLect_result$coef_table$tendency,
      'selected' = selected,
      'meanAbundance' = meanRA,
      'variance' = var,
      'prevalence' = as.numeric(pvl_globl),
      'prevalence_high' = as.numeric(pvl_high),
      'prevalence_low' = as.numeric(pvl_low),
      'logFC' = fc
    )
  }
  
  if(task == 'multi-class classification'){
    out <- PreLect_result$coef_table
    anchor <- grep("tendency", colnames(PreLect_result$coef_table), value=T)
    selected <- rowSums(is.na(PreLect_result$coef_table[,anchor]))
    out$selected <-ifelse(selected == length(anchor), 'others', 'Selected')
    out$meanAbundance <- meanRA
    out$variance <- var
    out$prevalence <- as.numeric(pvl_globl)
  }
  
  if(task == 'coxPH'){
    selected <- ifelse(PreLect_result$coef_table$coef != 0, 'Selected', 'others')
    out <- PreLect_result$coef_table
    out$selected <- selected
    out$meanAbundance <- meanRA
    out$variance <- var
    out$prevalence <- as.numeric(pvl_globl)
  }
  
  return(out)
}