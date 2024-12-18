#' Multi-Class Feature Selection with PreLect
#'
#' `PreLectMultiClass` performs feature selection for multi-class classification by identifying class-specific coefficients based on input lambda regularization.
#'
#' @param X_scale  Matrix or DataFrame. Scaled data with samples as rows and features as columns, used for machine learning. If no scaled data is provided, raw count data may be used.
#' @param X_raw    Matrix or DataFrame. Raw count data with samples as rows and features as columns, used to calculate feature prevalence.
#' @param lambda   Numeric. Lambda value, the intensity of regularization. Strongly suggested to be determined by the `LambdaTuningMultiClass` process.
#' @param run_echo Logical. If TRUE, prints the training result (default is FALSE).
#' @param max_iter Integer. Maximum number of iterations taken for the solvers to converge (default is 10000).
#' @param tol      Numeric. Tolerance for stopping criteria (default is 1e-4).
#' @param lr       Numeric. Learning rate in RMSprop optimizer (default is 0.001).
#' @param alpha    Numeric. Smoothing constant in RMSprop optimizer (default is 0.9).
#' @param epsilon  Numeric. Small constant added to the denominator to improve numerical stability (default is 1e-8).
#' @return A list containing:
#' - `coef_table`: A data frame with class-specific coefficients for each feature, including both coefficient values and class tendency.
#' - `loss_value`: Final loss value achieved during training.
#' - `convergence`: Difference between the final and penultimate iterations, indicating convergence level.
#' @export
#' @examples
#' set.seed(42)
#' n_samples <- 10
#' n_features <- 100
#' 
#' X_raw <- matrix(rnbinom(n_features * n_samples, size = 10, mu = 1), nrow = n_features, ncol = n_samples)
#' X_scaled <- t(scale(t(X_raw)))  # feature-wise z-standardization
#' diagnosis <- c('CRC','CRC','control','Adenoma','Adenoma','CRC','control','control','CRC','CRC')
#' 
#' result <- PreLectMultiClass(X_scaled, X_raw, diagnosis, lambda = 0.01)
#' 
PreLectMultiClass <- function(X_scale, X_raw, Y, lambda, run_echo=FALSE,
                              max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8){
  
  if (is.data.frame(X_scale)) {
    X_scale <- as.matrix(X_scale)
  }
  
  if (is.data.frame(X_raw)) {
    X_raw <- as.matrix(X_raw)
  }
  
  if (is.null(rownames(X_raw))) {
    stop("No feature names provided. Please ensure feature names are set as rownames(X).")
  }
  
  if (!all(dim(X_raw) == dim(X_scale))) {
    stop(paste("Dimension mismatch: raw data has dimensions", 
               paste(dim(X_raw), collapse = " x "), 
               "but scaled data has dimensions", 
               paste(dim(X_scale), collapse = " x ")))
  }
  
  if (ncol(X_raw) != length(Y)) {
    stop(paste("Found input label with inconsistent numbers of samples:", length(Y),  
               ", but data had:", ncol(X_raw)))
  }
  
  if (length(unique(Y)) == 1) {
    stop(paste("The provided labels contain only one class :", unique(Y)))
  }
  
  category <- unique(Y)
  pvlMat <- matrix(0, nrow = nrow(X_raw), ncol = length(category))
  labMat <- matrix(0, nrow = ncol(X_raw), ncol = length(category))
  for(i in 1:length(category)){
    pvlMat[, i] <- get_prevalence(as.matrix(X_raw[, Y == category[i]]))
    labMat[, i] <- ifelse(Y == category[i], 1, 0)
  }
  
  res <- prelect_multi_clr(X_scale, pvlMat, labMat, lmbd=lambda,
                           max_iter=max_iter, tol=tol, lr=lr, alpha=alpha, epsilon=epsilon,echo=run_echo)
  
  select_df <- data.frame(matrix(data = NA, nrow = nrow(X_raw), ncol = (length(category)*2)+1))
  select_df[,1] <- rownames(X_raw)
  colnames(select_df)[1] <- 'FeatName'
  for(i in 1:length(category)){
    best_w <- res$weights[1:nrow(X_raw),i]
    select_df[, i*2] <- best_w
    select_df[, i*2+1]   <- ifelse(best_w > 0, category[i], 'Rest')
    select_df[, i*2+1][best_w == 0] <- NA
    colnames(select_df)[i*2] <- paste0('coef_',category[i])
    colnames(select_df)[i*2+1]   <- paste0('tendency_',category[i])
  }
  return(list('coef_table'=select_df, 'loss_value'=res$loss, 'convergence'=res$diff))
}

#' Automatically Lambda Scanning for Multi-Class Classification
#'
#' This function scans a range of lambda values from 1e-10 to 0.1, identifying the upper and lower boundaries that represent when lasso starts filtering and when it drops all features. The range is divided into `k` parts for examining lambda values.
#'
#' @param X_scale  Matrix or DataFrame. Scaled data with samples as rows and features as columns, used for machine learning. If no scaled data is provided, raw count data may be used.
#' @param X_raw    Matrix or DataFrame. Raw count data with samples as rows and features as columns, used to calculate feature prevalence.
#' @param Y        Character vector. Labels for the data.
#' @param step     Integer. The number of intervals for splitting within the upper and lower bounds when examining lambda values (default is 50).
#' @param run_echo Logical. If TRUE, prints the training result for each lambda being tested (default is FALSE).
#' @param max_iter Integer. Maximum number of iterations taken for the solvers to converge (default is 10000).
#' @param tol      Numeric. Tolerance for stopping criteria (default is 1e-4).
#' @param lr       Numeric. Learning rate in RMSprop optimizer (default is 0.001).
#' @param alpha    Numeric. Smoothing constant in RMSprop optimizer (default is 0.9).
#' @param epsilon  Numeric. Small constant added to the denominator to improve numerical stability (default is 1e-8).
#' @return A vector for examining log-lambda.
#' @export
#' @examples
#' set.seed(42)
#' n_samples <- 10
#' n_features <- 100
#' 
#' X_raw <- matrix(rnbinom(n_features * n_samples, size = 10, mu = 1), nrow = n_features, ncol = n_samples)
#' X_scaled <- t(scale(t(X_raw)))  # feature-wise z-standardization
#' 
#' diagnosis <- c('CRC','CRC','control','Adenoma','Adenoma','CRC','control','control','CRC','CRC')
#' 
#' lrange <- AutoScanningMultiClass(X_scaled, X_raw, diagnosis, step=30)
#' 
#' tuning_res <- LambdaTuningMultiClass(X_scaled, X_raw, diagnosis, lrange, outpath=getwd())
#' 
#' lmbd_picking <- LambdaDecision(tuning_res$TuningResult, tuning_res$PvlDistSummary)
#' 
#' # optimal lambda
#' lmbd_picking$opt_lmbd
#' 
#' # segmented regression visualization
#' library(patchwork)
#' lmbd_picking$selected_lmbd_plot/lmbd_picking$pvl_plot
#'
AutoScanningMultiClass <- function(X_scale, X_raw, Y, step=50, run_echo=FALSE,
                                   max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8){
  
  if (is.data.frame(X_scale)) {
    X_scale <- as.matrix(X_scale)
  }
  
  if (is.data.frame(X_raw)) {
    X_raw <- as.matrix(X_raw)
  }

  if (!all(dim(X_raw) == dim(X_scale))) {
    stop(paste("Dimension mismatch: raw data has dimensions",
               paste(dim(X_raw), collapse = " x "),
               "but scaled data has dimensions",
               paste(dim(X_scale), collapse = " x ")))
  }
  
  if (ncol(X_raw) != length(Y)) {
    stop(paste("Label count mismatch: found", length(Y),
               "labels but data has", ncol(X_raw), "samples."))
  }
  
  if (length(unique(Y)) == 1) {
    stop(paste("The provided labels contain only one class :", unique(Y)))
  }
  
  category <- unique(Y)
  pvlMat <- matrix(0, nrow = nrow(X_raw), ncol = length(category))
  labMat <- matrix(0, nrow = ncol(X_raw), ncol = length(category))
  for(i in 1:length(category)){
    pvlMat[, i] <- get_prevalence(as.matrix(X_raw[, Y == category[i]]))
    labMat[, i] <- ifelse(Y == category[i], 1, 0)
  }
  
  n_features <- nrow(X_raw)
  exam_range <- 1/10**seq(10,1)
  select_number <- c()
  pb <- txtProgressBar(min = 0, max = 10, style = 3)
  count <- 0
  for(lmbd in exam_range){
    count <- count + 1
    setTxtProgressBar(pb, count)
    exam_res <- prelect_multi_clr(X_scale, pvlMat, labMat, lmbd=lmbd, max_iter=max_iter, tol=tol, lr=lr, alpha=alpha, epsilon=epsilon,echo=run_echo)
    select_feat <- 0 
    for(l in 1:length(category)){
      best_w <- exam_res$weights[1:n_features, l]
      select_feat <- select_feat + sum(best_w != 0)
    }
    select_feat <- ifelse(select_feat > 0, select_feat/length(category), 0)
    select_number <- c(select_number, select_feat)
  }
  close(pb)
  
  lower <- NaN
  upper <- 0.1
  for(i in 1:length(exam_range)){
    if(is.nan(lower)){
      if(select_number[i] < n_features*0.9){
        lower <- exam_range[i]
      }
    }
    if(select_number[i] < 1){
      upper <- exam_range[i]
      break
    }
  }
  
  sequence <- seq(log(lower), log(upper), length.out = step)
  return(sequence)
}

#' Lambda Tuning for Multi-Class Classification
#'
#' This function performs automatic tuning of the lambda parameter. By iterating through a given range of lambda values. For each lambda, the function assesses feature prevalence, calculates various performance metrics, and logs results to output files for further analysis.
#'
#' @param X_scale   Matrix or DataFrame. Scaled data with samples as rows and features as columns, used for machine learning. If no scaled data is provided, raw count data may be used.
#' @param X_raw     Matrix or DataFrame. Raw count data with samples as rows and features as columns, used to calculate feature prevalence.
#' @param Y         Character vector. Labels for the data.
#' @param lmbdrange Numeric vector. Examining log-lambda vector, provided form `AutoScanningMultiClass`.
#' @param outpath   Character. The absolute-path of output folder for saving tuning result.
#' @param spl_ratio Numeric. The splits ratio for training part (default is 0.7).
#' @param run_echo  Logical. If TRUE, prints the training result for each lambda being tested (default is FALSE).
#' @param max_iter  Integer. Maximum number of iterations taken for the solvers to converge (default is 10000).
#' @param tol       Numeric. Tolerance for stopping criteria (default is 1e-4).
#' @param lr        Numeric. Learning rate in RMSprop optimizer (default is 0.001).
#' @param alpha     Numeric. Smoothing constant in RMSprop optimizer (default is 0.9).
#' @param epsilon   Numeric. Small constant added to the denominator to improve numerical stability (default is 1e-8).
#' @return A list containing two elements:
#' - `TuningResult`: A data frame of metrics for each lambda, including the number of selected features, prevalence, and performance metrics.
#' - `PvlDistSummary`: A data frame summarizing prevalence distribution statistics (min, max, quartiles) for selected features at each lambda.
#' @export
#' @examples
#' set.seed(42)
#' n_samples <- 10
#' n_features <- 100
#' 
#' X_raw <- matrix(rnbinom(n_features * n_samples, size = 10, mu = 1), nrow = n_features, ncol = n_samples)
#' X_scaled <- t(scale(t(X_raw)))  # feature-wise z-standardization
#' 
#' diagnosis <- c('CRC','CRC','control','Adenoma','Adenoma','CRC','control','control','CRC','CRC')
#' 
#' lrange <- AutoScanningMultiClass(X_scaled, X_raw, diagnosis, step=30)
#' 
#' tuning_res <- LambdaTuningMultiClass(X_scaled, X_raw, diagnosis, lrange, outpath=getwd())
#' 
#' lmbd_picking <- LambdaDecision(tuning_res$TuningResult, tuning_res$PvlDistSummary)
#' 
#' # optimal lambda
#' lmbd_picking$opt_lmbd
#' 
#' # segmented regression visualization
#' library(patchwork)
#' lmbd_picking$selected_lmbd_plot/lmbd_picking$pvl_plot
#'
LambdaTuningMultiClass <- function(X_scale, X_raw, Y, lmbdrange, outpath, spl_ratio=0.7, run_echo=FALSE,
                                   max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8){
  
  if (is.data.frame(X_scale)) {
    X_scale <- as.matrix(X_scale)
  }
  
  if (is.data.frame(X_raw)) {
    X_raw <- as.matrix(X_raw)
  }
  
  if (!all(dim(X_raw) == dim(X_scale))) {
    stop(paste("Found input data with inconsistent numbers of samples with raw data:", 
               paste(dim(X_raw), collapse = " x "), 
               "but scaled data:", 
               paste(dim(X_scale), collapse = " x ")))
  }
  
  if (ncol(X_raw) != length(Y)) {
    stop(paste("Label count mismatch: found", length(Y),
               "labels but data has", ncol(X_raw), "samples."))
  }
  
  if (length(unique(Y)) == 1) {
    stop(paste("The provided labels contain only one class :", unique(Y)))
  }
  
  if (!dir.exists(outpath)) {
    if (!dir.create(outpath, recursive = TRUE)) {
      stop("Error: Unable to create directory at ", outpath)
    }
  }
  
  n_features <- nrow(X_raw)
  n_samples <- ncol(X_raw)
  n_lambda <- length(lmbdrange)
  category <- unique(Y)
  loci_pvlMat <- matrix(0, nrow = nrow(X_raw), ncol = length(category))
  glob_pvlMat <- matrix(0, nrow = nrow(X_raw), ncol = length(category))
  labMat <- matrix(0, nrow = ncol(X_raw), ncol = length(category))
  for(i in 1:length(category)){
    glob_pvlMat[, i] <- get_prevalence(X_raw[, Y == category[i]])
    labMat[, i] <- ifelse(Y == category[i], 1, 0)
  }
  
  split <- TrainTextSplit(Y, spl_ratio, 'classification')
  X_train <- X_scale[, split$train_idx]
  X_test <- X_scale[, split$test_idx]
  y_train <- labMat[split$train_idx, ]
  y_test <- labMat[split$test_idx, ]
  tmpraw <- X_raw[, split$train_idx]
  for(i in 1:length(category)){
    tmpraw <- X_raw[, split$train_idx]
    loci_pvlMat[, i] <- get_prevalence(tmpraw[, y_train[, i] == 1]) + 1e-8
  }
  
  metrics <- data.frame(matrix(0, n_lambda, 6))
  colnames(metrics) <- c('Feature_number','Percentage', 'Prevalence', 'meanAUC', 'loss_history', 'error_history')
  pvl_dist <- data.frame()
  
  pb <- txtProgressBar(min = 0, max = n_lambda, style = 3)
  
  for(i in 1:n_lambda) {
    lambda <- exp(lmbdrange[i])
    res = prelect_multi_clr(X_train, loci_pvlMat, y_train, lmbd=lambda, max_iter=max_iter, tol=tol, lr=lr, alpha=alpha, epsilon=epsilon,echo=run_echo)
    selected_n <- c()
    selected_c <- c()
    selected_p <- c()
    selected_p_vec <- c(0)
    AUC_vec <- c()
    for(l in 1:length(category)){
      best_w <- res$weights[1:n_features, l]
      selected_n <- c(selected_n, sum(best_w != 0))
      selected_c <- c(selected_c, sum(best_w != 0)/n_features)
      selected_p <- c(selected_p, ifelse(sum(best_w != 0) == 0, 0, median(glob_pvlMat[best_w != 0, l])))
      
      for(k in 1:length(best_w)){
        if(best_w[k] != 0){
          selected_p_vec <- c(selected_p_vec, glob_pvlMat[k, l])
        }
      }
      if(sum(best_w != 0) > 3){
        perf <- evaluation(X_train, y_train[, l], X_test, y_test[, l], best_w, 'classification')
        AUC_vec <- c(AUC_vec, perf$AUC)
      } else {
        AUC_vec <- c(AUC_vec, 0)
      }
    }
    metrics[i,'Feature_number'] = mean(selected_n)
    metrics[i,'Percentage']     = mean(selected_c)
    metrics[i,'Prevalence']     = mean(selected_p)
    metrics[i,'loss_history']   = res$loss
    metrics[i,'error_history']  = res$diff
    metrics[i,'meanAUC']  = mean(AUC_vec)
    pvl_dist_row <- data.frame('llmbd'=log(lambda), 'max'= max(selected_p_vec), 'q3'=quantile(selected_p_vec, 0.75),
                               'q2'=median(selected_p_vec), 'q1'=quantile(selected_p_vec, 0.25), 'min'=min(selected_p_vec))
    pvl_dist <- rbind(pvl_dist, pvl_dist_row)
    setTxtProgressBar(pb, i)
  }
  close(pb)
  metrics$loglmbd <- lmbdrange
  rownames(pvl_dist) <- NULL
  write.csv(metrics, paste0(outpath,'/TuningResult.csv'), quote = F)
  write.csv(pvl_dist, paste0(outpath,'/Pvl_distribution.csv'), quote = F, row.names = FALSE)
  return(list('TuningResult'=metrics, 'PvlDistSummary'=pvl_dist))
}

#' Run LambdaTuningMultiClass in Parallel
#' 
#' This function performs `LambdaTuningMultiClass` in parallel to accelerate the tuning process.
#' 
#' @param X_scale   Matrix or DataFrame. Scaled data with samples as rows and features as columns, used for machine learning. If no scaled data is provided, raw count data may be used.
#' @param X_raw     Matrix or DataFrame. Raw count data with samples as rows and features as columns, used to calculate feature prevalence.
#' @param Y         Character vector. Labels for the data.
#' @param lmbdrange Numeric vector. Examining log-lambda vector, provided form `AutoScanningMultiClass`.
#' @param n_cores   Integer. Number of cores for parallel processing.
#' @param outpath   Character. The absolute-path of output folder for saving tuning result.
#' @param spl_ratio Numeric. The splits ratio for training part (default is 0.7).
#' @param run_echo  Logical. If TRUE, prints the training result for each lambda being tested (default is FALSE).
#' @param max_iter  Integer. Maximum number of iterations taken for the solvers to converge (default is 10000).
#' @param tol       Numeric. Tolerance for stopping criteria (default is 1e-4).
#' @param lr        Numeric. Learning rate in RMSprop optimizer (default is 0.001).
#' @param alpha     Numeric. Smoothing constant in RMSprop optimizer (default is 0.9).
#' @param epsilon   Numeric. Small constant added to the denominator to improve numerical stability (default is 1e-8).
#' @return A list containing two elements:
#' - `TuningResult`: A data frame of metrics for each lambda, including the number of selected features, prevalence, and performance metrics.
#' - `PvlDistSummary`: A data frame summarizing prevalence distribution statistics (min, max, quartiles) for selected features at each lambda.
#' @importFrom parallel makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach %dopar% foreach
#' @export
#' @examples
#' set.seed(42)
#' n_samples <- 10
#' n_features <- 100
#' 
#' X_raw <- matrix(rnbinom(n_features * n_samples, size = 10, mu = 1), nrow = n_features, ncol = n_samples)
#' X_scaled <- t(scale(t(X_raw)))  # feature-wise z-standardization
#' 
#' diagnosis <- c('CRC','CRC','control','Adenoma','Adenoma','CRC','control','control','CRC','CRC')
#' 
#' lrange <- AutoScanningMultiClass(X_scaled, X_raw, diagnosis)
#' 
#' # How many cores usage available
#' available_cores <- parallel::detectCores()
#' available_cores
#' 
#' tuning_res <- LambdaTuningMultiClassParallel(X_scaled, X_raw, diagnosis, lrange, n_cores=available_cores-2, outpath=getwd())
#'
#' lmbd_picking <- LambdaDecision(tuning_res$TuningResult, tuning_res$PvlDistSummary)
#' 
#' # optimal lambda
#' lmbd_picking$opt_lmbd
#' 
#' # segmented regression visualization
#' library(patchwork)
#' lmbd_picking$selected_lmbd_plot/lmbd_picking$pvl_plot
#'
LambdaTuningMultiClassParallel <- function(X_scale, X_raw, Y, lmbdrange, n_cores, outpath, spl_ratio=0.7, run_echo=FALSE,
                                           max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8){
  
  if (is.data.frame(X_scale)) {
    X_scale <- as.matrix(X_scale)
  }
  
  if (is.data.frame(X_raw)) {
    X_raw <- as.matrix(X_raw)
  }
  
  if (!all(dim(X_raw) == dim(X_scale))) {
    stop(paste("Found input data with inconsistent numbers of samples with raw data:", 
               paste(dim(X_raw), collapse = " x "), 
               "but scaled data:", 
               paste(dim(X_scale), collapse = " x ")))
  }
  
  if (ncol(X_raw) != length(Y)) {
    stop(paste("Label count mismatch: found", length(Y),
               "labels but data has", ncol(X_raw), "samples."))
  }
  
  if (length(unique(Y)) == 1) {
    stop(paste("The provided labels contain only one class :", unique(Y)))
  }
  
  if (!dir.exists(outpath)) {
    if (!dir.create(outpath, recursive = TRUE)) {
      stop("Error: Unable to create directory at ", outpath)
    }
  }
  
  n_features <- nrow(X_raw)
  n_samples <- ncol(X_raw)
  n_lambda <- length(lmbdrange)
  category <- unique(Y)
  loci_pvlMat <- matrix(0, nrow = nrow(X_raw), ncol = length(category))
  glob_pvlMat <- matrix(0, nrow = nrow(X_raw), ncol = length(category))
  labMat <- matrix(0, nrow = ncol(X_raw), ncol = length(category))
  for(i in 1:length(category)){
    glob_pvlMat[, i] <- get_prevalence(X_raw[, Y == category[i]])
    labMat[, i] <- ifelse(Y == category[i], 1, 0)
  }
  
  split <- TrainTextSplit(Y, spl_ratio, 'classification')
  X_train <- X_scale[, split$train_idx]
  X_test <- X_scale[, split$test_idx]
  y_train <- labMat[split$train_idx, ]
  y_test <- labMat[split$test_idx, ]
  tmpraw <- X_raw[, split$train_idx]
  
  for(i in 1:length(category)){
    tmpraw <- X_raw[, split$train_idx]
    loci_pvlMat[, i] <- get_prevalence(tmpraw[, y_train[, i] == 1]) + 1e-8
  }
  
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl))
  
  metrics <- foreach(llambda = lmbdrange, .combine = rbind) %dopar% {
    res <- prelect_multi_clr(X_train, loci_pvlMat, y_train, lmbd=exp(llambda), max_iter=max_iter, tol=tol, lr=lr, alpha=alpha, epsilon=epsilon,echo=run_echo)
    selected_n <- c()
    selected_c <- c()
    selected_p <- c()
    selected_p_vec <- c(0)
    AUC_vec <- c()
    for(l in 1:length(category)){
      best_w <- res$weights[1:n_features, l]
      selected_n <- c(selected_n, sum(best_w != 0))
      selected_c <- c(selected_c, sum(best_w != 0)/n_features)
      selected_p <- c(selected_p, ifelse(sum(best_w != 0) == 0, 0, median(glob_pvlMat[best_w != 0, l])))
      
      for(k in 1:length(best_w)){
        if(best_w[k] != 0){
          selected_p_vec <- c(selected_p_vec, glob_pvlMat[k, l])
        }
      }
      
      if(sum(best_w != 0) > 3){
        perf <- evaluation(X_train, y_train[, l], X_test, y_test[, l], best_w, 'classification')
        AUC_vec <- c(AUC_vec, perf$AUC)
      } else {
        AUC_vec <- c(AUC_vec, 0)
      }
    }
    data.frame(
      'Feature_number' = mean(selected_n),
      'Percentage' = mean(selected_c),
      'Prevalence' = mean(selected_p),
      'AUC' = mean(AUC_vec),
      'loss_history' = res$loss,
      'error_history' = res$diff,
      'loglmbd' = llambda,
      'llmbd' = llambda,
      'max' = max(selected_p_vec),
      'q3'=quantile(selected_p_vec, 0.75),
      'q2'=median(selected_p_vec),
      'q1'=quantile(selected_p_vec, 0.25),
      'min'=min(selected_p_vec)
    )
  }
  rownames(metrics) <- NULL
  pvl_dist <- metrics[,8:13]
  metrics <- metrics[,1:7]
  write.csv(metrics, paste0(outpath,'/TuningResult.csv'), quote = F)
  write.csv(pvl_dist, paste0(outpath,'/Pvl_distribution.csv'), quote = F, row.names = FALSE)
  return(list('TuningResult'=metrics, 'PvlDistSummary'=pvl_dist))
}
