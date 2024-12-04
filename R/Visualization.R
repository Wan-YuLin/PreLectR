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
#' @examples
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
#' prop <- FeatureProperty(X_raw, diagnosis, result)
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
      control_sample <- as.character(sort(Y[1]))
    } else {
      control_sample <- levels(Y)[1]
    }
    y <- ifelse(Y == control_sample, 0, 1)
    case_sample <- as.character(unique(Y)[unique(Y) != control_sample])
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
    selected <- ifelse(PreLect_result$coef_table$coef != 0, 'Selected', 'Others')
    pvl_case <- get_prevalence(X[,Y != control_sample])
    pvl_control <- get_prevalence(X[,Y == control_sample])
    mean_control <- rowMeans(X[,Y == control_sample])
    mean_case    <- rowMeans(X[,Y != control_sample])
    fc <- log2(mean_case+1) - log2(mean_control+1)
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
    selected <- ifelse(PreLect_result$coef_table$coef != 0, 'Selected', 'Others')
    pvl_low <- get_prevalence(X[,Y < cutoff])
    pvl_high <- get_prevalence(X[,Y >= cutoff])
    mean_low <- rowMeans(X[,Y < cutoff])
    mean_high <- rowMeans(X[,Y >= cutoff])
    fc <- log2(mean_high+1) - log2(mean_low+1)
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
    anchor <- grep("tendency", colnames(PreLect_result$coef_table), value=T)
    selected <- rowSums(is.na(PreLect_result$coef_table[,anchor]))
    selected <- ifelse(selected != length(anchor), 'Selected', 'Others')
    out <- PreLect_result$coef_table
    out$selected <-selected
    out$meanAbundance <- meanRA
    out$variance <-var
    out$prevalence <- as.numeric(pvl_globl)
    for(y in unique(Y)){
      pvl_ <- GetPrevalence(X[, Y == y])
      out[[paste0('prevalence_', y)]] <- as.numeric(pvl_)
    }
  }
  
  if(task == 'coxPH'){
    selected <- ifelse(PreLect_result$coef_table$coef != 0, 'Selected', 'others')
    out <- data.frame(
      'FeatName' = PreLect_result$coef_table$FeatName,
      'coef' = PreLect_result$coef_table$coef,
      'tendency' = PreLect_result$coef_table$tendency,
      'selected' = selected,
      'meanAbundance' = meanRA,
      'variance' = var,
      'prevalence' = as.numeric(pvl_globl),
    )
  }
  return(out)
}

#' Taxonomy tendency visualization
#'
#' This function visualizes the properties of taxonomy derived from selected features and generates a plot summarizing the effect size of each taxonomy unit.
#'
#' @param feat_prop   DataFrame. A data frame containing feature properties, generated by `FeatureProperty`.
#' @param taxa_table  DataFrame. A data frame containing taxonomy information for each feature, generated by `DADA2Adapter`.
#' @param taxa_level  Character. Specifies the column name in `taxa_table` that corresponds to the desired taxonomy level for analysis (default is "Order").
#' @param pvl_filter  Numeric. A threshold for the prevalence of taxonomy units. Taxa with prevalence below this threshold will be excluded (default is 0.4).
#' @return A list containing two elements:
#' - `effectSizePlot`: A ggplot object visualizing the effect size of each taxonomy unit.
#' - `selectedInfo`:A data frame summarizing the information for each selected feature, including coefficients and taxonomy assignments.
#' @importFrom ggplot2 ggplot aes coord_flip scale_x_discrete ylab theme annotate geom_boxplot unit
#' @export
#' @examples
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
#' prop <- FeatureProperty(X_raw, diagnosis, result)
#'
#' TaxaProperty(prop, taxa, taxa_level = "Order", pvl_filter = 0.4)
#' 
TaxaProperty <- function(feat_prop, taxa_table, taxa_level = 'Order', pvl_filter = 0.4){
  
  if (!"coef" %in% colnames(feat_prop) ) {
    stop("Multi-class classification is not applicable in current version")
  }
  
  if (nrow(feat_prop) != nrow(taxa_table)) {
    stop(paste("feature property mismatch: found", nrow(feat_prop),
               "feature but taxa table has", nrow(taxa_table), "taxa."))
  }
  
  if (!taxa_level %in% colnames(taxa_table) ) {
    stop("The specified taxa label is not present in the taxa table.")
  }
  
  selected <- feat_prop[feat_prop$selected == 'Selected', ]
  selected$taxa <- taxa_table[selected$FeatName, taxa_level]
  selected <- selected[selected$taxa != 'unclassified', ]
  
  mean_pvl <- c()
  for(t in unique(selected$taxa)){
    mean_pvl <- c(mean_pvl, mean(selected$prevalence[selected$taxa == t]))
  }
  names(mean_pvl) <- unique(selected$taxa)
  choose_taxa <- names(mean_pvl[mean_pvl > pvl_filter])
  selected <- selected[selected$taxa %in% choose_taxa, ]
  
  mean_coef <- c()
  for(x in unique(selected$taxa)){
    mean_coef <- c(mean_coef, mean(selected$coef[selected$taxa == x]))
  }
  names(mean_coef) <- unique(selected$taxa)
  selected$taxa <- factor(selected$taxa, levels = names(sort(mean_coef)))
  
  control_sample <- unique(selected$tendency[selected$coef < 0])
  case_sample <- unique(selected$tendency[selected$tendency != control_sample])
  y_label <- paste0("coefficient (",case_sample," vs. ",control_sample,")",collapse = '')
  
  p1 <- ggplot(selected, aes(x = taxa, y = coef, fill=tendency)) + coord_flip() + 
    scale_x_discrete() + ylab(y_label) + 
    theme(panel.background = ggplot2::element_rect(fill = 'transparent'),
          panel.grid = ggplot2::element_blank(),
          axis.ticks.length = ggplot2::unit(0.4,"lines"),
          axis.ticks = ggplot2::element_line(color='black'),
          axis.line = ggplot2::element_line(colour = "black"),
          axis.text.y = ggplot2::element_text(colour='black',size=12),
          axis.title.y = ggplot2::element_blank(),
          legend.title = ggplot2::element_blank(),
          legend.position = 'top')
  
  sing = 1
  for (i in 1:(length(unique(selected$taxa))-1)){
    sing = sing * -1
    p1 <- p1 + annotate('rect', xmin = i+0.5, xmax = i+1.5, ymin = -Inf, ymax = Inf,
                        fill = ifelse(sing > 0, 'white', 'gray95'))
  }
  p1 <- p1 + geom_boxplot(outlier.shape = NA, width = 0.5) 
  
  return(list('effectSizePlot'=p1, 'selectedInfo'=selected))
}