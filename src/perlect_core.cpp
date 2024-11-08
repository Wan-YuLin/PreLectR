#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
using namespace std;


// [[Rcpp::export]]
arma::vec get_prevalence(const arma::mat& X) {
  int p = X.n_rows;
  int n = X.n_cols;
  arma::vec pvl_vec(p);
  for(int i = 0; i < p; ++i) {
    pvl_vec(i) = sum(X.row(i) > 0) / static_cast<double>(n);
  }
  return pvl_vec;
}

// [[Rcpp::export]]
arma::mat initialize_M(const arma::mat& X) {
  arma::rowvec b(X.n_cols, fill::ones);
  arma::mat X_ = join_vert(X, b);
  return X_;
}

// [[Rcpp::export]]
arma::vec initialize_w(const arma::mat& X) {
  arma::vec w(X.n_rows+1, fill::zeros);
  return w;
}

// [[Rcpp::export]]
arma::vec initialize_pvl(const arma::vec& pvlvec) {
  arma::vec b(1, fill::ones);
  return join_vert(pvlvec, b);
}

// [[Rcpp::export]]
arma::vec sigmoid(const arma::vec& z) {
  return 1 / (1 + exp(-z));
}

// [[Rcpp::export]]
double BCE(const arma::vec& y_true, const arma::vec& y_pred, double epsilon = 1e-8) {
  int n = y_true.size();
  double loss = 0;
  arma::vec y_pred_clamped = clamp(y_pred, epsilon, 1 - epsilon);
  
  for(int i = 0; i < n; i++) {
    loss += -(y_true[i] * log(y_pred_clamped[i]) + (1 - y_true[i]) * log(1 - y_pred_clamped[i]));
  }
  return loss / n;
}

// [[Rcpp::export]]
arma::vec logistic_gradient(const arma::mat& X, const arma::vec& y, const arma::vec& w) {
  arma::vec diff = sigmoid(X.t() * w) - y;
  arma::vec grad = X * diff / X.n_cols;
  return grad;
}

// [[Rcpp::export]]
List proximal_GD_clr(const arma::mat& X, const arma::vec& y, const arma::vec& w, const arma::vec& v, 
                     const arma::vec& pvl, const arma::vec& thres, double lr, double alpha, double epsilon) {
  arma::vec gradient = logistic_gradient(X, y, w);
  arma::vec v_ = alpha * square(v) + (1 - alpha) * square(gradient) + epsilon;
  v_ = sqrt(v_);
  arma::vec z = w - (lr / v_) % pvl % gradient;
  arma::vec w_ = sign(z) % clamp(abs(z) - thres, 0.0, datum::inf);
  w_(w_.size() - 1) = z(z.size() - 1);
  return List::create(Named("update_w") = w_, 
                      Named("update_v") = v_);
}

// [[Rcpp::export]]
List prelect_clr(const arma::mat& X, const arma::vec& pvl,  const arma::vec& y, 
                 double lmbd = 1e-5, int max_iter = 10000, double tol = 1e-4, 
                 double lr = 0.001, double alpha = 0.9, double epsilon = 1e-8, bool echo = false) {
  
  arma::mat X_ = initialize_M(X);
  arma::vec current_w = initialize_w(X);
  arma::vec best_w = initialize_w(X);
  arma::vec current_v = initialize_w(X);
  arma::vec pvl_ = initialize_pvl(pvl);
  arma::vec thres = lmbd/pvl_;
  
  int min_iter = 0;
  double min_loss = 1e+10;
  double min_diff = 1e+10;
  
  for(int iter = 0; iter < max_iter; iter++) {
    arma::vec prev_w = current_w;
    arma::vec prev_v = current_v;
    
    List prox_result = proximal_GD_clr(X_, y, current_w, current_v, pvl_, thres, lr, alpha, epsilon);
    current_w = as<vec>(prox_result["update_w"]);
    current_v = as<vec>(prox_result["update_v"]);
    
    double lossvalue = BCE(sigmoid(X_.t() * current_w), y);
    double diff_w = norm(current_w - prev_w, 2);
    
    if (min_loss > lossvalue) {
      min_iter = iter; min_loss = lossvalue; min_diff = diff_w;
      best_w = current_w;
    }
    
    if (diff_w <= tol) {
      break;
    }
  }
  
  if (echo) {
    Rcout << "lambda : " << lmbd << "; minimum epoch : " << min_iter << "; minimum loss : " << min_loss << "; diff weight : " << min_diff << std::endl;
  }
  
  return List::create(Named("weights") = best_w, 
                      Named("loss") = min_loss,
                      Named("diff") = min_diff);
}

// [[Rcpp::export]]
arma::vec predict_proba(const arma::mat& X, const arma::vec& w) {
  arma::mat X_ = initialize_M(X);
  arma::vec res = sigmoid(X_.t() * w);
  return res;
}

// [[Rcpp::export]]
double MSE(const arma::vec& y_true, const arma::vec& y_pred, double epsilon = 1e-8) {
  int n = y_true.size();
  arma::vec diff = y_true - y_pred;
  return dot(diff, diff) / n;
}

// [[Rcpp::export]]
arma::vec mse_gradient(const arma::mat& X, const arma::vec& y, const arma::vec& w) {
  arma::vec diff = (X.t() * w) - y;
  arma::vec grad = 2 * X * diff / X.n_cols;
  return grad;
}

// [[Rcpp::export]]
List proximal_GD_reg(const arma::mat& X, const arma::vec& y, const arma::vec& w, const arma::vec& v, 
                     const arma::vec& pvl, const arma::vec& thres, double lr, double alpha, double epsilon) {
  arma::vec gradient = mse_gradient(X, y, w);
  arma::vec v_ = alpha * square(v) + (1 - alpha) * square(gradient) + epsilon;
  v_ = sqrt(v_);
  arma::vec z = w - (lr / v_) % pvl % gradient;
  arma::vec w_ = sign(z) % clamp(abs(z) - thres, 0.0, datum::inf);
  w_(w_.size() - 1) = z(z.size() - 1);
  return List::create(Named("update_w") = w_, 
                      Named("update_v") = v_);
}

// [[Rcpp::export]]
List prelect_reg(const arma::mat& X, const arma::vec& pvl,  const arma::vec& y, 
                 double lmbd = 1e-5, int max_iter = 10000, double tol = 1e-4, 
                 double lr = 0.001, double alpha = 0.9, double epsilon = 1e-8, bool echo = false) {
  
  arma::mat X_ = initialize_M(X);
  arma::vec current_w = initialize_w(X);
  arma::vec best_w = initialize_w(X);
  arma::vec current_v = initialize_w(X);
  arma::vec pvl_ = initialize_pvl(pvl);
  arma::vec thres = lmbd/pvl_;
  
  int min_iter = 0;
  double min_loss = 1e+10;
  double min_diff = 1e+10;
  
  for(int iter = 0; iter < max_iter; iter++) {
    arma::vec prev_w = current_w;
    arma::vec prev_v = current_v;
    
    List prox_result = proximal_GD_clr(X_, y, current_w, current_v, pvl_, thres, lr, alpha, epsilon);
    current_w = as<vec>(prox_result["update_w"]);
    current_v = as<vec>(prox_result["update_v"]);
    
    double lossvalue = MSE(sigmoid(X_.t() * current_w), y);
    double diff_w = norm(current_w - prev_w, 2);
    
    if (min_loss > lossvalue) {
      min_iter = iter; min_loss = lossvalue; min_diff = diff_w;
      best_w = current_w;
    }
    
    if (diff_w <= tol) {
      break;
    }
  }
  
  if (echo) {
    Rcout << "lambda : " << lmbd << "; minimum epoch : " << min_iter << "; minimum loss : " << min_loss << "; diff weight : " << min_diff << std::endl;
  }
  
  return List::create(Named("weights") = best_w, 
                      Named("loss") = min_loss,
                      Named("diff") = min_diff);
}

// [[Rcpp::export]]
double R2(NumericVector y_true, NumericVector y_pred) {
  int n = y_true.size();
  double y_mean = mean(y_true);
  double ss_res = 0.0;
  for (int i = 0; i < n; ++i) {
    double residual = y_true[i] - y_pred[i];
    ss_res += residual * residual;
  }
  double ss_tot = 0.0;
  for (int i = 0; i < n; ++i) {
    double diff = y_true[i] - y_mean;
    ss_tot += diff * diff;
  }
  double r2 = 1 - (ss_res / ss_tot);
  return r2;
}

// [[Rcpp::export]]
arma::mat initialize_M_multi(const arma::mat& X) {
  arma::rowvec b(X.n_cols, fill::ones);
  arma::mat X_ = join_vert(X, b);
  return X_;
}

// [[Rcpp::export]]
arma::mat initialize_w_multi(const arma::mat& X, const int& n_class) {
  arma::mat w(X.n_rows+1, n_class, fill::zeros);
  return w;
}

// [[Rcpp::export]]
arma::mat initialize_pvl_multi(const arma::mat& pvlmat) {
  arma::rowvec b(pvlmat.n_cols, fill::ones);
  arma::mat pvlmat_b = join_vert(pvlmat, b);
  return pvlmat_b;
}

// [[Rcpp::export]]
List prelect_multi_clr(const arma::mat& X, const arma::mat& pvl,  const arma::mat& y, 
                       double lmbd = 1e-5, int max_iter = 10000, double tol = 1e-4, 
                       double lr = 0.001, double alpha = 0.9, double epsilon = 1e-8, bool echo = false) {
  int n_class = y.n_cols;
  arma::mat X_ = initialize_M_multi(X);
  arma::mat current_w = initialize_w_multi(X, n_class);
  arma::mat prev_w    = initialize_w_multi(X, n_class);
  arma::mat best_w    = initialize_w_multi(X, n_class);
  arma::mat current_v = initialize_w_multi(X, n_class);
  arma::mat prev_v    = initialize_w_multi(X, n_class);
  arma::mat pvl_      = initialize_pvl_multi(pvl);
  arma::mat thres     = lmbd/pvl_;
  
  int min_iter = 0;
  double min_loss = 1e+10;
  double min_diff = 1e+10;
  
  for(int iter = 0; iter < max_iter; iter++) {
    prev_w = current_w;
    prev_v = current_v;
    double lossvalue = 0.0;
    double diff_w = 0.0;
    
    for(int l = 0; l < n_class; l++){
      List prox_result = proximal_GD_clr(X_, y.col(l), current_w.col(l), current_v.col(l), pvl_.col(l), thres.col(l), lr, alpha, epsilon);
      current_w.col(l) = as<vec>(prox_result["update_w"]);
      current_v.col(l) = as<vec>(prox_result["update_v"]);
      lossvalue += BCE(sigmoid(X_.t() * current_w.col(l)), y.col(l));
      diff_w += norm(current_w.col(l) - prev_w.col(l), 2);
    }
    
    lossvalue /= n_class;
    
    if (min_loss > lossvalue) {
      min_iter = iter; min_loss = lossvalue; min_diff = diff_w;
      best_w = current_w;
    }
    
    if (diff_w <= tol) {
      break;
    }
  }
  
  if (echo) {
    Rcout << "lambda : " << lmbd << "; minimum epoch : " << min_iter << "; minimum loss : " << min_loss << "; diff weight : " << min_diff << std::endl;
  }
  
  return List::create(Named("weights") = best_w, 
                      Named("loss") = min_loss,
                      Named("diff") = min_diff);
}

// [[Rcpp::export]]
double cox_loss(const arma::mat& X, const arma::vec& w, const arma::vec& events) {
  arma::vec prod = X * w;
  arma::vec theta = exp(prod);
  arma::vec theta_l = cumsum(theta);
  arma::vec res = prod - log(theta_l);
  return -1*sum(events % res);
}

// [[Rcpp::export]]
arma::vec cox_gradient(const arma::mat& X, const arma::vec& w, const arma::vec& events) {
  arma::vec theta = exp(X * w);
  arma::vec theta_l = cumsum(theta);
  arma::mat theta_l_v = cumsum(X.each_col() % theta, 0);
  arma::mat res = X - (theta_l_v.each_col() / theta_l);
  arma::mat g = -1*sum(events % res.each_col(), 0);
  return conv_to<vec>::from(g.t());
}

// [[Rcpp::export]]
List proximal_GD_cox(const arma::mat& X, const arma::vec& events, arma::vec w, arma::vec v,
                     const arma::vec& pvl, const arma::vec& thres,
                     double lr = 0.001, double alpha = 0.9, double epsilon = 1e-8) {
  arma::vec grad = cox_gradient(X, w, events);
  arma::vec v_ = alpha * square(v) + (1 - alpha) * square(grad) + epsilon;
  v_ = sqrt(v_);
  arma::vec z = w - (lr / v_) % pvl % grad;
  arma::vec w_ = sign(z) % clamp(abs(z) - thres, 0.0, datum::inf);
  return List::create(Named("update_w") = w_, 
                      Named("update_v") = v_);
}

// [[Rcpp::export]]
List prelect_cox(const arma::mat& X, const arma::vec& pvl,  const arma::vec& events, 
                 double lmbd = 1e-5, int max_iter = 10000, double tol = 1e-4, 
                 double lr = 0.001, double alpha = 0.9, double epsilon = 1e-8, bool echo = false) {
  arma::vec current_w(X.n_cols, fill::zeros);
  arma::vec current_v(X.n_cols, fill::zeros);
  arma::vec best_w(X.n_cols, fill::zeros);
  arma::vec thres = lmbd/pvl;
  
  int min_iter = 0;
  double min_loss = 1e+10;
  double min_diff = 1e+10;
  
  for (int iter = 0; iter < max_iter; ++iter) {
    arma::vec prev_w = current_w;
    
    List prox_result = proximal_GD_cox(X, events, current_w, current_v, pvl, thres, lr, alpha, epsilon);
    current_w = as<vec>(prox_result["update_w"]);
    current_v = as<vec>(prox_result["update_v"]);
    
    double lossvalue = cox_loss(X, current_w, events);
    double diff_w = norm(current_w - prev_w, 2);
    
    if (lossvalue < min_loss) {
      min_iter = iter; min_loss = lossvalue; min_diff = diff_w;
      best_w = current_w;
    }
    
    if (diff_w <= tol) {
      break;
    }
  }
  
  if (echo) {
    Rcout << "lambda : " << lmbd << "; minimum epoch : " << min_iter << "; minimum loss : " << min_loss << "; diff weight : " << min_diff << std::endl;
  }
  
  return List::create(Named("weights") = best_w, 
                      Named("loss") = min_loss,
                      Named("diff") = min_diff);
}

// [[Rcpp::export]]
double concordance_index(NumericVector time, NumericVector status, NumericVector risk) {
  int n = time.size();
  int comparable_pairs = 0;
  int concordant_pairs = 0;
  int ties = 0;
  
  // Iterate over all pairs (i, j)
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Only consider pairs where one is uncensored
      if ((status[i] == 1 && time[i] < time[j]) || (status[j] == 1 && time[j] < time[i])) {
        comparable_pairs++;
        
        if (risk[i] == risk[j]) {
          ties++;
        } else if ((risk[i] > risk[j] && time[i] < time[j]) || (risk[i] < risk[j] && time[i] > time[j])) {
          concordant_pairs++;
        }
      }
    }
  }
  
  return (concordant_pairs + 0.5 * ties) / comparable_pairs;
}