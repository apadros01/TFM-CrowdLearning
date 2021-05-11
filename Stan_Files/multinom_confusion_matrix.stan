data{
  int<lower=0> n; //number of tasks
	int<lower=1> k; //number of classes
  int<lower=0> x[n,k]; //matrix of annotations
  vector[k] prior_p;
  vector[k] prior_pi;
}

parameters{
	simplex[k] p;
  simplex[k] pi[k]; 

}

transformed parameters{
  vector[k] log_tc[n]; 
  for (i in 1:n){
    log_tc[i] = log(p);
    for (j in 1:k){
      log_tc[i,j] = log_tc[i,j] + multinomial_lpmf(x[i,] | pi[j]);
    }

  }

}


model{
 
	p ~ dirichlet(prior_p);
  for (i in 1:k){
    pi[i] ~ dirichlet(prior_pi);
  }

  for (i in 1:n){
    target += log_sum_exp(log_tc[i]);
  }
  
}

generated quantities{
  int<lower=1,upper=3> tc[n];
  for (i in 1:n){
    tc[i] = categorical_logit_rng(softmax(log_tc[i]));
  }

}

