functions {

vector[] log_transpose(vector[] m, int k, int k2) {
    // int k = size(m);
    vector[k] log_m_t[k2];
    for (yreal in 1:k)
        for (yemitted in 1:k2)
            log_m_t[yemitted,yreal] = log(m[yreal,yemitted]);
    return log_m_t;
}
}



data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations
  
  int<lower=2> k; //number of internal classes
  int<lower=0> k2; //sum of external and internal classes
  int<lower=1,upper=t> t_A[a]; // the item the n-th annotation belongs to
  int<lower=1,upper=w> w_A[a]; // the annotator which produced the n-th annotation
  int<lower=1,upper=k2> ann[a]; // the annotation
  vector[k] tau_prior;
  vector[k2] pi_prior[k];
}

parameters {
  simplex[k] tau;
  simplex[k2] pi[w,k];
}

transformed parameters {
  // log_p_t_C[_t][_k] is the log of the probability that t_C=_k for task _t 
  vector[k] log_p_t_C[t];
  vector[k] t_C[t]; //the true class distribution of each item

  // Initialize with the prior
  
  log_p_t_C = rep_array(log(tau), t);
  
  // Update log_p_t_C with each of the annotations
  
  { 
        // Make the log and transpose the emission matrix
        vector [k] log_emission_t[w,k2];
        
        for(yw in 1:w){
          log_emission_t[yw] = log_transpose(pi[yw],k,k2);
        }
                
        // Update each task with the information contributed by its annotations 
        
        for (ya in 1:a){
            log_p_t_C[t_A[ya]] += log_emission_t[w_A[ya],ann[ya]];
        }
  }

  // Compute the probabilities from the logs

  for(yt in 1:t)
    t_C[yt] = softmax(log_p_t_C[yt]);

}


model {

  // Prior over pi

  for(yw in 1:w){ 
    for(yk in 1:k){
      pi[yw,yk]~ dirichlet(pi_prior[yk]);
    }
  }
  // Prior over tau
  tau ~ dirichlet(tau_prior);

  // Observation model

  // Summing over hidden var t_C
  for (yt in 1:t)
     target += log_sum_exp(log_p_t_C[yt]);
}