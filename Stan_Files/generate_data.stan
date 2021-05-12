data{
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=2> k; //number of classes
  int<lower=1> ann_per_task;
  int<lower=1> workers_random_order[t,w];
  simplex[k] tau;
  simplex[k] pi[w,k];
}

generated quantities{
  int ann[t,w];
  int tc[t];
  for(yt in 1:t){
    for(yw in 1:w){
      ann[yt,yw] = -1;
    }
  }

  for(yt in 1:t){
    tc[yt] = categorical_rng(tau);
    for(ya in 1:ann_per_task){
      ann[yt,workers_random_order[yt,ya]] = categorical_rng(pi[workers_random_order[yt,ya],tc[yt]]); 
    }
  }
}