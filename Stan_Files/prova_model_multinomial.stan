data{
	int<lower=0> n; //numero d'observacions
	int<lower=1> k; //numero de possibles outcomes
  int<lower=0> x[n,k]; //resultats de cada 1000 tirades
  vector[k] probs_p;
}

parameters{
	simplex[k] p; 
}

model{
  p ~ dirichlet(probs_p);
  for (i in 1:n) {
        x[i,] ~ multinomial(p);
    }
}