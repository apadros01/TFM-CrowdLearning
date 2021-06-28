# Master on Fundamentals of Data Science: Final Project
## Facing the Label-Switching problem when using generic inference platforms for crowd annotation models

### Àlex Padrós Zamora

In this Master Thesis we study some classical approaches for crowd annotation models such as the pooled multinomial model or the Dawid-Skene models. These models try to learn from the crowd, which is not required to be expert. In particular, the problem of label aggregation that we deal with can be seen as a probabilistic graphical model. We propose an algorithm that aims to solve the problem of label-switching for generic inference platforms such as STAN without any previous intervention to the optimization/sampling method. We also study its performance by means of the Kullback-Leibler divergence, where we see that the results are better after applying our proposed correction.

In the "Notebooks" folder we find all the"Jupyter Notebooks that are necessary to follow the thesis. Notebok [pooled_multinomial.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/pooled_multinomial.ipynb) contains the Python code that is necessary to compile for obtaining the results of the pooled multinomial model. This version is the first one that is explained in section 5.1 of the project report, the one that takes as an input a matrix of accumulated annotations. The version that requires as an input each worker's annotation is contained in [pooled_multinomial_optimal.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/pooled_multinomial_optimal.ipynb). Notebook [dawid_skene_models.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/dawid_skene_models.ipynb) contains the Python code for compiling each of the three versions of the Dawid-Skene models: General DS, Conditional DS and Homogeneous DS. Notebook [extern_classes.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/extern_classes.ipynb) a general version of the DS models. That is, the addition of an extern class. 
