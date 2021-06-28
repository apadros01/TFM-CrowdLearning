# Master on Fundamentals of Data Science: Final Project
## Facing the Label-Switching problem when using generic inference platforms for crowd annotation models

### Àlex Padrós Zamora

In this Master Thesis we study some classical approaches for crowd annotation models such as the pooled multinomial model or the Dawid-Skene models. These models try to learn from the crowd, which is not required to be expert. In particular, the problem of label aggregation that we deal with can be seen as a probabilistic graphical model. We propose an algorithm that aims to solve the problem of label-switching for generic inference platforms such as STAN without any previous intervention to the optimization/sampling method. We also study its performance by means of the Kullback-Leibler divergence, where we see that the results are better after applying our proposed correction.

In the "Notebooks" folder we find all the Jupyter Notebooks that are necessary to follow the thesis. Notebok [pooled_multinomial.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/pooled_multinomial.ipynb) contains the Python code that is necessary to compile for obtaining the results of the pooled multinomial model. This version is the first one that is explained in section 5.1 of the project report, the one that takes as an input a matrix of accumulated annotations. The version that requires as an input each worker's annotation for each task is contained in [pooled_multinomial_optimal.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/pooled_multinomial_optimal.ipynb). Notebook [dawid_skene_models.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/dawid_skene_models.ipynb) contains the Python code for compiling each of the three versions of the Dawid-Skene models: General DS, Conditional DS and Homogeneous DS. Notebook [extern_classes.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/extern_classes.ipynb) a general version of the DS models. That is, the addition of an extern class. The experiments of this project are included in [generate_data.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/generate_data.ipynb), [initial_points_help.ipynb](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Notebooks/initial_points_help.ipynb).
The first one contains the experiments that do not give any initial point to the optimization algorithm, as well as the plots. The second one contains the same functions than the previous one plus others that slightly modify our pipeline: Firstly, it applies majority voting in order to obtain an initial point for the pooled multinomial model. Then, this model is used for estimating some new initial points for the general DS.

In the "Stan_files" folder we find all the STAN codes that are necessary to compile the previous notebooks. File [multinom_confusion_matrix.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/multinom_confusion_matrix.stan) contains the pooled multinomial model, the version that takes as an input a matrix of accumulated annotations. File [Multinomial.fit_and_consensus.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/Multinomial.fit_and_consensus.stan) contains the version of the pooled multinomial model that takes as an input each worker's annotation for each task. [general_dawid-skene.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/general_dawid-skene.stan), [conditional_dawid-skene.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/conditional_dawid-skene.stan) and [homogeni_dawid-skene.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/homogeni_dawid-skene.stan) contain the Dawid-Skene models. [extern_classes.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/extern_classes.stan) contains the STAN code for the model that accepts extern classes. [generate_data.stan](https://github.com/apadros01/TFM-CrowdLearning/blob/main/Stan_Files/generate_data.stan) contains the STAN code that, given a generative distribution, it generates a dataset of annotations. 

In the "txt_files" folder we find all the txt files that contain the values of the experiments, which have been plotted in the notebooks.
