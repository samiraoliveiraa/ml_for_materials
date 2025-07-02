## Relevance of local attributes in machine learning for formation energy
**Samira de Oliveira Moreira, Felipe Crasto de Lima**

**Abstract:** Materials have always been essential to human development, with their discovery evolving from empirical approaches to data-driven methods. Despite high predictive performance, many machine learning models lack interpretability. This work proposes interpretable models to predict material formation energy using only local attributes, such as atomic properties and bonding features. The methods used were SISSO and Explainable Boosting Machine (EBM). Results show that these attributes effectively capture structure–stability relationships, yielding simple yet interpretable models with satisfactory performance.

### Files:

- ```data/``` — contains files related to data extraction.
  - ```bonds.py```: Script for extracting chemical bonds from materials using POSCAR files obtained from the Materials Project.
  - ```features.py```: Script for generating bond-related features.
  - ```full_data.7z```: Compressed dataset including both bond information and the features used in the models.
  - ```material_data.csv```: Dataset containing the features used in the models in CSV format.
  - ```material_data.dat```: Dataset containing the same features as above, in DAT format.
  - ```data_information.ipynb```: Jupyter notebook with general information about the dataset.

- ```ebm/``` — contains files related to training and analyzing the Explainable Boosting Machine model.
  - ```ebm_optuna.py```: Script for training and optimizing the EBM model using Optuna.
  - ```ebm_model_optimized.joblib```: File containing the trained EBM model.
  - ```ebm.ipynb```: Jupyter notebook with EBM model analysis and visualization.
 
- ```random_forest/``` — contains files related to the training of the Random Forest model.
  - ```rfr_optuna.py```: Script for training and optimizing the RF model using Optuna.
  - ```rfr.ipynb```: Jupyter notebook with RF model analysis and visualization.

- ```sisso/``` — contains files related to SISSO model analysis. Note: The models were trained in a high-performance computing (HPC) environment.
  - ```sisso-analysis.ipynb```: Jupyter notebook for analyzing the results of the SISSO models.
