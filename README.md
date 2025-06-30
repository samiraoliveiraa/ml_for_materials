## Relevance of local attributes in machine learning for formation energy
**Samira de Oliveira Moreira, Felipe Crasto de Lima**

**Abstract:** Materials have always been essential to human development, with their discovery evolving from empirical approaches to data-driven methods. Despite high predictive performance, many machine learning models lack interpretability. This work proposes interpretable models to predict material formation energy using only local attributes, such as atomic properties and bonding features. The methods used were SISSO and Explainable Boosting Machine (EBM). Results show that these attributes effectively capture structure–stability relationships, yielding simple yet interpretable models with satisfactory performance.

### Files:

- ```data/``` — contains files related to data extraction.
  - ```bonds.py```: Script for extracting chemical bonds from materials using POSCAR files obtained from the Materials Project.
  - ```data_information.ipynb```: Jupyter notebook with general information about the dataset.
  - ```features.py```: Script for generating bond-related features.
  - ```full_data.7z```: Compressed dataset including both bond information and the features used in the models.
  - ```material_data.csv```: Dataset containing the features used in the models in CSV format.
  - ```material_data.dat```: Dataset containing the same features as above, in DAT format.
