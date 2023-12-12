Projet IFT712 - Technique d'apprentissage
==============================


## Introduction

Notre projet vise à explorer en profondeur diverses méthodologies de classification appliquées à un ensemble de données Kaggle dans le but de classer des feuilles d’arbre.

L'objectif principal de ce projet est d'évaluer et de comparer six méthodes de classification distinctes dans le contexte de l'identification des feuilles d'arbres. Notre approche met l'accent sur l'utilisation de la validation croisée et de l'ajustement des hyperparamètres pour optimiser un modèle afin de trouver les solutions les plus efficaces pour la classification des feuilles.

Nos méthodes de classification comprennent AdaBoost, les arbres de décision, le perceptron, les réseaux neuronaux, les machines à vecteurs de support (SVM) et les K-means. 


## Organisation du projet 

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   ├── data_manager.py
        |   ├── constants.py
        │   └── pretreatment.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── model.py
        │   ├── ada_boost.py
        │   ├── decision_tree.py
        │   ├── k_means.py
        │   ├── neural_network.py
        │   ├── perceptron.py
        |   └── svm.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
