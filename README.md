# Project M05

Repository of the M05 project made by Lucas and Gary.

## 1. Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## 2. Project installation

Go to the folder in which you want to import the git respository, open a terminal in it and type: 

```bash
git clone https://github.com/gatrikh/project_m05.git
```

Then move into the newly created folder: 

```bash 
cd project_m05/
```

Once in it, you must ensure that you have the correct environment to run the project. Firstly, the Python installation will be tested, indeed, **Python 3.8.8** is required: 

```bash
python test_environment.py
```

If the environment pass all test, you are ready to install the packages. Otherwise, you have to install **Python 3.8.8**. To install the packages, type the following command to ensure that you have the correct version of the required packages: 

```bash
pip install -r requirements.txt --upgrade
```

Now that the Python environment and the packages are installed, type the following commands to ensure that the project passes all tests. There are **13 tests and all must pass**: 

```bash
pytest tests
```

Once all the tests pass, the project is ready to run and you can go into `notebooks/Pipeline.ipynb` and run it. 

