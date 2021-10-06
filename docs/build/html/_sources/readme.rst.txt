Project M05
===========

Repository of the M05 project made by Lucas and Gary.

The KDD Cup 1999 dataset (Third International Knowledge Discovery and Data Mining Tools Competition in 1999) is used to create a **Network intrusions detector**. 

=============================
0. Initial project hypothesis
=============================

Our initial project hypthesis is: 

**“We can achieve a good classification accuracy (>** **90%** **on the test set) in classifying the network connections into the correct labels.”** 

=======================
1. Project installation
=======================

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

================
2. Final results
================

Using an Random Forest Classification model, we achieved a classification accuracy of **0.9999** with **979650** correctly predicted labels and **46** incorrectly predicted labels. 

**We can thus verify our initial hypothesis that we would achieve an accuracy of more than 90% on the test set.**

