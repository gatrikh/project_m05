# Project M05

Repository of the M05 project made by Lucas and Gary.

The KDD Cup 1999 dataset (Third International Knowledge Discovery and Data Mining Tools Competition in 1999) is used to create a **Network intrusions detector**. 

## 0. Initial project hypothesis

Our initial project hypthesis is: 

**“We can achieve a good classification accuracy (>** **90%** **on the test set) in classifying the network connections into the correct labels.”** 

## 1. Project Organization

+---.github
|   +---workflows
+---data 
+---docs
|   +---build
|   +---source
+---nid
|   +---models
|   +---tests
+---presentation

## 2. Project installation

It is recommended to install an environment specific to this package with python 3.8:

```bash 
conda create --name nidenv python=3.8
```

and to activate it : 

```bash 
conda activate nidenv
```

Go to the folder in which you want to import the git respository, open a terminal in it and type : 

```bash
git clone https://github.com/gatrikh/project_m05.git
```

Then move into the newly created folder : 

```bash 
cd project_m05/
```

Once in it, you have to install all the dependencies of the project :

```bash
pip install -e .
```

From here, you can launch the project. To do this, we need to execute the main function. 
the main function takes 4 arguments as parameters: 
* -path PATH    : Path of KDD99 Cup dataset in .gz format **IS REQUIRED**
* -test         : Running the package tests
* -fig          : Saving figures of confusion matrices in working directory
* -train        : Retrain completely the model and save it in the working directory

we can for example launch the base code with its tests and confusion matrices :

```bash
python -m nid -path data/kddcup.data.gz -test -fig
```

Once the test is launched, you can check the coverage of the package : 

```bash
coverage report -m
```

if you juste want to lauch the base code :

```bash
python -m nid -path data/kddcup.data.gz
```

**NOTE** 
You can't use the -train argument with -test and -fig
if you want to retrain the model you can use this command :

```bash
python -m nid -path data/kddcup.data.gz -train
```

if you want to change the parameters of the random forest, you can do it in the file nid/__main__.py. In this file, at line 71, there is a dict "parameters", you can add parameters, as many as you want, as long as it respects the parameters of the random forest classifier of sklearn.ensemble (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Of course, once this is done you have to run the above command. The program will choose the model with the best results on the test set

## 3. Final results

Using an Random Forest Classification model, we achieved a classification accuracy of **0.9999** with **979650** correctly predicted labels and **46** incorrectly predicted labels. 

with these results, we cannot reject our basic hypothesis: 

**“We can achieve a good classification accuracy (>** **90%** **on the test set) in classifying the network connections into the correct labels.”** 

