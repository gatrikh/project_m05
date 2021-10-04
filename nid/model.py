import pandas as pd
import os as os
import pickle as pickle
from sklearn.ensemble import RandomForestClassifier
import itertools
from tqdm import tqdm
from . import visualize as vis

def get_x_y_from_df(df):
    """get the features and labels from the dataset
     
    Parameters
    ----------
    df : pandas.DataFrame
        the dataset
    
    Returns
    -------
    x : pandas.DataFrame
        the features of the dataset
    y : pandas.core.series.Series
        the labels of the dataset
    """  
    x = df.drop(["label"], axis="columns")
    y = df["label"]

    return x, y

def generate_models_parameters(parameters): 
    """give a dict of individual model parameters from a list of parameters
     
    Parameters
    ----------
    parameters : dict
        dict that contains a list of parameters to test
    
    Returns
    -------
    models_dict : dict
        dict that contains individuals parameters for each model to test
    """  
    
    models_parameters = []
    name_parameters = []
    for key in parameters:
        models_parameters.append(parameters[key])
        name_parameters.append(key)

    models_dict = dict()
    for model_count, model_parameters in enumerate(itertools.product(*models_parameters)):
        model_dict = dict()
        for parameter_count, parameter in enumerate(model_parameters):
            model_dict[name_parameters[parameter_count]] = parameter
        models_dict[model_count] = model_dict

    return models_dict

def train_validate_models(train, val, models_dict): 
    """train and validate severals models and peak the best parameters
     
    Parameters
    ----------
    train : pandas.DataFrame
        the train set
    val : pandas.DataFrame
        the validation set
    models_dict : dict
        dict that contains individuals parameters for each model to test
    
    Returns
    -------
    best_model : dict
        dict that contains the best parameters for the given model

    """  
    x_train, y_train = get_x_y_from_df(train)
    x_val, y_val = get_x_y_from_df(val)

    results = list()
    for key in tqdm(models_dict): 
        clf = RandomForestClassifier(
            n_estimators=models_dict[key]["n_estimators"], 
            criterion=models_dict[key]["criterion"],
            max_depth=models_dict[key]["max_depth"],
            min_samples_split=models_dict[key]["min_samples_split"],
            max_features=models_dict[key]["max_features"],
            random_state=0, 
            n_jobs=-1)
            
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        _, correct, _ = vis.accuracy(y_val, pred)
        results.append(correct)      

    result = results.index(max(results))
    best_model = models_dict[result]

    return best_model

def train_model(train, val, model_parameters): 
    """train and validate a model with the best parameters
     
    Parameters
    ----------
    train : pandas.DataFrame
        the train set
    val : pandas.DataFrame
        the validation set
    models_dict : dict
        dict that contains the best parameters for the model
    
    Returns
    -------
    clf : estimat or estimator instance
        the estimator trained

    """ 
    x_train, y_train = get_x_y_from_df(train)
    x_val, y_val = get_x_y_from_df(val)

    clf = RandomForestClassifier(
            n_estimators=model_parameters["n_estimators"], 
            criterion=model_parameters["criterion"],
            max_depth=model_parameters["max_depth"],
            min_samples_split=model_parameters["min_samples_split"],
            max_features=model_parameters["max_features"],
            random_state=0, 
            n_jobs=-1)
            
    clf.fit(pd.concat([x_train, x_val], axis=0), pd.concat([y_train, y_val], axis=0))
        
    return clf

def write_model(model, file_name, path="./"):
    """save the model in pickle format
     
    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        the model to save
    file_name : string
        the name of the file where we save the model
    path : string
        the path where we save the model
    """  
    
    pickle.dump(model, open(path + file_name + ".pkl", 'wb'))
        
def read_model(file_name, path="./"):
    """save the model in pickle format
     
    Parameters
    ----------
    file_name : string
        the name of the file where we save the model
    path : string
        the path where we save the model
     
    
    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        the trained model
    """
    
    model = pickle.load(open(path + file_name + ".pkl", 'rb'))
    return model