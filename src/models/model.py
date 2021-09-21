import pandas as pd
import os as os
import pickle as pickle
from sklearn.ensemble import RandomForestClassifier
import itertools
from tqdm import tqdm
from src.visualization import visualize as vis

def get_x_y_from_df(df):

    x = df.drop(["label"], axis="columns")
    y = df["label"]

    return x, y

def generate_models_parameters(parameters): 
    
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

def write_model(model, file_name, path="../models/"):
    
    pickle.dump(model, open(path + file_name + ".pkl", 'wb'))
        
def read_model(file_name, path="../models/"):
    
    model = pickle.load(open(path + file_name + ".pkl", 'rb'))
    return model