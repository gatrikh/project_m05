from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
import itertools
import inspect 
from tqdm import tqdm
from src.visualization import visualize as vis

def train_model(train, val, parameters): 
    
    x_train = train.drop(["label"], axis="columns")
    y_train = train["label"]

    x_val = val.drop(["label"], axis="columns")
    y_val = val["label"]
    
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

    clf = RandomForestClassifier(
            n_estimators=models_dict[result]["n_estimators"], 
            criterion=models_dict[result]["criterion"],
            max_depth=models_dict[result]["max_depth"],
            min_samples_split=models_dict[result]["min_samples_split"],
            max_features=models_dict[result]["max_features"],
            random_state=0, 
            n_jobs=-1)
            
    clf.fit(pd.concat([x_train, x_val], axis=0), pd.concat([y_train, y_val], axis=0))
        
    return clf