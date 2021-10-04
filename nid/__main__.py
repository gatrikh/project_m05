import numpy as np
import pandas as pd
import sys
import os
from importlib import reload
import matplotlib.pyplot as plt
import time
import subprocess
import argparse

from nid import data
from nid import preprocess
from nid import model
from nid import visualize as vis

REQUIRED_PYTHON = "3.8"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program options')
    parser.add_argument('-path', dest='path', type=str, required=True, help='Path of KDD99 Cup dataset in .gz format')
    parser.add_argument('-test', dest='test', action='store_true', help='Running the package tests')
    parser.add_argument('-fig', dest='fig', action='store_true', help='Saving figures of confusion matrices in working directory')
    parser.add_argument('-train', dest='train', action='store_true', help='Retrain completely the model and save it in the working directory')

    args = parser.parse_args()

    start = time.time()
    print("Main script of the Network Intrusions Package (NID)")

    module_path = os.path.dirname(data.__file__)
    if args.test:

        print("Testing environment...")
        python_version = str(sys.version_info.major) + '.' + str(sys.version_info.minor) # + '.' + str(sys.version_info.micro)

        if REQUIRED_PYTHON != python_version:
            raise TypeError(
                "This project requires Python {}. Found: Python {}".format(
                    REQUIRED_PYTHON, python_version))
        else:
            print(">>> Python {} found. The environment is adequate!".format(python_version))
        
        print("Loading data...")
        df = pd.read_csv(args.path, compression='gzip', delimiter=",", low_memory=False)
        print("Creating testing samples...")
        df = df.sample(frac=0.005, random_state=30)
        df.to_csv("kddcup_test.data.gz", index=False, compression="gzip")
        print("Running tests...")
        subprocess.run(["coverage", "run", "-m", "pytest", "nid"])
        print("Removing testing samples...")
        os.remove("kddcup_test.data.gz") 

    else:

        print("Loading data...")
        df = data.load_data(args.path)
        print("Preprocessing step...")
        df_encoded = preprocess.features_encoder(df)
        df_normalized = preprocess.normalize(df_encoded)
        train_df, val_df, test_df = preprocess.split_data(df_normalized)

        print("Training step...")
        retrain = args.train

        if retrain: 

            # Best parameters: {'n_estimators': 150, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 'max_features': None}
            
            # Paramters tested:
            parameters = {
                "n_estimators": [100, 150, 200], # Number of trees in the forest.
                "criterion": ["gini", "entropy"], # The function to measure the quality of a split.
                "max_depth": [None], # The maximum depth of the tree.
                "min_samples_split": [2], # The minimum number of samples required to split an internal node.
                "max_features": ["auto", None] # The number of features to consider when looking for the best split. 
            }
        
            models_dict = model.generate_models_parameters(parameters)
            best_model = model.train_validate_models(train_df, val_df, models_dict)
            print("Best model: ", best_model)
            clf = model.train_model(train_df, val_df, best_model)
            model.write_model(clf, "model_rfc")
        else: 
            clf = model.read_model("model_rfc", module_path + "/models/")

        print("Evaluation step...")
        x_test, y_test = model.get_x_y_from_df(test_df)
        pred = clf.predict(x_test)
        acc, correct, incorrect = vis.accuracy(y_test, pred)
        fig1, ax, fig2, ax2 = vis.get_confusion_matrix(clf, x_test, y_test, args.fig, "confusion_matrix.png")

        print("============================================ Results ============================================")
        print("Accuracy:", acc)
        print("Correct:", correct)
        print("Incorrect:", incorrect)
        print("=================================================================================================")
    
    end = time.time()
    print("Running time taken: ", round(end - start, 2), " seconds")