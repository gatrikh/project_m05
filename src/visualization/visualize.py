import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def dsp(data, rows=10, columns=None):
    
    pd.options.display.max_rows = rows
    pd.options.display.max_columns = columns

    display(data)

    pd.options.display.max_rows = 15
    pd.options.display.max_columns = 20


def memory(df): 
    return round(df.memory_usage(index=True).sum() / 1000000, 2)


def accuracy(labels, pred):

    correct = int(np.sum(pred==labels))
    total = labels.shape[0]
    incorrect = total - correct
    acc = correct/total

    return acc, correct, incorrect

def get_confusion_matrix(classifier, x_test, y_test, save_fig=False, savefile_name="", path="../reports/figures/"):
    
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    disp2 = plot_confusion_matrix(classifier, x_test, y_test, ax=ax2,normalize = "true")
    disp2.ax_.set_title("Confusion matrix, without normalization")
    
    fig1, ax = plt.subplots(figsize=(15, 15))
    disp1 = plot_confusion_matrix(classifier, x_test, y_test, ax=ax,normalize = None)
    disp1.ax_.set_title("Confusion matrix, without normalization")
    disp1.ax_.autoscale
    
    if save_fig: 
        if savefile_name:
            savefile_name_normalized = path + "normalized_" + savefile_name
            savefile_name_unnormalized = path + "unnormalized_" + savefile_name
            fig1.savefig(savefile_name_unnormalized)
            fig2.savefig(savefile_name_normalized)
            
    return fig1, ax, fig2, ax2
    