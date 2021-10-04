import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def dsp(data, rows=10, columns=None):
    """displays the data
    
    this function increases the limit of rows and columns that can be displayed. 
    It is useful in a jupyter notebook to display all data at once.
    once the data is displayed, the limit of rows and columns is reduced again.
        
    Parameters
    ----------
    data : pandas.DataFrame
        the data we want to display
    rows : int
        the limit of rows we want to display
    columns : int
        the limit of columns we want to display
    """
    
    pd.options.display.max_rows = rows
    pd.options.display.max_columns = columns

    display(data)

    pd.options.display.max_rows = 15
    pd.options.display.max_columns = 20


def memory(df): 
    """give the memory usage
    
    this function gives the memory used of a dataframe in Mb. 
        
    Parameters
    ----------
    df : pandas.DataFrame
        the data we want to know the memory used
    
    Returns
    ----------
    float
        the memory used in Mb
    """
    return round(df.memory_usage(index=True).sum() / 1000000, 2)


def accuracy(labels, pred):
    """compute the accuracy for our model
    
    this function computes the accuracy given the labels and the predictions.
    it simply sums the number of labels that corresponds to the prediction
    and divids it by the length of labels
        
    Parameters
    ----------
    labels : pandas.core.series.Series
        the correct labels of our data
    pred : numpy.ndarray
        the labels that we predicted
    Returns
    ----------
    acc : float
        the accuracy of the predictions
    correct : int
        the number of correct predictions
    incorrect : int
        the number of incorrect predictions
    """
    correct = int(np.sum(pred==labels))
    total = labels.shape[0]
    incorrect = total - correct
    acc = correct/total

    return acc, correct, incorrect

def get_confusion_matrix(classifier, x_test, y_test, save_fig=False, savefile_name="", path="./"):
    """gives the objects allowing to display the normalized and unnormalized confusion matrix 
    
    this function uses the plot_confusion_matrix from sklearn.metrics
        
    Parameters
    ----------
    classifier : estimat or estimator instance
        the classifier from wich we want the confusion matrix
    x_test : pandas.core.frame.DataFrame
        the test set 
    y_test : pandas.core.series.Series
        the test labels
    save_fig : bool
        save the created figure if save_fig is true
    savefile_name : string
        the path where we want to save the figure
    
    Returns
    ----------
    fig1 : matplotlib.figure.Figure
        the normalized confusion matrix
    ax : matplotlib.axes._subplots.AxesSubplot
        the axe for the normalized matrix
    fig2 : matplotlib.figure.Figure
        the unnormalized confusion matrix
    ax2 : matplotlib.axes._subplots.AxesSubplot
        the axe for the unnormalized matrix
    """
    
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
            print(">>> normalized_confusion_matrix.png & unnormalized_confusion_matrix.png generated!")

            
    return fig1, ax, fig2, ax2
    