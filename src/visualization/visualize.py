import pandas as pd
import numpy as np
from IPython.display import display

def dsp(data, rows=10, columns=None):
    
    pd.options.display.max_rows = rows
    pd.options.display.max_columns = columns

    display(data)

    pd.options.display.max_rows = 15
    pd.options.display.max_columns = 20


def memory(df): 
    return round(df.memory_usage(index=True).sum() / 1000000, 2)


def accuracy(labels, pred, show=False):

    correct = np.sum(pred==labels)
    total = labels.shape[0]
    incorrect = total - correct
    acc = correct/total

    if show: 
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")

    return acc, correct, incorrect
