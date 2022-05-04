"""
The scipt that contains all the statistical metrics for measuring the result of the fit
Each metric takes at least a pred and truth list with (n_sample, n_dimension) of y pair
"""

import numpy as np
import os
from scipy.stats import spearmanr       # Spearman's Rho
from scipy.stats import kendalltau                # Tau
from sklearn.metrics import r2_score

# Mean Squared Error metric
def MSE(pred, truth, mean=False):
    """
    :extra param: mean (default: False): if yes, output one single number instead of a list
    """
    mse_list = np.mean(np.square(pred - truth), axis=1)
    if mean:
        return np.mean(mse_list)
    else:
        return mse_list

# Mean Absolute Error
def MAE(pred, truth, mean=False):
    """
    :extra param: mean (default: False): if yes, output one single number instead of a list
    """
    mae_list = np.mean(np.abs(pred - truth), axis=1)
    if mean:
        return np.mean(mae_list)
    else:
        return mae_list

# Mean relative absolute error
def MARE(pred, truth, mean=False):
    """
    :extra param: mean (default: False): if yes, output one single number instead of a list
    """
    mare_list = np.mean(np.abs(pred - truth)/truth, axis=1)
    if mean:
        return np.mean(mare_list)
    else:
        return mare_list

# Mean relative square error
def MARE(pred, truth, mean=False):
    """
    :extra param: mean (default: False): if yes, output one single number instead of a list
    """
    mare_list = np.mean(np.square(pred - truth)/truth, axis=1)
    if mean:
        return np.mean(mare_list)
    else:
        return mare_list

# R^2
def R2(pred, truth):
    return r2_score(truth, pred)

# Spearman's Rho
def SpearmanRho(pred, truth):
    return spearmanr(pred, truth).correlation

# Kendall's Tau
def KendallTau
    tau, p_value = kendalltau(pred, truth)
    return tau


