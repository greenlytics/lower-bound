# -*- coding: utf-8 -*-
"""
This program estimates how well a function (wind mill output power)
can be learnt from a set of training data (wind speed weather forecast).

The challenge is to write procedures prepareData() and predict()
producing an RMSE smaller than the lower bound.

@author: Martin Nilsson, RISE (mn@drnil.com)
Updated: 2019-06-20
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd

def test(days=365, windmill=1, predictionDelay = 0, predictionDuration = 24):
    """ Top level function.
    """
    # Load data
    (t, x, y) = loadData(windmill=windmill)
    # Show bounds
    bounds(y)
    sumerr = np.zeros(days)
    for day in range(days):
        # Prepare and split output data into training and validation sets
        m = y.size - int(predictionDelay + predictionDuration)
        ttrain = t[:m]
        ytrain = y[:m]
        tvalid = t[m+predictionDelay:]
        yvalid = y[m+predictionDelay:]
        # Split input data into training and validation sets
        xtrain, xvalid = prepareData(t, x, ytrain, tvalid.size)
        # Compute validation error
        sumerr[day] += validationError((ttrain, xtrain, ytrain),
                                       (tvalid, xvalid, yvalid),
                                       show=(day==days-1))
        # Step back one day
        t = t[:-24]
        x = x[:-24,:]
        y = y[:-24]
    print("Current:                 RMSE = {:.6f}".format(rmse(sumerr)))

def loadData(windmill=1):
    """ Load data from file.
    """
    power = pd.read_csv('df_power.csv')
    wind = pd.read_csv('df_wind.csv', header=[0, 1], index_col=0)
    n = power.values.shape[0]
    y = power.values[0:n, windmill].astype(np.float64)
    x = wind.values[0:n, 4*(windmill-1):4*windmill].astype(np.float64)
    t = np.arange(x.shape[0], dtype=np.int32)
    return (t, x, y)

def bounds(y, hours = 24):
    """ Estimate bounds based on output data only.
    """
    # Use all available outputs
    err = y - np.mean(y)
    # Bound based on mean of all data
    print("Baseline (upper bound):  RMSE = {:.6f}".format(rmse(err)))
    # Compute bound based on Wold's theorem
    z = y[:-hours]
    # Stack up history for each t
    for k in range(1,hours): z = np.vstack((z,y[k:k-hours]))
    # Compute AR(n) coefficients
    w = scipy.linalg.lstsq(z.T, y[hours:])[0]
    # Compute the innovation
    err = z.T.dot(w) - y[hours:]
    print("Approximate lower bound: RMSE = {:.6f}".format(rmse(err)))

def validationError(trainingSet, validationSet, show=True):
    """ Compute validation error.
    """
    (tvalid, xvalid, yvalid) = validationSet
    pred = np.zeros(yvalid.size)
    for k in range(yvalid.size):
        pred[k] = predict(k, tvalid, xvalid, *trainingSet)
    # Plot predictions in diagram
    if show:
        plt.plot(tvalid, yvalid, tvalid, pred)
    return rmse(pred - yvalid)

def rmse(x):
    return np.sqrt(x.dot(x)/x.size)

# ---------- Prediction (dummies)

def prepareData(t, x, ytrain, validsize):
    """ Prepare data (not allowed to use yvalid).
    This procedure implements feature engineering.
    """
    # Add some non-linear features, e.g.
    x = np.block([x, x**2, x**3, x**4])
    x = x.T
    # Split the input into training and validation sets
    n = ytrain.size
    xtrain = x[:,:n]
    xvalid = x[:,-validsize:]
    return xtrain, xvalid

def predict(k, tvalid, xvalid, ttrain, xtrain, ytrain):
    """ Predict the output given input x0 (not allowed to use yvalid).
    This procedure implements the actual prediction.
    """
    x0 = xvalid[:,k]
    return np.mean(ytrain)

