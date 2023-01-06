from turtle import shape
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from preprocess.gaussianize import Gaussianize
from copy import deepcopy
import random

import matplotlib.pyplot as plt

def split_series(series, n_past=24, n_future=0):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        if n_future > 0:
            future_end = past_end + n_future
            if future_end > len(series):
                break
        elif past_end > len(series):
            break
        if n_future > 0:
            past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        else:
            past = series[window_start:past_end, :]
        X.append(past)
        if n_future > 0:
            y.append(future)
    if n_future > 0:
        return np.array(X), np.array(y)
    else:
        return np.array(X)



def data_to_scaled_data(df_ret, lambert_flag=False, num_factors=0, scaler_type="MinMaxScaler"):
    if isinstance(df_ret, pd.DataFrame):
        log_returns = df_ret.Values
    else:
        log_returns = deepcopy(df_ret)
    if num_factors > 0:
        log_returns = log_returns[:, :num_factors]

    if scaler_type.upper() == 'StandardScaler'.upper():
        scaler1 = StandardScaler()
    else:
        scaler1 = MinMaxScaler()

    log_returns_preprocessed = scaler1.fit_transform(log_returns)
    return log_returns, log_returns_preprocessed, scaler1


def inverse_trans(train_data, scaler):
    num_instances, num_time_steps, num_features = train_data.shape
    train_data = np.reshape(train_data, (-1, num_features))

    train_data = scaler.inverse_transform(train_data)
    train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
    return train_data
