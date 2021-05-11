
import numpy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import stats
from filters import corr_filter, mse_filter


def normalize_data(x, y):
    min_max_scaler = preprocessing.MinMaxScaler()

    x_values = x.values
    x_scaled = min_max_scaler.fit_transform(x_values)

    y_values = y.values
    y_values = np.reshape(y_values, (len(y_values), 1))
    y_scaled = min_max_scaler.fit_transform(y_values)

    x = pd.DataFrame(x_scaled, columns=x.columns)
    y = pd.DataFrame(y_scaled, columns=y.columns)

    return x, y


def remove_outliers(x, y):
    z_scores = stats.zscore(y)

    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < 2)

    x, y = x[filtered_entries], y[filtered_entries]
    return x, y


def label_encoder(x, column):
    le = preprocessing.LabelEncoder()
    x.iloc[:,column] = le.fit_transform(x.iloc[:,column])
    return x


def load_data(file, preprocess=False, filter="no_filter"):
    if file == "OnlineNewsPopularity.csv":
        df = pd.read_csv(file, encoding="ISO-8859-1", sep=",")
        df = df.sample(frac=1).reset_index(drop=True)
        if filter == "mse":
            df = mse_filter(df)
        elif filter == "corr":
            df = corr_filter(df)
        x = df.iloc[:, 1:-1] # select independent parameters
        y = df.iloc[:, [-1]] # select dependent (target) parameter
    else:
        df = pd.read_csv(file, encoding="ISO-8859-1", sep=";")
        df = df.sample(frac=1).reset_index(drop=True)
        x = df.iloc[:, 0:-1] # select independent parameters
        y = df.iloc[:, [-1]] # select dependent (target) parameter
        # preprocess dates
        for i in range(len(x)):
            x.iloc[i, 0] = x.iloc[i, 0][-7:-5] # reformat date column
        ### encode string parameters
        # seasons
        x = label_encoder(x, 10)
        # Holiday and Functioning day
        x = label_encoder(x, 11)
        x = label_encoder(x, 12)

    if preprocess:
        x, y = normalize_data(x, y)
        x, y = remove_outliers(x, y)
    return x, y


def split_hold_ou(x, y, test_size):
    return (train_test_split(x, y, test_size=test_size, random_state=0))


def split_cross_val(data, k=10):
    kf = KFold(n_splits=k)
    indices = []
    for train, test in kf.split(data):
        indices.append((train, test))
        
    return indices
