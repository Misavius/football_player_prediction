# Import Statements Galor!
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import traceback
import time
import sys

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

# Setting up model parameters
params = {
    "N_STEPS": 3,
    "LAYERS": [(256, LSTM), (128, Dense)],
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "EPOCHS": 100,
    "BATCH_SIZE": 32,
    "PATIENCE": 200
}



# Data loading
files = os.scandir("../Data/QBs")
df_list = []
awards_scorer = {"PB": 1, "MVP-1":4, "MVP-2":3, "MVP-3":2, "MVP-4":1, "MVP-5":1, 
    "AP1":4, "AP2":3, "OPoY-1":3, "OPoY-2":2, "OPoY-3":1, "OPoY-4":1, "OPoY-5":1,
    "CPoY-1":1, "CPoY-2":1, "CPoY-3":1, "ORoY-1":2, "ORoY-2":1, "ORoY-3":1}

def score_awards(df):
    if "Awards" in df.columns:
        awards = df["Awards"].fillna("0").values.astype(str)
        awards_byY_list = np.char.split(awards, sep=" ")
        awards_score = []
        for year in awards_byY_list:
            curr_score = 0
            for award in year:
                if award in awards_scorer:
                    curr_score += awards_scorer[award]
            awards_score.append(curr_score)

        df["Awards"] = awards_score
    else: 
        df["Awards"] = [0 for x in range(len(df["Age"]))]
    
    return df

def int_years(df):
    intermediate = df["Year"].values.astype(str)
    intermediate = np.char.strip(intermediate, "*+")
    df["Year"] = intermediate.astype(int)
    return df

def drop_unwanted(df, unwanted):
    for ewwwwwww in unwanted:
        if ewwwwwww in df:
            df = df.drop(columns=ewwwwwww)
    return df

def get_scaling_factor(df):
    column_scaler = {}
    for column in df.columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler
    return column_scaler

def apply_scaling(df, scalers):
    for column in df.columns:
        df[column] = scalers[column].transform(np.expand_dims(df[column].values, axis=1))
    return df

def remove_scaling(df, scalers):
    for column in df.columns:
        df[column] = scalers[column].inverse_transform(np.expand_dims(df[column].values, axis=1))
    return df

def split_data(X, y):
    data_splits = {3:[1, 1, 1], 4:[2, 1, 1], 5:[3, 1, 1], 6:[4, 1, 1], 7:[5, 1, 1], 8:[4, 2, 2], 9:[5, 2, 2],
        10:[6, 2, 2], 11:[7, 2, 2], 12:[8, 2, 2], 13:[7, 3, 3], 14:[8, 3, 3], 15:[9, 3, 3], 16:[10, 3, 3], 
        17:[11, 3, 3], 18:[10, 4, 4], 19:[11, 4, 4], 20:[12, 4, 4], 21:[13, 4, 4], 22:[14, 4, 4], 23:[13, 5, 5]}

    # print(len(X))
    train_len, valid_len, test_len = data_splits[len(X)][0], data_splits[len(X)][1], data_splits[len(X)][2]
    # print(train_len, valid_len, test_len)

    result = {}
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_len,
      shuffle=False)
    result["X_train"], result["X_valid"], result["y_train"], result["y_valid"] = train_test_split(result["X_train"], 
        result["y_train"], test_size=valid_len, shuffle=False)
    # print(result["X_train"], result["X_valid"], result["X_test"])
    # print(result["y_train"], result["y_valid"], result["y_test"])
    return result

def make_tensor_slices(Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest, batch_size):
    train = Dataset.from_tensor_slices((Xtrain, ytrain))
    valid = Dataset.from_tensor_slices((Xvalid, yvalid))
    test = Dataset.from_tensor_slices((Xtest, ytest))

    train = train.batch(batch_size)
    valid = valid.batch(batch_size)
    test = test.batch(batch_size)
    
    train = train.cache()
    valid = valid.cache()
    test = test.cache()

    train = train.prefetch(buffer_size=AUTOTUNE)
    valid = valid.prefetch(buffer_size=AUTOTUNE)
    test = test.prefetch(buffer_size=AUTOTUNE)

    return train, valid, test


test_var = "Yds"
unwanted = ["Pos", "No.", "Tm", "QBrec", "QBR", "1D"]

for file in files:
    # try:
    print(f"\n{file}")
    df = pd.read_csv(file)
    df = score_awards(df)

    QBrec = df["QBrec"].fillna("0-0-0").values.astype(str)
    QBrec = np.char.split(QBrec, "-")
    QBrec = np.array([np.array(xi) for xi in QBrec])
    df["QBW"], df["QBL"] = QBrec[:, 0], QBrec[:, 1]

    df = int_years(df)
    df = drop_unwanted(df, unwanted)
    df = df.fillna(0.0)
    # print(df)
    df_list.append(df)

all_dfs = df.append(df_list)
# print(all_dfs)
scale_factors = get_scaling_factor(all_dfs)
# print(scale_factors)

Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest = [], [], [], [], [], []

for df in df_list:
    predicting_columns = df.columns
    # print(df)
    df = apply_scaling(df, scale_factors)
    df["future"] = df[test_var].shift(-1)
    # print(df)

    sequence_data = []
    sequences = deque(maxlen=3)
    for entry, target in zip(df[predicting_columns].values, df["future"].values):
        sequences.append(entry)
        if len(sequences) == 3:
            sequence_data.append([np.array(sequences), target])

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    X = np.delete(X, len(X) - 1, 0)
    y = np.delete(y, len(y) - 1, 0)

    # print(X.shape[0], X.shape[2], X.shape[1])
    # X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # print(X)
    # print(y)
    result = {}
    result = split_data(X, y)

    Xtrain.append(result["X_train"])
    Xvalid.append(result["X_valid"])
    Xtest.append(result["X_test"])
    ytrain.append(result["y_train"])
    yvalid.append(result["y_valid"])
    ytest.append(result["y_test"])

Xtrain = np.array(Xtrain)
Xvalid = np.array(Xvalid)
Xtest = np.array(Xtest)
ytrain = np.array(ytrain)
yvalid = np.array(yvalid)
ytest = np.array(ytest)

con_Xtrain = np.concatenate(Xtrain)
con_Xvalid = np.concatenate(Xvalid)
con_Xtest = np.concatenate(Xtest)
con_ytrain = np.concatenate(ytrain)
con_yvalid = np.concatenate(yvalid)
con_ytest = np.concatenate(ytest)


print(len(con_Xtrain), len(con_Xvalid), len(con_Xtest))
print(len(con_ytrain), len(con_yvalid), len(con_ytest))

ten_train, ten_valid, ten_test = make_tensor_slices(con_Xtrain, con_Xvalid, con_Xtest, 
    con_ytrain, con_yvalid, con_ytest, params["BATCH_SIZE"])

        # print(len(df.columns))

    # except Exception:
    #     exit_info = sys.exc_info()
    #     print(exit_info)
    #     print(str(exit_info[1]) + "\n")
    #     print(traceback.print_tb(tb=exit_info[2]))
    #     print(" \n\n\n\n\n FIX ME \n\n\n\n", flush=True)
    #     # sys.exit(-1)
    #     time.sleep(1)



