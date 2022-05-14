# Import Statements Galor!
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import tensorflow as tf
import numpy as np
import copy
import traceback
import time
import sys



# Setting up model parameters for all tests
params = {
    "LAYERS":[[(8, LSTM)], [(16, LSTM), (16, Dense)], [(48, LSTM), (48, Dense)], 
        [(48, LSTM), (48, Dense), (48, Dense)], [(256, LSTM), (256, Dense), (128, Dense)],
        [(256, LSTM), (256, Dense), (128, Dense), (64, Dense)], [(256, LSTM), (256, LSTM), (256, Dense), (64, Dense)]],
    "EPOCHS":[20, 200, 400, 800, 1600],
    "LOSS": "huber_loss",
    "OPTIMIZER": "adam",
    "DROPOUT": .2,
    "BATCH_SIZE": 128,
    "PATIENCE": 100,
    "N_STEPS": 3,
    
}



# Data loading fuctions
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

def strip_percent(df, column):
    intermediate = df[column].values.astype(str)
    intermediate = np.char.strip(intermediate, "%")
    df[column] = intermediate.astype(float)
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

def handle_data(df_list, scale_factors, test_var, params):
    Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest = [], [], [], [], [], []
    df_list_c = copy.deepcopy(df_list)
    for df in df_list_c:
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

        result = {}
        result = split_data(X, y)

        Xtrain.append(result["X_train"])
        Xvalid.append(result["X_valid"])
        Xtest.append(result["X_test"])
        ytrain.append(result["y_train"])
        yvalid.append(result["y_valid"])
        ytest.append(result["y_test"])

    Xtrain = np.array(Xtrain, dtype=object)
    Xvalid = np.array(Xvalid, dtype=object)
    Xtest = np.array(Xtest, dtype=object)
    ytrain = np.array(ytrain, dtype=object)
    yvalid = np.array(yvalid, dtype=object)
    ytest = np.array(ytest, dtype=object)

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
    
    return ten_train, ten_valid, ten_test

## Model creation functions
def layer_name_converter(layer):
    # print(layer, flush=True)
    string = ""
    
    if str(layer[1]) == "<class 'keras.layers.recurrent_v2.LSTM'>":
        string += "LSTM"
    elif str(layer[1]) == "<class 'keras.layers.recurrent.SimpleRNN'>":
        string += "SRNN"
    elif str(layer[1]) == "<class 'keras.layers.recurrent_v2.GRU'>":
        string += "GRU"
    elif str(layer[1]) == "<class 'keras.layers.core.dense.Dense'>":
        string += "Den"
    elif str(layer[1]) == "<class 'tensorflow_addons.layers.esn.ESN'>":
        string += "ESN"
    else:
        string += str(layer[1])

    return string

def layers_string(layers):
    string = "["
    for layer in layers:
        string += str(layer[0]) + layer_name_converter(layer)
    string += "]"

    return string

def get_model_name(nn_params, i, j):
    return f"""Layers{layers_string(nn_params["LAYERS"][i])}-epoch{nn_params["EPOCHS"][j]}""" 

def create_model(params, i):
    model = Sequential()
    # print(bi_string)
    for layer in range(len(params["LAYERS"][i])):
        if layer == 0:
            model_first_layer(model, params["LAYERS"][i], layer, params["N_STEPS"], predicting_columns)
        elif layer == len(params["LAYERS"][i]) - 1:
            model_last_layer(model, params["LAYERS"][i], layer)
        else:
            model_hidden_layers(model, params["LAYERS"][i], layer)
    
        model.add(Dropout(params["DROPOUT"]))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=params["LOSS"], metrics=["mean_absolute_error"], optimizer=params["OPTIMIZER"])

    return model

def model_first_layer(model, layers, ind, n_steps, features):
    layer_name = layer_name_converter(layers[ind])
    if len(layers) == 1:
        next_layer_name = "Den"
    else:
        next_layer_name = layer_name_converter(layers[ind + 1])
    

    if layer_name == "Den":
        model.add(layers[ind][1](layers[ind][0], activation="elu", input_shape=(None, len(features))))
    else:
        if next_layer_name == "Den":
            model.add(layers[ind][1](layers[ind][0], return_sequences=False, 
                input_shape=(n_steps, len(features))))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=True, 
                input_shape=(n_steps, len(features))))

    return model

def model_hidden_layers(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])
    next_layer_name = layer_name_converter(layers[ind + 1])

    if layer_name == "Den":
        model.add(layers[ind][1](layers[ind][0], activation="elu"))
    else:
        if next_layer_name == "Den":
            model.add(layers[ind][1](layers[ind][0], return_sequences=False))
        else:
            model.add(layers[ind][1](layers[ind][0], return_sequences=True))

    return model

def model_last_layer(model, layers, ind):
    layer_name = layer_name_converter(layers[ind])

    if layer_name == "Den":
        model.add(layers[ind][1](layers[ind][0], activation="elu"))
    else:
        model.add(layers[ind][1](layers[ind][0], return_sequences=False))
    
    return model

# Code for model creation and testing
def create_test(test_var_list, df_list, scale_factors):
    for test_var in test_var_list:
        # print(all_dfs)
        for i, layer in enumerate(params["LAYERS"]):
            for j, layer in enumerate(params["EPOCHS"]):
                print(f"""Now testing test var {test_var} with layers {layers_string(params["LAYERS"][i])} with epochs {params["EPOCHS"][j]}""")
                ten_train, ten_valid, ten_test = handle_data(df_list, scale_factors, test_var, params)

                model_name = (f"{position}-{test_var}-{get_model_name(params, i, j)}")    
                model = create_model(params, i)
                # print(model.summary())
                early_stop = EarlyStopping(patience=params["PATIENCE"], verbose=0)

                history = model.fit(ten_train,
                    batch_size=params["BATCH_SIZE"],
                    epochs=params["EPOCHS"][j],
                    verbose=2,
                    validation_data=ten_valid,
                    callbacks = [early_stop]   
                )

                epochs_used = len(history.history["loss"])

                huber_lost, nn_mae = model.evaluate(ten_test)
                mae = scale_factors[test_var].inverse_transform([[nn_mae]])[0][0]
                please.append([get_model_name(params, i, j), position, test_var, mae, epochs_used])

    result_df = pd.DataFrame(please, columns=["Model Name", "Position", "Test Variable", "MAE", "Epochs Used"])
    print(f"""Static variables were batch size of {params["BATCH_SIZE"]}, n_steps of {params["N_STEPS"]}, """
        f"""dropout of {params["DROPOUT"]}, loss function {params["LOSS"]}, """
        f"""optimizer of {params["OPTIMIZER"]}, and patience of {params["PATIENCE"]}""")
    print(result_df)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.expand_frame_repr', False)

# Code for processing data then training/test for QBs
files = os.scandir("../Data/QBs")
df_list = []

unwanted = ["Pos", "No.", "Tm", "QBrec", "QBR", "1D"]

for file in files:
    # print(f"\n{file}")
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
scale_factors = get_scaling_factor(all_dfs)

test_var_list = ["Yds", "Att", "TD"]
position = "QB"
predicting_columns = all_dfs.columns
please = []


s_time = time.time()
create_test(test_var_list, df_list, scale_factors)
print(f"All testing for {position} took {(time.time() - s_time) / 60}")


# Code for processing data then training/test for WRs
files = os.scandir("../Data/WRs")
df_list = []

unwanted = ["Pos", "No.", "Tm", "Ctch%", "Ret", "Rt", "Y/Rt", "Tgt", "Y/Tgt", "Yds.1", "TD.1", "1D.1", "Lng.1", 
    "Y/A", "Y/G.1", "Touch", "Y/Tch", "YScm", "RRTD", "Fmb", "1D", "APYd"]

for file in files:
    # print(f"\n{file}")
    df = pd.read_csv(file)
    df = score_awards(df)
    df = int_years(df)
    df = drop_unwanted(df, unwanted)
    
    df = df.fillna(0.0)
    # print(df)
    df_list.append(df)

all_dfs = df.append(df_list)
scale_factors = get_scaling_factor(all_dfs)

test_var_list = ["Yds", "Rec", "TD"]
position = "WR"
predicting_columns = all_dfs.columns
please = []

s_time = time.time()
create_test(test_var_list, df_list, scale_factors)
print(f"All testing for {position} took {(time.time() - s_time) / 60}")



# Code for processing data then training/test for RBs
files = os.scandir("../Data/RBs")
df_list = []

unwanted = ["Pos", "No.", "Tm", "QBrec", "QBR", "1D", "Tgt", "Rec", "Yds.1", "Y/R", "TD.1", "Lng.1", "R/G", 
    "Y/G.1", "Ctch%", "1D.1", "Y/Tgt"]

for file in files:
    # print(f"\n{file}")
    df = pd.read_csv(file)
    df = score_awards(df)

    df = int_years(df)
    df = drop_unwanted(df, unwanted)
    df = df.fillna(0.0)
    # print(len(df.columns))
    # print(df)
    df_list.append(df)

all_dfs = df.append(df_list)
scale_factors = get_scaling_factor(all_dfs)

test_var_list = ["Yds", "Rush", "TD"]
position = "RB"
predicting_columns = all_dfs.columns
please = []

s_time = time.time()
create_test(test_var_list, df_list, scale_factors)
print(f"All testing for {position} took {(time.time() - s_time) / 60}")
