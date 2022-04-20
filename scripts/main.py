# Import Statements Galor!
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import traceback
import time
import sys

# pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Data loading
files = os.scandir("../Data/QBs")
df_list = []
awards_scorer = {"PB": 1, "MVP-1":4, "MVP-2":3, "MVP-3":2, "MVP-4":1, "MVP-5":1, 
    "AP1":4, "AP2":3, "OPoY-1":3, "OPoY-2":2, "OPoY-3":1, "OPoY-4":1, "OPoY-5":1,
    "CPoY-1":1, "CPoY-2":1, "CPoY-3":1, "ORoY-1":2, "ORoY-2":1, "ORoY-3":1}

for file in files:
    try:
        print(f"\n{file}")
        # print(pd.read_csv(file, delim_whitespace=True)) 
        tmp_df = pd.read_csv(file)

        if "Awards" in tmp_df.columns:
            awards = tmp_df["Awards"].fillna("0").values.astype(str)
            awards_byY_list = np.char.split(awards, sep=" ")
            awards_score = []
            for year in awards_byY_list:
                curr_score = 0
                for award in year:
                    if award in awards_scorer:
                        curr_score += awards_scorer[award]
                awards_score.append(curr_score)

            tmp_df["Awards"] = awards_score
        else: 
            tmp_df["Awards"] = [0 for x in range(len(tmp_df["Age"]))]
        
        QBrec = tmp_df["QBrec"].fillna("0-0-0").values.astype(str)
        QBrec = np.char.split(QBrec, "-")
        QBrec = np.array([np.array(xi) for xi in QBrec])
        tmp_df["QBW"], tmp_df["QBL"] = QBrec[:, 0], QBrec[:, 1]

        intermediate = tmp_df["Year"].values.astype(str)
        intermediate = np.char.strip(intermediate, "*+")
        tmp_df["Year"] = intermediate.astype(int)

        tmp_df = tmp_df.drop(columns=["Pos", "No.", "Tm", "QBrec"])
        if "QBR" in tmp_df.columns:
            tmp_df = tmp_df.drop(columns=["QBR"])
        if "1D" in tmp_df.columns:
            tmp_df = tmp_df.drop(columns=["1D"])
        tmp_df = tmp_df.fillna(0.0)
        df_list.append(tmp_df)
        # print(len(tmp_df.columns))

    except Exception:
        exit_info = sys.exc_info()
        print(exit_info)
        print(str(exit_info[1]) + "\n")
        print(traceback.print_tb(tb=exit_info[2]))
        print(" \n\n\n\n\n FIX ME \n\n\n\n", flush=True)
        # sys.exit(-1)
        time.sleep(1)


