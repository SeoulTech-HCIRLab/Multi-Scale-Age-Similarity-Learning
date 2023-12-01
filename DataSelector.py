import pandas as pd


def data_select(dataname, phase):
    if dataname == "utk":
        database = "./datalists/UTKFace/"
        df = pd.read_csv(database + phase + ".csv")
        return df
    elif dataname == "cacd":
        database = "./datalists/CACD/"
        df = pd.read_csv(database + phase + ".csv")
        return df
