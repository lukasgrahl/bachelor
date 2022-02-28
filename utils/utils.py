import os
import datetime as dt

import numpy as np
import pandas as pd
import pickle

from scipy.stats import normaltest
from statsmodels.tsa.stattools import adfuller

from matplotlib import pyplot as plt

from utils.cast_data import cast_data, apply_datetime_format


def load_csv(file_path: str,
             **kwargs):
    return pd.read_csv(file_path, **kwargs)


def load_excel(file_path: str,
               **kwargs):
    return pd.read_excel(file_path, **kwargs)


def load_feather(file_path: str,
                 **kwargs):
    return pd.read_feather(file_path, **kwargs)


def load_pkl(file_name,
             file_path: str = None):
    if file_path is None:
        file_path = os.getcwd()

    with open(os.path.join(file_path, file_name), 'rb') as data:
        return pickle.load(data)
    pass


def save_pkl(file,
             file_name: str,
             file_path: str = None):
    if file_path is None:
        file_path = os.getcwd()

    with open(os.path.join(file_path, file_name), 'wb') as data:
        pickle.dump(file, data)
        pass


@cast_data
def load_data(file_name: str,
              file_path: str,
              **kwargs):
    file_end = file_name.split(".")[-1]
    full_path = os.path.join(file_path, file_name)

    if file_end == "csv":
        data = load_csv(full_path, **kwargs)
    elif file_end == "xlsx":
        data = load_excel(full_path, **kwargs)
    elif file_end == "feather":
        data = load_feather(full_path, **kwargs)
    elif file_end == "pkl":
        data = load_pkl(file_name, file_path)
    else:
        raise TypeError(f"File type unknown {file_end}")

    return data


def apply_date_to_week(x):
    # return apply_datetime_format(x).isocalendar()[0:2]
    x = apply_datetime_format(x).isocalendar()[0:2]
    return str(x[0]) + str(x[1])


def save_file(data,
              file_name: str,
              file_path: str,
              **kwargs):
    if ".csv" in file_name:
        data.to_csv(os.path.join(file_path, file_name), **kwargs)
        pass
    elif ".feather" in file_name:
        data.to_feather(os.path.join(file_path, file_name), **kwargs)
        pass
    elif ".xlsx" in file_name:
        data.to_excel(os.path.join(file_path, file_name), **kwargs)
        pass
    elif ".pkl" in file_name:
        save_pkl(data, file_name=file_name, file_path=file_path)
    else:
        raise KeyError(f"File tye unkonw {file_name.split('.')[-1]}")
        pass


def cut_to_weekly_data(df: pd.DataFrame,
                       relevant_cols: list = ["all"]):
    if "week" not in df.columns:
        df["week"] = df["date"].apply(lambda x: apply_date_to_week(x))

    if relevant_cols != ["all"]:
        df = df[relevant_cols]

    return df.dropna(axis=0).drop_duplicates("week")


def apply_textmonth_to_nummonth(x):
    month_dict = {
        "Dec": 12,
        "Nov": 11,
        "Oct": 10,
        "Sep": 9,
        "Aug": 8,
        "Jul": 7,
        "Jun": 6,
        "May": 5,
        "Apr": 4,
        "Mar": 3,
        "Feb": 2,
        "Jan": 1
    }

    x = x.split(" ")
    x[0] = int(month_dict[x[0]])
    x[1] = int(x[1][:-1])
    x[2] = int(x[2])

    return dt.datetime(month=x[0], day=x[1], year=x[2])
