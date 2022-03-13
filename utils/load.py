import pandas as pd
import datetime as dt

import os
import pickle

from src.src import ModelValidation
from utils.cast_data import cast_data


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


def save_model(df,
               model,
               data_dict: dict,
               name: str,
               file_path: str):

    timestamp = str(dt.datetime.utcnow())[:10]

    print(f'Are you sure you want to save model as: f"{timestamp}_{name}_model.pkl"? (y/n)')
    conf = input()
    if conf == "y":
        file_path = os.path.join(file_path, f"{name}")
        os.makedirs(file_path)
        save_file(data=df, file_name=f"{timestamp}_{name}_df.csv", file_path=file_path)
        save_file(data=model, file_name=f"{timestamp}_{name}_model.pkl", file_path=file_path)
        save_file(data=data_dict, file_name=f"{timestamp}_{name}_data_dict.pkl", file_path=file_path)
    pass
