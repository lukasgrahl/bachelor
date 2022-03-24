import pandas as pd
import datetime as dt

import os
import pickle

from matplotlib import pyplot as plt

from src.src import ModelValidation
from utils.cast_data import cast_data
from utils.utils import is_dir_existant


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


def save_png(file,
             file_name: str,
             file_path: str,
             **kwargs):
    try:
        file.savefig(os.path.join(file_path, file_name), **kwargs)
    except Exception as e:
        print(e)
        raise TypeError(f"File type {file_name} is not plt.Figure, file type unknonw")


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
    elif ".png" in file_name:
        save_png(data, file_name=file_name, file_path=file_path, **kwargs)
    else:
        raise KeyError(f"File tye unkonw {file_name.split('.')[-1]}")
        pass


def save_model(model,
               df_train: pd.DataFrame,
               data_dict: dict,
               model_dir: str,
               plt_figures: list,
               fig_titles: list,
               **kwargs):
    model_name = data_dict["model_metrics"]["model_name"]
    timestamp = str(dt.datetime.utcnow())[:10]
    model_name_file = f'{timestamp}_{model_name}'

    print(f'Are you sure you want to save model as: f"{model_name_file}"? (y/n)')
    conf = input()
    if conf == "y":

        file_path = os.path.join(model_dir, f"{model_name}")
        is_existant = is_dir_existant(file_path)
        if is_existant:
            conf2 = input("DIR EXISTS: DO YOU WANT TO REPLACE IT? (y/n)")
            if conf2 != "y":
                raise KeyError("No output directory specified")
        else:
            os.makedirs(file_path)

        # Save files
        assert len(plt_figures) == len(fig_titles), 'Please align figure names and figures'
        for i in range(0, len(plt_figures)):
            save_file(plt_figures[i], file_name=f"{model_name_file}_{fig_titles[i]}.png",
                      file_path=file_path, **kwargs)

        save_file(data=df_train, file_name=f"{model_name_file}_df.csv", file_path=file_path,
                  **kwargs)
        save_file(data=model, file_name=f"{model_name_file}_model.pkl", file_path=file_path,
                  **kwargs)
        save_file(data=data_dict, file_name=f"{model_name_file}_data_dict.pkl", file_path=file_path,
                  **kwargs)
        pass
