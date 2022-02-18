import os

import pandas as pd

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


@cast_data
def load_data(file_name: str,
              file_path: str,
              **kwargs):
    file_end = file_name.split(".")[-1]
    full_path = os.path.join(file_path, file_name)

    if file_end == "csv":
        df = load_csv(full_path, **kwargs)
    elif file_end == "xlsx":
        df = load_excel(full_path, **kwargs)
    elif file_end == "feather":
        df = load_feather(full_path, **kwargs)
    else:
        raise TypeError("File type unknown")

    return df


def save_file(df: pd.DataFrame,
              file_name: str,
              file_path: str,
              *args):

    if ".csv" in file_name:
        df.to_csv(os.path.join(file_path, file_name))
        pass
    elif ".feather" in file_name:
        df.to_feather(os.path.join(file_path, file_name))
        pass
    elif ".xlsx" in file_name:
        df.to_excel(os.path.join(file_path, file_name))
        pass
    else:
        raise KeyError("File tye unkonw, file not saved")
        pass


if __name__ == "__main__":
    from settings import RAW_DATA_DIR, WORK_DATA_DIR

    fears = load_data("fears.csv", file_path=RAW_DATA_DIR)
    save_to_csv(df=fears, file_name="fears", file_path=WORK_DATA_DIR)
