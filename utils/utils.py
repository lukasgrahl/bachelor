import datetime as dt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_white

from matplotlib import pyplot as plt

from settings import random_state
from utils.cast_data import apply_date_to_week


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


def translate_neg_dist(arr):
    if min(arr) < 0:
        return True, arr + (abs(arr.min()) + 1)
    else:
        return False, arr
    pass


def arr_log_return(arr: pd.Series):
    is_trans, arr = translate_neg_dist(arr)
    return is_trans, np.log1p(arr / arr.shift(1))


def arr_log_transform(arr: pd.Series):
    is_trans, arr = translate_neg_dist(arr)
    return is_trans, np.log1p(arr)


def shift_var_relative_to_df(df_in,
                             shift_var: str,
                             new_var_name: str = [None],
                             no_lags: int = [-1]):
    if max(no_lags) > 0:
        print("Applying shifts in future")
    assert len(no_lags) == len(new_var_name), "Please name new cols"

    df = df_in.copy()
    for i, lag in enumerate(no_lags):
        if new_var_name != [None]:
            df[new_var_name[i]] = df[shift_var].shift(lag)
        else:
            df[shift_var] = df[shift_var].shift(lag)

    return df.dropna(axis=0)


def df_heteroskedasticity_white(y,
                                X,
                                significance_level: float = 0.05,
                                print_results: bool = True):
    # H0: Homoscedasticity is present (residuals are equally scattered)

    list_1 = []
    for item in np.mean(X, axis=0):
        if item == 1:
            list_1.append(True)
        else:
            pass
    assert len(list_1) == 1, "Het White test requires a constant in X data"

    wtest = het_white(y, X)
    labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
    test_result = dict(zip(labels, wtest))

    bool_result = test_result["Test Statistic p-value"] < significance_level

    plt.scatter(np.sum(X, axis=1), y)

    if print_results:
        print("Test for Heteroskedasticity")
        print(f"Test p-value: {test_result['Test Statistic p-value']}")
        print(f"Heteroskedasticity is present: {bool_result}")

    return bool_result


def tts_data(df_in,
             y: str,
             x: list,
             add_const: bool = True,
             random_split: bool = True,
             test_size: float = 0.3):
    # Asumption: times series ordered past - future

    df = df_in.copy()
    y = df[y].copy()
    X = df[x].copy()
    if add_const:
        X["intercept"] = list([1] * len(X))

    if random_split:
        tts = train_test_split(X, y, test_size=.3, random_state=random_state)

    else:
        test_size = round(len(df) * (1 - test_size))
        y_train = y.iloc[:test_size]
        X_train = X.iloc[:test_size]

        y_test = y.iloc[test_size:]
        X_test = X.iloc[test_size:]

        tts = [X_train, X_test, y_train, y_test]

    for i, _ in enumerate(tts):
        tts[i] = tts[i].reset_index(drop=True)

    return tts
