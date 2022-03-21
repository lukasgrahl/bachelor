import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

from contextlib import contextmanager
import sys, os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm

from settings import random_state
from utils.plotting import corr_plot


# data transformation
def translate_neg_dist(arr):
    if min(arr.dropna()) <= 0:
        return True, arr + (abs(arr.min()) + 1)
    else:
        return False, arr
    pass


def arr_log_transform(arr: pd.Series):
    is_trans, arr = translate_neg_dist(arr)
    return is_trans, np.log(arr)


def df_transform(df_in: pd.DataFrame,
                 cols: list,
                 func):
    df = df_in.copy()
    dist_trans = []

    for col in cols:
        is_trans, df[col] = func(df[col])
        dist_trans.append(is_trans)

    dist_translation = dict(zip(cols, dist_trans))
    log_returns = dict(zip(cols, list([True] * len(cols))))

    df.dropna(inplace=True)

    return df, dist_translation, log_returns

def arr_log_return(arr: pd.Series):
    # Assumption, df is ordered past to future
    is_trans, arr = translate_neg_dist(arr)
    return is_trans, np.log(1 + arr.pct_change())


def arr_inv_log_returns(arr):
    return np.exp(arr)


def shift_var_relative_to_df(df_in,
                             shift_var,
                             new_var_name: list = None,
                             no_lags: list = [1]):
    df = df_in.copy()
    shift_dict = {}

    if max(no_lags) < 0:
        print("Applying shifts in future")

    if new_var_name is None:
        if type(shift_var) == str:
            for i in no_lags:
                if i > 0:
                    shift_direction = "lag"
                if i < 0:
                    shift_direction = "lead"
                df[f"{shift_var}_{shift_direction}{abs(i)}"] = df[shift_var].shift(i)
            shift_dict[shift_var] = no_lags
            return df, shift_dict
        else:
            assert len(no_lags) == len(shift_var), "shift_var and no_lags don't correspond"
            for i, var in enumerate(shift_var):
                df[var] = df[var].shift(no_lags[i])
        shift_dict = dict(zip(shift_var, no_lags))
        return df, shift_dict

    elif new_var_name is not None:
        assert len(no_lags) == len(new_var_name) == len(shift_var), "Please name new cols"

        for i, var in enumerate(shift_var):
            df[new_var_name[i]] = df[var].shift(no_lags[i])
        shift_dict = dict(zip(shift_var, no_lags))
        return df, shift_dict


def add_constant(df_in,
                 constant_value=1,
                 constant_name: str = "intercept"):
    """
    Inserts constant column into df
    :param df_in:
    :param constant_value: value of the constant
    :param constant_name: name of the constant col
    :return:
    """
    df = df_in.copy()
    df[constant_name] = list([constant_value] * len(df))
    return df


# cut & sort data
def cut_to_weekly_data(df: pd.DataFrame,
                       filter_col: str):
    return df[df[filter_col] == True].dropna()


def tts_data(df_in,
             y: str,
             x: list,
             add_const: bool = True,
             random_split: bool = True,
             test_size: float = 0.3,
             reset_index: bool = False):
    # Asumption: times series ordered past - future

    df = df_in.copy()
    y = df[y].copy()
    X = df[x].copy()

    _ = X.apply(pd.Series.nunique) == 1
    if len(_[_]) > 0:
        print(f"Constant columns exist: {list(_[_].index)}")
        add_const = False
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

    if reset_index:
        for i, _ in enumerate(tts):
            tts[i] = tts[i].reset_index(drop=True)

    return tts


# other
def get_variance_inflation_factor(df,
                                  cols,
                                  col_pred):
    vif = [variance_inflation_factor(df[cols].values, i) for i in range(df[cols].shape[1])]
    vif = pd.DataFrame(index=cols, data=vif, columns=["VIF"])
    vif = vif.join(df[cols].corrwith(df[col_pred]).rename(f"corr_{col_pred}"))
    return vif.sort_values(f"corr_{col_pred}")


# other
def is_dir_existant(path):
    return os.path.isdir(path)


def get_df_time_overview_report(df_dict: dict,
                                date_col: str = 'date'):
    _out = []
    for df in df_dict.values():
        _out.append(
            [
                df[date_col].min(),  # start date
                df[date_col].max(),  # end date
                (df[date_col].max() - df[date_col].min()).days,  # no of days
                round((df[date_col].max() - df[date_col].min()).days / 7),
                len(df) / (df[date_col].max() - df[date_col].min()).days,  # number of days (incld we or not)
                (df[date_col] - df[date_col].shift(1)).apply(lambda x: x.days).mean()
                # weekly or not weekly ~ 7, daily ~ 1 <= x < 1.5
            ]
        )
    return pd.DataFrame(_out,
                        columns=["min", "max", "days", "weeks", "days_perc", "week_or_day"],
                        index=df_dict.keys())


def orthogonalise_vars(df_in,
                       X: str,
                       y: str,
                       show_fig: bool = True):
    df = df_in.copy()
    df["intercept"] = list([1] * len(df))
    ortho = sm.OLS(endog=df[y], exog=df[[X, "intercept"]]).fit()

    if show_fig is True:
        plt.figure(figsize=(20, 4))
        plt.plot(ortho.resid, marker="o", lw=.5)
        plt.plot(df[X])

    df[y] = ortho.resid

    return df


def is_day(arr,
           day=None):
    _dict = {"Mon": 0,
             "Tue": 1,
             "Wed": 2,
             "Thu": 3,
             "Fri": 4,
             "Sat": 5,
             "Sun": 6}

    if day is None:
        return arr.apply(lambda x: 99 if x == np.nan else x.weekday())
    else:
        return arr.apply(lambda x: 99 if x == np.nan else x.weekday() == _dict[day]).rename(f"is_{day}")


def datetime_range(start, end):
    dt_range = []
    day_span = end - start
    for i in range(0, day_span.days):
        dt_range.append(start + dt.timedelta(i))
    return dt_range


def update_dict(dict_in: dict,
                update_keys: list,
                update_vals: list):
    dict_ = dict_in.copy()

    for i, key in enumerate(update_keys):
        dict_[key] = update_vals[i]

    return dict_


def get_performance_metrics(y_true,
                            y_pred):
    rmse = round(mean_squared_error(y_true, y_pred, squared=False), 10)
    mse = round(mean_squared_error(y_true, y_pred), 10)
    mae = round(mean_absolute_error(y_true, y_pred), 10)
    r2 = round(r2_score(y_true, y_pred), 10)
    return rmse, mse, mae, r2


@contextmanager
def suppress_cmd_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# old func
def lag_correl(df,
               cols,
               col_predicted: str,
               max_lag: int = 15,
               show_fig: bool = False):
    corr_list = []

    for col in cols:
        fig = plt.figure()
        col_corr = plt.xcorr(df[col], df[col_predicted], maxlags=max_lag, usevlines=True, normed=True, lw=2,
                             color="black")
        corr_ = pd.DataFrame(col_corr[1], index=col_corr[0], columns=["lag_corr"])
        print(corr_)
        highest_lag = corr_.iloc[0:].sort_values("lag_corr", ascending=True).iloc[-1].name

        plt.title(col)
        if not show_fig:
            plt.close(fig)
        else:
            plt.show()

        corr_list.append([col, highest_lag])

    return corr_list
