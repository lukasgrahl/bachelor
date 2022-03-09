import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm

from settings import random_state
from utils.cast_data import apply_date_to_week
from utils.plotting import corr_plot


def cut_to_weekly_data(df: pd.DataFrame,
                       relevant_cols: list = ["all"]):
    if "week" not in df.columns:
        df["week"] = df["date"].apply(lambda x: apply_date_to_week(x))

    if relevant_cols != ["all"]:
        df = df[relevant_cols]

    return df.dropna(axis=0).drop_duplicates("week")


def translate_neg_dist(arr):
    if min(arr) <= 0:
        return True, arr + (abs(arr.min()) + 1)
    else:
        return False, arr
    pass


def df_log_return(df_in: pd.DataFrame,
                  cols: list):
    df = df_in.copy()
    dist_trans = []

    for col in cols:
        is_trans, df[col] = arr_log_return(df[col])
        dist_trans.append(is_trans)

    dist_translation = dict(zip(cols, dist_trans))
    log_returns = dict(zip(cols, list([True] * len(cols))))

    df.dropna(inplace=True)

    return df, dist_translation, log_returns


def update_dict(dict_in: dict,
                update_keys: list,
                update_vals: list):
    dict_ = dict_in.copy()

    for i, key in enumerate(update_keys):
        dict_[key] = update_vals[i]

    return dict_


def arr_log_return(arr: pd.Series):
    # Assumption, df is ordered past to future
    is_trans, arr = translate_neg_dist(arr)
    return is_trans, np.log(1 + arr.pct_change())


def arr_inv_log_returns(arr):
    return np.exp(arr)


def shift_var_relative_to_df(df_in,
                             shift_var: list,
                             new_var_name: list = None,
                             no_lags: int = [1]):
    df = df_in.copy()

    if max(no_lags) < 0:
        print("Applying shifts in future")

    if new_var_name is None:
        assert len(no_lags) == len(shift_var), "shift_var and no_lags don't correspond"
        for i, var in enumerate(shift_var):
            df[var] = df[var].shift(no_lags[i])

        return df.dropna(axis=0)

    if new_var_name is not None:
        assert len(no_lags) == len(new_var_name) == len(shift_var), "Please name new cols"

        for i, var in enumerate(shift_var):
            df[new_var_name[i]] = df[var].shift(no_lags[i])

        return df.dropna(axis=0)


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

    for i, _ in enumerate(tts):
        tts[i] = tts[i].reset_index(drop=True)

    return tts


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


def cross_corr(arr_x,
               arr_y,
               no_lags: int = 10,
               **kwargs):
    cross_corr = []
    lags = range(-no_lags, no_lags + 1)
    for i in lags:
        cross_corr.append(arr_x.shift(i).corr(arr_y))

    corr_plot(lags, cross_corr, title=arr_x.name, **kwargs)

    return list(lags), cross_corr


def df_cross_corr(df,
                  cols_x,
                  pred_y,
                  no_lags: int = 10,
                  **kwargs):
    corr_res = []
    for col in cols_x:
        lags, corr = cross_corr(df[col], df[pred_y], **kwargs)
        corr_res.append(corr)

    df_corr = pd.DataFrame(corr_res, index=cols_x, columns=lags).transpose()
    df_corr = df_corr.loc[0:].abs()

    results = []
    for col in cols_x:
        results.append([col, df_corr[col].idxmax(), round(df_corr[col].max(), 3)])
    return results


def get_variance_inflation_factor(df,
                                  cols,
                                  col_pred):
    vif = [variance_inflation_factor(df[cols].values, i) for i in range(df[cols].shape[1])]
    vif = pd.DataFrame(index=cols, data=vif, columns=["VIF"])
    vif = vif.join(df[cols].corrwith(df[col_pred]).rename(f"corr_{col_pred}"))
    return vif.sort_values(f"corr_{col_pred}")


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
