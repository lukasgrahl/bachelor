import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_white

from utils.utils import arr_log_return, arr_log_transform


class StatsTest:

    def __init__(self,
                 significance=0.05,
                 plot: bool = True,
                 print_results: bool = False):

        self.significance = significance
        self.plot = plot
        self.print_results = print_results
        pass

    def _check_sanity(self,
                      arr):
        if np.isnan(arr).sum() > 0:
            print(f"arr contains nan, dropping {np.isnan(arr).sum()} NaNs")
            arr.dropna(inplace=True)
        return arr

    def _line_plot(self,
                   arr,
                   is_test):
        if self.plot is True:
            fig = plt.figure(figsize=(5, 3))
            plt.plot(arr)
            if type(arr) == pd.core.series.Series:
                plt.title(f"{arr.name}: {is_test}")
            plt.tight_layout()
            plt.show()
        pass

    def _qq_plot(self,
                 arr,
                 is_test):

        if self.plot is True:
            fig, ax = plt.subplots()
            _ = stats.probplot(arr, dist="norm", plot=ax)
            if type(arr) == pd.core.series.Series:
                plt.title(f"{arr.name}: {is_test}")
            plt.tight_layout()
            plt.show()
        pass

    def arr_stationarity_adfuller(self,
                                  arr,
                                  **kwargs):

        # H0: The time series is non-stationary

        arr = self._check_sanity(arr)

        test = adfuller(arr, **kwargs)
        pvalue = test[1]
        test_stat = test[0]
        is_stationary = pvalue < self.significance

        self._line_plot(arr, is_stationary)

        if self.print_results is True:
            print("Stationarity Test Results")
            print(f"P-Values: {pvalue}")
            print(f"Test-stats: {test_stat}")
            print(f"Time series is stationary: {is_stationary}")
            pass
        return pvalue < self.significance

    def arr_test_normality(self,
                           arr,
                           **kwargs):
        # H0: sample comes from a normal distribution

        arr = self._check_sanity(arr)

        test_stat, pvalue = stats.normaltest(arr,
                                             **kwargs)
        is_normal = pvalue > self.significance

        self._qq_plot(arr, is_normal)

        if self.print_results is True:
            print("Normality Test Results")
            print(f"P-Values: {pvalue}")
            print(f"Test-stats: {test_stat}")
            print(f"Series is normally distributed: {is_normal}")

        return pvalue > self.significance

    def df_test_stationarity(self,
                             df_in: pd.DataFrame,
                             cols: list,
                             **kwargs):
        stationary = []
        for col in cols:
            is_stationary = self.arr_stationarity_adfuller(df_in[col])
            stationary.append(is_stationary)

        return dict(zip(cols, stationary))

    def df_test_normality(self,
                          df_in: pd.DataFrame,
                          cols: list,
                          **kwargs):
        normality = []
        for col in cols:
            is_normal = self.arr_test_normality(df_in[col])
            normality.append(is_normal)

        return dict(zip(cols, normality))


class DataTransformation:

    def __init__(self,
                 df_in: pd.DataFrame,
                 data_dict: dict,
                 neg_dist_trans_key: str = "neg_dist_trans"):

        self.df = df_in.copy()
        self.dict_ = data_dict.copy()
        self.neg_dist_trans_key = neg_dist_trans_key
        pass

    def _update_neg_trans_dict(self,
                               new_dict,
                               some_value=False):

        if self.neg_dist_trans_key in self.dict_.keys():
            old_dict = self.dict_[self.neg_dist_trans_key]

            for item in new_dict.keys():
                if item in old_dict.keys():
                    if old_dict[item] == some_value:
                        old_dict[item] = new_dict[item]
                    else:
                        old_dict[item] = old_dict[item]
                else:
                    old_dict[item] = new_dict[item]

            updated_dict = old_dict.copy()

            # updated_dict = {item: new_dict[item] if old_dict[item] == False else old_dict[item] for item in new_dict.keys()}
            self.dict_[self.neg_dist_trans_key] = updated_dict
        else:
            self.dict_[self.neg_dist_trans_key] = new_dict

    def df_log_returns(self,
                       cols: list):
        dist_translation = []

        for col in cols:
            is_trans, self.df[col] = arr_log_return(self.df[col])
            dist_translation.append(is_trans)

        dist_translation = dict(zip(cols, dist_translation))
        self._update_neg_trans_dict(dist_translation)

        log_return = dict(zip(cols, list([True] * len(cols))))
        self.dict_.update(log_return=log_return)

        pass

    def df_log_transform(self,
                         cols: list):
        trans_dist = []

        for col in cols:
            is_trans, self.df[col] = arr_log_transform(self.df[col])
            trans_dist.append(is_trans)

        trans_dist = dict(zip(cols, trans_dist))
        self._update_neg_trans_dict(trans_dist)

        log_trans = dict(zip(cols, list([True] * len(cols))))
        self.dict_.update(log_trans=log_trans)

        pass
