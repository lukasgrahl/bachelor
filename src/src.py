import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import adfuller


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
        assert np.isnan(arr).sum() == 0, "arr contains nan"
        pass

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

        self._check_sanity(arr)

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

        self._check_sanity(arr)

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
                 df_dictionary: dict):

        self.df = df_in.copy()
        self.df_dictionary = df_dictionary.copy()
        pass

    def _translate_neg_dist(self,
                            arr):
        if min(arr) < 0:
            return True, arr + (abs(arr.min()) + 1)
        else:
            return False, arr
        pass

    def get_log_returns(self,
                        cols: list):

        dist_translation = []

        for col in cols:
            is_trans, self.df[col] = self._translate_neg_dist(self.df[col])
            dist_translation.append(is_trans)

            self.df[col] = np.log1p(self.df[col] / self.df[col].shift(1))

        dist_translation = dict(zip(cols, dist_translation))
        log_transform = dict(zip(cols, list([True] * len(cols))))

        self.df_dictionary.update(neg_dist_trans=dist_translation)
        self.df_dictionary.update(log_transform=log_transform)

        return self.df
