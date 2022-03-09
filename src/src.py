import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller

from utils.utils import arr_log_return, arr_log_transform, arr_inv_log_returns


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
        return is_stationary

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

        return is_normal

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

    def df_heteroskedasticity_white(self,
                                    y,
                                    X):
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

        bool_result = test_result["Test Statistic p-value"] < self.significance

        self._line_plot(arr=y, is_test=bool_result)
        if self.print_results:
            print("Test for Heteroskedasticity")
            print(f"Test p-value: {test_result['Test Statistic p-value']}")
            print(f"Heteroskedasticity is present: {bool_result}")

        return bool_result


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

    def df_transform(self,
                     cols: list,
                     transformation: str = "log_return",
                     dict_trans_name: str = None):

        """

        :param dict_trans_name: name of transformation in data_dict, default: dict_trans_name = transformation
        :param cols: columns to apply transformation on
        :param transformation: log_return, log_trans
        :return: df and data_dict will be altered internally, call them as class attributed after transforming
        """
        dist_translation = []

        for col in cols:
            if transformation == "log_return":
                is_trans, self.df[col] = arr_log_return(self.df[col])
                pass
            elif transformation == "log_trans":
                is_trans, self.df[col] = arr_log_transform(self.df[col])
                pass
            dist_translation.append(is_trans)

        dist_translation = dict(zip(cols, dist_translation))
        self._update_neg_trans_dict(dist_translation)

        trans_cols = dict(zip(cols, list([True] * len(cols))))

        if dict_trans_name is None:
            self.dict_[transformation] = trans_cols
        else:
            self.dict_[dict_trans_name] = trans_cols

        pass


class ModelValidation:

    def __init__(self,
                 X_validate,
                 y_validate,
                 model,
                 data_dict):

        self.X = X_validate
        self.y = y_validate
        self.model = model
        self.dict_ = data_dict

        self.pred = None
        self.resid = None
        self.resid_inv = None
        self.pred_inv = None
        self.y_inv = None

        self.mse = None
        self.mae = None
        self.r2 = None

        self._get_predictions()
        self._invers_trans_y()
        pass

    def _get_predictions(self):
        self.pred = self.model.predict(self.X)
        pass

    def _invers_trans_y(self):
        self.pred_inv = arr_inv_log_returns(self.pred)
        self.y_inv = arr_inv_log_returns(self.y)
        pass

    def _get_resids(self):
        self.resid = self.y - self.pred
        self.resid_inv = self.y_inv - self.pred_inv
        pass

    def _plot_pred_vs_true(self):
        plt.figure(figsize=(20, 5))
        plt.plot(self.y_inv)
        plt.plot(self.pred_inv)
        plt.legend(["y_test", "y_pred"])
        plt.show()

    def get_model_performance(self):
        from sklearn.metrics import r2_score
        self.mse = round(np.mean((self.pred_inv - self.y_inv) ** 2), 8)
        self.mae = round(np.mean(abs(self.pred_inv - self.y_inv)), 8)
        self.r2 = round(r2_score(self.y_inv, self.pred_inv), 4)

        self._plot_pred_vs_true()

        print("Validation Scores")
        print(f'mean squared error: {self.mse}')
        print(f'mean absolute error: {self.mae}')
        print(f'R2: {self.r2}')

    def analyse_resids(self,
                       **kwargs):
        self._get_resids()

        stest = StatsTest(**kwargs)
        stest.arr_stationarity_adfuller(self.resid)
        stest.arr_test_normality(self.resid)
        stest.df_heteroskedasticity_white(y=self.resid, X=self.X)
        pass


