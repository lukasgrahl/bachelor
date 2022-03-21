import os

import lightgbm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller
from yellowbrick.model_selection import LearningCurve

from settings import random_state
from utils.plotting import plot_learning_curve, plot_lgbm_learning_curve
from utils.utils import arr_inv_log_returns, add_constant, get_performance_metrics, suppress_cmd_print


class StatsTest:

    def __init__(self,
                 significance=0.05,
                 plot: bool = True,
                 print_results: bool = False,
                 **kwargs):

        """
        Class for statistic test on df as well as arr level
        :param significance: signifcance level
        :param plot: plot data bool
        :param print_results: print test report to cmd
        :param kwargs:
        """

        self.significance = significance
        self.plot = plot
        self.print_results = print_results
        self.kwargs = kwargs
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

    def _qq_kde_plot(self,
                     arr,
                     is_test):

        if self.plot is True:

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            _ = stats.probplot(arr, dist="norm", plot=ax[0])
            sns.kdeplot(arr, ax=ax[1])
            if type(arr) == pd.core.series.Series:
                ax[0].set_title(f"{arr.name}: {is_test}")
                ax[1].set_title(f"{arr.name}: {is_test}")
            plt.tight_layout()
            plt.show()
        pass

    def arr_stationarity_adfuller(self,
                                  arr,
                                  **kwargs):
        """
        Applies statsmodel adfuller test to array
        :param arr:
        :param kwargs:
        :return:
        """

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
        """
        Applies statsmodel normaltest to arr
        :param arr:
        :param kwargs:
        :return:
        """
        # H0: sample comes from a normal distribution

        arr = self._check_sanity(arr)

        test_stat, pvalue = stats.normaltest(arr,
                                             **kwargs)
        is_normal = pvalue > self.significance

        self._qq_kde_plot(arr, is_normal)

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
        """
        Applies arr_test_stationarity to cols in df_in
        :param df_in:
        :param cols:
        :param kwargs:
        :return:
        """
        stationary = []
        for col in cols:
            is_stationary = self.arr_stationarity_adfuller(df_in[col], **kwargs)
            stationary.append(is_stationary)

        return dict(zip(cols, stationary))

    def df_test_normality(self,
                          df_in: pd.DataFrame,
                          cols: list,
                          **kwargs):
        """
        Applies arr_test_normality to cols in df_in
        :param df_in:
        :param cols:
        :param kwargs:
        :return:
        """
        normality = []
        for col in cols:
            is_normal = self.arr_test_normality(df_in[col])
            normality.append(is_normal)

        return dict(zip(cols, normality))

    def df_heteroskedasticity_white(self,
                                    y_in,
                                    X_in):
        """
        Applies het_white test for homoscedasticity to y drawing X from dataframe
        :param y_in:
        :param X_in:
        :return:
        """
        # H0: Homoscedasticity is present (residuals are equally scattered)

        X = X_in.copy()
        y = y_in.copy()

        # Check for constant
        list_1 = []
        for item in np.mean(X, axis=0):
            if item == 1:
                list_1.append(True)
            else:
                pass
        if len(list_1) != 1:
            print("HET WHITE TEST REQUIRES A CONSTANT IN X DATA")
            print("adding constant to data")
            X = add_constant(X)

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


class ModelValidation:

    def __init__(self,
                 X_train,
                 y_train,
                 X_validate,
                 y_validate,
                 model,
                 print_results: bool = False):

        """
        General model validation class: Performs residual analysis and plots learning curve
        :param X_train:
        :param y_train:
        :param X_validate:
        :param y_validate:
        :param model:
        :param print_results: print results to cmd bool
        """

        self.print_results = print_results

        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

        self.X_test = X_validate.reset_index(drop=True)
        self.y_test = y_validate.reset_index(drop=True)
        self.model = model

        self.y_pred = None
        self.resid = None

        self.resid_inv = None
        self.y_pred_inv = None
        self.y_test_inv = None

        self.df_r = None

        self.rmse = None
        self.mse = None
        self.mae = None
        self.r2 = None

        self._get_predictions()
        self._invers_trans_y()

    def _get_predictions(self):
        self.y_pred = self.model.predict(self.X_test)
        pass

    def _invers_trans_y(self):
        self.y_pred_inv = arr_inv_log_returns(self.y_pred)
        self.y_test_inv = arr_inv_log_returns(self.y_test)
        pass

    def _get_resids(self):
        self.resid = (self.y_test - self.y_pred).rename("residuals")
        self.resid_inv = (self.y_test_inv - self.y_pred_inv).rename("inv_residuals")
        pass

    def _plot_pred_vs_true(self,
                           plot_inv: bool = False):
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        if plot_inv:
            ax.plot(self.y_test_inv)
            ax.plot(self.y_pred_inv)
        else:
            ax.plot(self.y_test)
            ax.plot(self.y_pred)
        plt.title("True vs. Predicted")
        plt.legend(["y_test", "y_pred"])
        plt.tight_layout()
        plt.show()
        return fig

    def get_model_performance(self,
                              **kwargs):
        """
        Get model performance measures (rmse, mse, mae, r2) and plot y_test vs. y_true
        :param kwargs: plot_inv: plot inverse transformed values
        :return: plt.figure
        """
        self.rmse, self.mse, self.mae, self.r2 = get_performance_metrics(self.y_test, self.y_pred)
        fig = self._plot_pred_vs_true(**kwargs)

        if self.print_results:
            print("Validation Scores")
            print(f'root mean squared error: {self.rmse}')
            print(f'mean squared error: {self.mse}')
            print(f'mean absolute error: {self.mae}')
            print(f'R2: {self.r2}')
        return fig

    def analyse_resids(self,
                       **kwargs):
        """
        Plots and performs statistics tests (stationarity, normality, heteroskedasticity) on residuals
        :param kwargs:
        :return: booleans for: stationarity, normality, heteroskedasticity
        """
        self._get_resids()

        stest = StatsTest(**kwargs, print_results=self.print_results)
        stationarity = stest.arr_stationarity_adfuller(self.resid)
        normality = stest.arr_test_normality(self.resid)
        heteroskedasticity = stest.df_heteroskedasticity_white(y_in=self.resid, X_in=self.X_test)

        return stationarity, normality, heteroskedasticity

    def _plot_learning_curve(self,
                             scoring: str = "r2",
                             n_splits: int = 5,
                             max_train_size=None,
                             test_size=None):
        """
        Plot learning curve for
        :param scoring:
        :param n_splits:
        :param max_train_size:
        :param test_size:
        :return:
        """

        tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size)
        visualizer = LearningCurve(self.model, cv=tscv, random_state=random_state, scoring=scoring)
        visualizer.fit(self.X_train, self.y_train)
        plt.tight_layout()
        visualizer.show()
        pass

    def sm_learning_curve(self,
                          plot_title: str,
                          scoring: str = "neg_mean_squared_error",
                          n_splits: int = 5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fig = plot_learning_curve(self.model, plot_title, self.X_test, self.y_test, cv=tscv, scoring=scoring)
        return fig

    def lgbm_learning_curve(self,
                            params: dict,
                            lgb_train: lightgbm.Dataset,
                            lgb_test: lightgbm.Dataset,
                            plot_title: str,
                            n_splits: int = 5,
                            suppress_lgb_print_output: bool = True):

        plot_title = f"{plot_title} metric: {params['metric'][0]}"

        if suppress_lgb_print_output:
            with suppress_cmd_print():
                fig = plot_lgbm_learning_curve(params, lgb_train, lgb_test, plot_title, n_splits)
        else:
            fig = plot_lgbm_learning_curve(params, lgb_train, lgb_test, plot_title, n_splits)
        return fig

    def plot_results_on_price_scale(self,
                                    df_weekly: pd.DataFrame,
                                    df_weekly_sub: pd.DataFrame,
                                    sp_true_vals: str,
                                    xlim=None,
                                    show_pred_only: bool = False):
        assert sp_true_vals in df_weekly.columns, f"Please add {sp_true_vals} to df_weekly"

        self.df_r = df_weekly.loc[df_weekly_sub.index].reset_index(drop=True).copy()

        self.df_r["sp_tot_pred_test"] = np.concatenate([np.array(list([np.nan] * len(self.X_train))),
                                                        (self.y_pred + 1)]) * self.df_r[sp_true_vals]

        self.df_r["sp_tot_pred_train"] = np.concatenate([(self.model.predict(self.X_train) + 1),
                                                         np.array(list([np.nan] * len(self.y_pred)))]) * self.df_r[sp_true_vals]

        self.df_r["sp_tot_pred_train"] = self.df_r["sp_tot_pred_train"].shift(1)
        self.df_r["sp_tot_pred_test"] = self.df_r["sp_tot_pred_test"].shift(1)

        fig, ax = plt.subplots(1, 1, figsize=(30, 8))
        ax.plot(self.df_r["sp_tot_pred_train"], marker="o", lw=.5, alpha=.3, color="blue")
        ax.plot(self.df_r["sp_tot_pred_test"], marker="o", lw=.5, alpha=.3, color="red")
        ax.plot(self.df_r[sp_true_vals], color="black", lw=.8)
        plt.title("True vs. predicted prices")

        if show_pred_only:
            plt.xlim([len(self.df_r) - len(self.y_pred), len(self.df_r)])
            assert xlim is None, "show_pred_only and xlim are mutually exclusive"
        else:
            if xlim is not None:
                plt.xlim(xlim)
        plt.tight_layout()
        plt.show()

        rmse, mse, mae, r2 = get_performance_metrics(self.df_r.loc[self.df_r.sp_tot_pred_test.dropna().index,
                                                                   sp_true_vals],
                                                     self.df_r.sp_tot_pred_test.dropna())
        if self.print_results:
            print("Validation Scores Test Data")
            print(f"mean squared error: {mse}")
            print(f"mean absolute error: {mae}")
            print(f"mean absolute error in %: {mae / np.mean(self.df_r[sp_true_vals])}")
            print(f"r2: {r2}")
        return fig


class SKLearnWrap(BaseEstimator, RegressorMixin):

    def __init__(self,
                 model_class,
                 fit_intercept: bool = True):
        """
        SKLearn Wrapper for Statsmodels models
        :param model_class: Statsmodel.model e.g. sm.OLS
        :param fit_intercept:
        """
        self.model_class = model_class
        self.fit_intercept = fit_intercept

        self.model_ = None
        self.results_ = None
        pass

    def fit(self, X, y):
        """
        Fit trainig data to model
        :param X: X_train
        :param y: y_train
        :return: fitted model
        """
        if self.fit_intercept:
            X = add_constant(X, constant_value=1)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        """
        Model prediction
        :param X: X_test
        :return: predictions for X
        """
        if self.fit_intercept:
            X = add_constant(X, constant_value=1)
        return self.results_.predict(X)
