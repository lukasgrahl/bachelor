import os

import lightgbm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from yellowbrick.model_selection import LearningCurve

from settings import random_state
from utils.plotting import plot_learning_curve, plot_lgbm_learning_curve
from utils.utils import arr_inv_log_returns, add_constant, get_performance_metrics, suppress_cmd_print, get_tcsv


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
        no_lags = test[2]
        is_stationary = pvalue < self.significance

        self._line_plot(arr, is_stationary)

        if self.print_results is True:
            print("Stationarity Test Results")
            print(f"P-Values: {pvalue}")
            print(f"Test-stats: {test_stat}")
            print(f"Time series is stationary: {is_stationary}")
            print(f'Number of lags used: {no_lags}')
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

    def arr_ttest_1samp(self,
                        arr,
                        mean):
        x = ttest_1samp(arr, 0)
        stat = x[0]
        pval = x[1]
        is_test = pval < self.significance

        if self.print_results:
            print(f"TTest one sample for mean: {mean}")
            print(f'Test statistics: {stat}')
            print(f'Test pvalue: {pval}')
            print(f'Population mean is equal to {mean}: {is_test}')
        return is_test

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
            is_stationary = self.arr_stationarity_adfuller(df_in[col], autolag="AIC", **kwargs)
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

    def arr_durbin_watson(self,
                          resid):
        pvalue = durbin_watson(resid)
        is_test = 1.5 < pvalue < 2.5
        if self.print_results:
            print(f'Durbin watson test for first oder autocorrelation')
            print(f'Test statistics: 1.5 < {round(pvalue, 3)} < 2.5')
            print(f'First order autocorrlation is not present: : {is_test}')
        return is_test


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
        self._get_resids()

    def _get_predictions(self):
        self.y_pred = self.model.predict(self.X_test)
        pass

    def _get_resids(self):
        self.resid = (self.y_test - self.y_pred).rename("residuals")
        pass

    def _plot_pred_vs_true(self):

        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
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
        print('\n')
        stationarity = stest.arr_stationarity_adfuller(self.resid)
        print('\n')
        normality = stest.arr_test_normality(self.resid)
        print('\n')
        heteroskedasticity = stest.df_heteroskedasticity_white(y_in=self.resid, X_in=self.X_test)
        print('\n')
        zero_mean = stest.arr_ttest_1samp(self.resid, mean=0)
        print('\n')
        d_watson = stest.arr_durbin_watson(self.resid)

        return stationarity, normality, heteroskedasticity, d_watson, zero_mean

    def learning_curve(self,
                       plot_title: str,
                       scoring: str = "neg_mean_squared_error"):

        fig = self.model.plot_learning_curve(plot_title, scoring=scoring)
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
                                                         np.array(list([np.nan] * len(self.y_pred)))]) * self.df_r[
                                             sp_true_vals]

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

        self.model = None
        self.results = None
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
        self.model = self.model_class(y, X)
        self.trained_model = self.model.fit()
        return self

    def predict(self, X):
        """
        Model prediction
        :param X: X_test
        :return: predictions for X
        """
        if self.fit_intercept:
            X = add_constant(X, constant_value=1)
        return self.trained_model.predict(X)


class SeasonalTrend:

    def __init__(self,
                 df,
                 time_series,
                 season_col,
                 intra_season_period_col,
                 seasonal_period,
                 show_fig: bool = False):

        """
        Class is dividing data into season and trend data
        :param df: data
        :param time_series: col on which seasonal splitting is performed
        :param season_col: column which indicates the season (year, quater, month, week): constant for each season
        :param intra_season_period_col: column which indicates the intra seasonal periods (week, day)
        :param seasonal_period: number of intra season periods by season
        :param show_fig: show plot
        """

        self.seasonal_estimator = None
        self.df = df.copy()
        self.time_series = time_series
        self.season_col = season_col
        self.intra_season_period_col = intra_season_period_col
        self.seasonal_period = seasonal_period

        self.show_fig = show_fig

        self.fig = None

    def _filter_for_complete_seasons(self):
        _ = self.df.groupby(self.season_col)[self.intra_season_period_col].count() == self.seasonal_period
        _ = _[_].index

        self.df = self.df[self.df[self.season_col].isin(_)]
        print(f'\n Len of complete weeks in X_train: {len(self.df)}')
        pass

    def split_time_series(self):
        self._filter_for_complete_seasons()
        # create statsmodel seasonality object
        self.seasonal_estimator = seasonal_decompose(self.df[self.time_series], period=self.seasonal_period)
        self.fig = self.seasonal_estimator.plot()

        if self.show_fig:
            plt.show()
        else:
            plt.close()

        # get seasonality by weekday
        self.seasonal = self.seasonal_estimator.seasonal
        self.trend = self.seasonal_estimator.trend
        self.df["seasonal"] = self.seasonal
        self.dict_map_sasonal = dict(zip(self.df.groupby(self.intra_season_period_col).seasonal.first().index,
                                         self.df.groupby(self.intra_season_period_col).seasonal.first().values))
        pass


class ExpandingPredictionOLS:

    def __init__(self,
                 model_in,
                 X_train,
                 y_train,
                 X_test,
                 y_test):

        self.training_index = None
        self.testing_index = None
        self.model = None
        self.y_pred = None

        self.model_in = model_in
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()

        self.X = pd.concat([X_train,
                            X_test])
        self.y = pd.concat([y_train,
                            y_test])

        if 'intercept' in self.X.columns:
            self.X.drop('intercept', axis=1, inplace=True)

    def predict(self,
                X_,
                **kwargs):
        self.training_index, self.testing_index = get_tcsv(self.X, self.y, n_splits=len(self.X_test))
        self.y_pred = []

        for i in range(0, len(self.training_index)):
            model = self.model_in

            model.fit(self.X.iloc[self.training_index[i]], self.y.iloc[self.training_index[i]])
            pred = model.predict(self.X.iloc[self.testing_index[i]]).values[0]

            self.y_pred.append(pred)

            if i == 0:
                self.model = model

        return self.y_pred

    def plot_learning_curve(self,
                            plot_title: str,
                            scoring: str):
        if self.model is None:
            self.predict('_')

        plot_title = f"{plot_title} metric: {scoring}"

        tscv = TimeSeriesSplit()
        fig = plot_learning_curve(self.model, plot_title, self.X_test, self.y_test, cv=tscv, scoring=scoring)
        return fig


class ExpandingPredictionLGB:

    def __init__(self,
                 model_in,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 params: dict,
                 categorical_features: list,
                 early_stopping_rounds: int = 10000,
                 suppress_lgb_print_output: bool = True):

        self.lgb_test = None
        self.lgb_train = None
        self.eval_results = None
        self.model = None
        self.y_pred = None
        self.y_true = None
        self.categorical_features = categorical_features
        self.early_stopping = early_stopping_rounds
        self.params = params
        self.model_in = model_in
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.suppress_lgb_print_output = suppress_lgb_print_output

        self.X = pd.concat([X_train,
                            X_test])
        self.y = pd.concat([y_train,
                            y_test])

    def predict(self,
                X_,
                **kwargs):
        self.training_index, self.testing_index = get_tcsv(self.X, self.y, n_splits=len(self.X_test))
        self.y_pred = []
        self.y_true = []

        self.eval_results = {}

        for i in range(0, len(self.training_index)):
            model = self.model_in
            _eval_results = {}

            if self.suppress_lgb_print_output:
                with suppress_cmd_print():
                    lgb_train = lightgbm.Dataset(self.X.iloc[self.training_index[i]],
                                                 self.y.iloc[self.training_index[i]],
                                                 categorical_feature=self.categorical_features,
                                                 free_raw_data=False)

                    lgb_test = lightgbm.Dataset(self.X.iloc[self.testing_index[i]],
                                                self.y.iloc[self.testing_index[i]],
                                                categorical_feature=self.categorical_features,
                                                free_raw_data=False,
                                                reference=lgb_train)

                    model_ = lightgbm.train(self.params,
                                            lgb_train,
                                            valid_sets=[lgb_test, lgb_train],
                                            callbacks=[lightgbm.early_stopping(self.early_stopping),
                                                       lightgbm.record_evaluation(_eval_results)])
            self.eval_results[f'eval_results_{i}'] = _eval_results

            pred = model_.predict(self.X.iloc[self.testing_index[i]])
            self.y_pred.append(pred[0])
            self.y_true.append(self.y.iloc[self.testing_index[i]])

            if i == 0:
                self.model = model_
                self.lgb_train = lightgbm.Dataset(self.X_train,
                                                  self.y_train,
                                                  categorical_feature=self.categorical_features,
                                                  free_raw_data=False)

                self.lgb_test = lightgbm.Dataset(self.X_test,
                                                 self.y_test,
                                                 categorical_feature=self.categorical_features,
                                                 free_raw_data=False)

        return self.y_pred

    def plot_learning_curve(self,
                            plot_title: str,
                            **kwargs):

        if 'scoring' in kwargs:
            kwargs.pop('scoring')

        if self.model is None:
            self.predict('_')

        plot_title = f"{plot_title} metric: {self.model.params['metric'][0]}"

        if self.suppress_lgb_print_output:
            with suppress_cmd_print():
                fig = plot_lgbm_learning_curve(self.params,
                                               self.lgb_train,
                                               self.lgb_test,
                                               plot_title,
                                               n_splits=(len(self.X_test) - 2))
                return fig
        else:
            fig = plot_learning_curve(self.params,
                                      self.lgb_train,
                                      self.lgb_test,
                                      plot_title,
                                      n_splits=(len(self.X_test) - 2))
            return fig


class RandomWalk:

    def __init__(self,
                 X_train,
                 X_test):
        self.X_train = X_train
        self.X_test = X_test

        self.X = pd.concat([X_train, X_test])
        self.y_pred = None
        pass

    def _test_zero_mean(self):
        stest = StatsTest(print_results=True)
        is_test = stest.arr_ttest_1samp(self.X, mean=0)
        is_test = is_test[0]
        print('\n')
        print('Testing for zero mean')
        print(f'Time series has a zero mean: {is_test}')
        print(f'Random walk requires a drift: {~is_test}')
        pass

    def predict(self,
                X_):
        self._test_zero_mean()

        self.y_pred = self.X_test.values.reshape(-1)
        return self.y_pred

    def plot_learning_curve(self,
                            plot_title: None,
                            **kwargs):
        if 'scoring' in kwargs.keys():
            kwargs.pop('scoring')
        return None
