import lightgbm
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit

from src.src import StatsTest
from utils.analysis import plot_learning_curve
from utils.plotting import plot_lgbm_learning_curve
from utils.utils import get_tcsv, suppress_cmd_print, add_constant


class SKLearnWrapOLS(BaseEstimator, RegressorMixin):

    def __init__(self,
                 model_class,
                 fit_intercept: bool = True):
        """
        SKLearn Wrapper for Statsmodels models
        :param model_class: Statsmodel.model e.g. sm.OLS
        :param fit_intercept:
        """
        self.trained_model = None
        self.model_class = model_class
        self.fit_intercept = fit_intercept

        self.model = None
        self.results = None

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


class SKLearnWrapARIMA(BaseEstimator, RegressorMixin):

    def __init__(self,
                 model_class):
        """
        SKLearn Wrapper for Statsmodels models
        :param model_class: Statsmodel.model e.g. sm.OLS
        :param fit_intercept:
        """
        self.model_class = model_class

        super().__init__()

        self.model = None
        self.results = None
        pass

    def fit(self, y, order: tuple, **kwargs):
        """
        Fit trainig data to model
        :param X: X_train
        :param y: y_train
        :return: fitted model
        """

        self.model = self.model_class(y, order=order, **kwargs)
        self.trained_model = self.model.fit()
        return self.trained_model

    def predict(self):
        """
        Model prediction
        :param X: X_test
        :return: predictions for X
        """
        return np.array(self.trained_model.forecast())[0]


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
        pass

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
                 # categorical_features: list,
                 early_stopping_rounds: int = 10000,
                 suppress_lgb_print_output: bool = True,
                 debug_single_pred: bool = False):

        self.lgb_test = None
        self.lgb_train = None
        self.eval_results = None
        self.model = None
        self.y_pred = None
        self.y_true = None
        # self.categorical_features = categorical_features
        self.early_stopping = early_stopping_rounds
        self.params = params
        self.model_in = model_in
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.suppress_lgb_print_output = suppress_lgb_print_output
        self.debug_single_pred = debug_single_pred

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
            _eval_results = {}

            if self.suppress_lgb_print_output:
                with suppress_cmd_print():
                    lgb_train = lightgbm.Dataset(self.X.iloc[self.training_index[i]],
                                                 self.y.iloc[self.training_index[i]],
                                                 # categorical_feature=self.categorical_features,
                                                 free_raw_data=False)

                    lgb_test = lightgbm.Dataset(self.X.iloc[self.testing_index[i]],
                                                self.y.iloc[self.testing_index[i]],
                                                # categorical_feature=self.categorical_features,
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
                                                  # categorical_feature=self.categorical_features,
                                                  free_raw_data=False)

                self.lgb_test = lightgbm.Dataset(self.X_test,
                                                 self.y_test,
                                                 # categorical_feature=self.categorical_features,
                                                 free_raw_data=False)

            if self.debug_single_pred:
                self.y_true = self.y_test
                return model_.predict(self.X_test)

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
            fig = plot_lgbm_learning_curve(self.params,
                                           self.lgb_train,
                                           self.lgb_test,
                                      plot_title,
                                      n_splits=(len(self.X_test) - 2))
            return fig


class ExpandingPredictionARIMA:

    def __init__(self,
                 model_in,
                 arima_order,
                 target_var: str,
                 X_train,
                 y_train,
                 X_test,
                 y_test):

        self.training_index = None
        self.testing_index = None
        self.model = None
        self.y_pred = None

        self.target_var = target_var
        self.arima_order = arima_order
        self.model_in = model_in
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test
        self.y_train = y_train

        self.X = pd.concat([X_train,
                            X_test])
        self.y = pd.concat([y_train,
                            y_test])

        if 'intercept' in self.X.columns:
            self.X.drop('intercept', axis=1, inplace=True)
        pass

    def predict(self,
                X_,
                **kwargs):
        self.training_index = list(range(0, len(self.X_train)))
        self.testing_index = list(range(len(self.X_train), len(self.X)))
        self.y_pred = []

        for i in self.testing_index:
            model = self.model_in
            model.fit(self.X.loc[: i - 1][self.target_var], order=self.arima_order)

            pred = model.predict()

            self.y_pred.append(pred)

            if i == self.testing_index[-2]:
                self.model = model

        return self.y_pred


class RandomWalk:

    def __init__(self,
                 X_train,
                 X_test,
                 y_test,
                 y_train):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train

        self.X = pd.concat([X_train, X_test])
        self.y_pred = None
        pass

    def _test_zero_mean(self):
        stest = StatsTest(print_results=True)
        is_test = stest.arr_ttest_1samp(self.y_train, mean=0)
        is_test = is_test
        print('\n')
        print('Testing for zero mean')
        print(f'Time series has a zero mean: {is_test}')
        print(f'Random walk requires a drift: {~is_test}')
        return is_test

    def predict(self,
                X_):
        is_zeromean = self._test_zero_mean()
        if is_zeromean:
            self.y_pred = list([0] * len(self.X_test))
        else:
            self.y_pred = list([self.y_train.mean()] * len(self.X_test))
            print(f"\n Random walk with drift (mean return): {self.y_train.mean()}")

        return self.y_pred

    def plot_learning_curve(self,
                            plot_title: None,
                            **kwargs):
        if 'scoring' in kwargs.keys():
            kwargs.pop('scoring')
        return None
