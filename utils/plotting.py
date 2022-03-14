import lightgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import learning_curve, TimeSeriesSplit


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,
           verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x),
                       rand_jitter(y),
                       s=s,
                       c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha,
                       linewidths=linewidths, **kwargs)


def corr_heatmap(df: pd.DataFrame,
                 cols: list,
                 show_fig: bool = True,
                 size_factor: float = .7,
                 **kwargs):
    """
    Pearson correlation matrix
    :param df: dataframe with cols
    :param cols: cols to be correlated
    :param show_fig: bool
    :param size_factor: size of one matrix field
    :param kwargs:
    :return:
    """
    corr_matrix = df[cols].corr()
    mask = np.triu(corr_matrix)

    fig = plt.figure(figsize=(size_factor * len(corr_matrix), size_factor * len(corr_matrix)))
    sns.heatmap(abs(corr_matrix),
                annot=True,
                cmap="Blues",
                vmin=0,
                vmax=1,
                mask=mask,
                cbar=False,
                **kwargs)
    plt.tight_layout()
    if not show_fig:
        plt.close(fig)


def kde_plot(arr,
             **kwargs):
    """
    Seaborn kde plot wrapper
    :param arr:
    :param kwargs:
    :return:
    """
    if "figsize" in kwargs.keys():
        plt.figure(figsize=kwargs["figsize"])
        kwargs.pop("figsize")
    else:
        plt.figure(figsize=(5, 4))
    sns.kdeplot(arr, **kwargs)
    if type(arr) == pd.Series:
        plt.title(arr.name)
    plt.tight_layout()
    plt.show()
    pass


def corr_plot(x,
              vals,
              show_plot: bool = False,
              title=None,
              *args,
              **kwargs):
    """
    Plot lag correlation and lags
    :param x: array of lags
    :param vals: array of corresponding correlations
    :param show_plot: bool
    :param title: Plot title
    :param kwargs:
    """
    if show_plot:
        plt.figure(**kwargs)
        plt.bar(x, vals, width=.2, color="black")
        plt.plot(x, list([0] * len(x)), color="black", lw=.8)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    pass


# cross corr
def cross_corr(arr_x,
               arr_y,
               no_lags: int = 10,
               *args,
               **kwargs):
    """
    Lagged pearson correlation for two arrays
    :param arr_x: array to be lagged
    :param arr_y: reference array
    :param no_lags: no of lags and leads
    :param kwargs: show_plot
    :return:
    """
    corr_list = []
    lags = range(-no_lags, no_lags + 1)
    for i in lags:
        corr_list.append(arr_x.shift(i).corr(arr_y))

    corr_plot(lags, corr_list, title=arr_x.name, *args, **kwargs)

    return list(lags), corr_list


def df_cross_corr(df: pd.DataFrame,
                  cols_x: list,
                  pred_y: str,
                  no_lags: int = 10,
                  *args,
                  **kwargs):
    """
    Time lagged pearson cross correlation
    :param df: dataframe to be lagged
    :param cols_x: cols to be lagged against pred_y
    :param pred_y: col
    :param no_lags: no of lags and leads future (range(-no_lags, no_lags))
    :param kwargs: show_plot
    :return: [[col_x, best lag, best lag abs(corr)]]
    """
    corr_res = []
    lags = range(-no_lags, no_lags + 1)
    for col in cols_x:
        _, corr = cross_corr(df[col], df[pred_y], no_lags=no_lags, *args, **kwargs)
        corr_res.append(corr)

    df_corr = pd.DataFrame(corr_res, index=cols_x, columns=lags).transpose()
    df_corr = df_corr.loc[0:].abs()

    results = []
    for col in cols_x:
        results.append([col, df_corr[col].idxmax(), round(df_corr[col].max(), 3)])
    return results


def plot_lgbm_learning_curve(params: dict,
                             lgb_train: lightgbm.Dataset,
                             lgb_test: lightgbm.Dataset,
                             plot_title: str,
                             n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits)

    cv_train = lightgbm.cv(params,
                           lgb_train,
                           folds=tscv)
    cv_test = lightgbm.cv(params,
                          lgb_test,
                          folds=tscv)

    cv_train = {item: np.array(cv_train[item]) for item in cv_train.keys()}
    cv_test = {item: np.array(cv_test[item]) for item in cv_test.keys()}

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(cv_test["l2-mean"], color="blue")
    ax.plot(cv_train["l2-mean"], color="red")

    # plt.fill_between(list(range(0, len(cv_test["l2-mean"]))), cv_test["l2-mean"] + cv_test["l2-stdv"], cv_test["l2-mean"] - cv_test["l2-stdv"], color="blue", alpha=.1)
    # plt.fill_between(list(range(0, len(cv_train["l2-mean"]))), cv_train["l2-mean"] + cv_train["l2-stdv"], cv_train["l2-mean"] - cv_train["l2-stdv"], color="red", alpha=.1)
    plt.title(plot_title)
    plt.legend(["test", "train", "test_std", "train_std"])
    plt.tight_layout()
    plt.show()
    return fig


def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        scoring: str = "neg_mean_squared_error",
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    scoring: sklearn scoring metrics
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")

    return fig
