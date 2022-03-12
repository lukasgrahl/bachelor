import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


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
              show_plot,
              title=None,
              *args,
              **kwargs):
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
    corr_list = []
    lags = range(-no_lags, no_lags + 1)
    for i in lags:
        corr_list.append(arr_x.shift(i).corr(arr_y))

    corr_plot(lags, corr_list, title=arr_x.name, *args, **kwargs)

    return list(lags), corr_list


def df_cross_corr(df,
                  cols_x,
                  pred_y,
                  no_lags: int = 10,
                  *args,
                  **kwargs):
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
