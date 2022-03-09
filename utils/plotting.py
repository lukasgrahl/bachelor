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
              **kwargs):
    if show_plot:
        plt.figure(**kwargs)
        plt.bar(x, vals, width=.2, color="black", **kwargs)
        plt.plot(x, list([0] * len(x)), color="black", lw=.8, **kwargs)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    pass
