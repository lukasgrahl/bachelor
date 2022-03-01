import numpy as np
import matplotlib.pyplot as plt

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