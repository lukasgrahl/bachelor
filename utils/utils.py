import datetime as dt
import numpy as np


def apply_datetime_format(x):
    x = str(x)
    try:
        x = dt.datetime.strptime(x, "%Y-%m-%d")
        return x
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%d.%m.%Y")
        return x
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%d.%m.%y")
        return x
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%m.%d.%Y %H:%M:%S")
        return x
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return x
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
        return x
    except ValueError:
        pass

    if x == np.nan:
        return np.datetime64('NaT')
    elif x == "nan":
        return np.datetime64('NaT')

    print(f"Datetime Assignment failed with {x}")
    raise ValueError(501, x)
