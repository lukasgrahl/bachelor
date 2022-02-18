from functools import wraps
import datetime as dt

cast_dict = {
    'CMA': float,
    'HML': float,
    'Mean/Average': float,
    'Mkt-RF': float,
    'Most Bearish Response': float,
    'Most Bullish Response': float,
    'NAAIM Number': float,
    'Quart 1 (25% at/below)': float,
    'Quart 2 (median)': float,
    'Quart 3 (25% at/above)': float,
    'RF': float,
    'RMW': float,
    'S&P 500': float,
    'SMB': float,
    'Standard Deviation': float,
    'date': 'date',
    'fears25': float,
    'fears30': float,
    'fears35': float,
    'termspread': float,
    'vix': float,
    'vixh': float,
    'vixl': float,
    'vixo': float,
    'vxd': float,
    'vxdh': float,
    'vxdl': float,
    'vxdo': float,
    'vxn': float,
    'vxnh': float,
    'vxnl': float,
    'vxno': float,
    'vxo': float,
    'vxoh': float,
    'vxol': float,
    'vxoo': float
}


def apply_datetime_format(x,
                          ret_time: bool = False):
    x = str(x)
    try:
        x = dt.datetime.strptime(x, "%Y-%m-%d")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%d.%m.%Y")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%d.%m.%Y")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%m/%d/%Y")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%Y%m%d")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%m.%d.%Y %H:%M:%S")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    try:
        x = dt.datetime.strptime(x, "%m/%d/%Y %H:%M:%S")
        if ret_time is False:
            return x.date()
    except ValueError:
        pass

    raise ValueError(501, f"{x} format unknonw")


def cast_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)

        unknown_cols = []
        for col in df.columns:
            try:
                if cast_dict[col] == "date":
                    for i in df[col].index:
                        df.loc[i, col] = apply_datetime_format(df.loc[i, col], **kwargs)
                else:
                    df[col] = df[col].astype(cast_dict[col])
            except Exception as e:
                unknown_cols.append(col)
                pass

        if len(unknown_cols) > 0:
            print("Unknown columns found")
            print(unknown_cols)

        return df

    return wrapper
