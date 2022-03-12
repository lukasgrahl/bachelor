from functools import wraps
import datetime as dt

cast_dict = {
    'Call': float,
    'Mean/Average': float,
    'Most Bearish Response': float,
    'Most Bullish Response': float,
    'Put': float,
    'Quart 1 (25% at/below)': float,
    'Quart 2 (median)': float,
    'Quart 3 (25% at/above)': float,
    'S&P 500': float,
    'Total': float,
    'aaii_Bearish': float,
    'aaii_Bull-Bear Spread': float,
    'aaii_Bullish': float,
    'aaii_Bullish 8-week Mov Avg': float,
    'aaii_Bullish Average': float,
    'aaii_Bullish Average +St. Dev.': float,
    'aaii_Bullish Average - St. Dev.': float,
    'aaii_Neutral': float,
    'aaii_Total': float,
    'date': "date",
    "date_week": "date",
    'fears25': float,
    'fears30': float,
    'fears35': float,
    'ff_CMA': float,
    'ff_HML': float,
    'ff_HML_3': float,
    'ff_M_RF': float,
    'ff_M_RF_3': float,
    'ff_RF': float,
    'ff_RF_3': float,
    'ff_RMW': float,
    'ff_SMB': float,
    'ff_SMB_3': float,
    'naaim_ind': float,
    'naaim_std': float,
    'pc_ratio': float,
    'sp_adj_close': float,
    'sp_close': float,
    'sp_High': float,
    'sp_Low': float,
    'sp_Open': float,
    'sp_volume': float,
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
    'vxoo': float,
    'week': str,
    'goog_sent': float,
    'sp_close_lead1': float,
    'sp_close_lag1': float,
    'sp_close_lag2': float,
    'sp_close_lag3': float,
    'sp_close_lag4': float,
    'sp_true_vals': float,
    'sp_agg1': float,
    'date_aaii': 'date',
    'date_goog': 'date',
    'date_naaim': 'date',
    'is_thu': bool,
    'sp_close_lead14': float,
    'sp_close_lead13': float,
    'sp_close_lead12': float,
    'sp_close_lead11': float,
    'sp_close_lead10': float,
    'sp_close_lead9': float,
    'sp_close_lead8': float,
    'sp_close_lead7': float,
    'sp_close_lead6': float,
    'sp_close_lead5': float,
    'sp_close_lead4': float,
    'sp_close_lead3': float,
    'sp_close_lead2': float,
    'sp_close_lead0': float,
    'sp_close_lag5': float,
    'sp_close_lag6': float,
    'sp_close_lag7': float
}


def apply_date_to_week(x):
    # return apply_datetime_format(x).isocalendar()[0:2]
    x = apply_datetime_format(x).isocalendar()[0:2]
    return str(x[0]) + str(x[1])


def apply_datetime_format(x,
                          ret_time: bool = False,
                          europe_time_slash: bool = False):
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
        if europe_time_slash:
            x = dt.datetime.strptime(x, "%d/%m/%Y")
        else:
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

    raise ValueError(f"{x} format unknonw")


def check_singularity_of_values(arr):
    assert (arr.value_counts() > 1).sum() == 0, "Non singular values found"


def check_datetime_sanity(arr,
                          order: str = "past_to_future"):
    """
    Checking order of datetime ordered df
    :param arr: datetime arr
    :param order: "past_to_future", "future_to_past"
    """
    check_singularity_of_values(arr)
    _ = (arr - arr.shift(1)).min().days
    if order == "past_to_future":
        assert _ > 0, f"Datetime col is not order {order}"
    elif order == "future_to_past":
        assert _ < 0, f"Datetime col is not order {order}"
    pass


def cast_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "europe_time_slash" in kwargs.keys():
            europe_time_slash = kwargs["europe_time_slash"]
            kwargs.pop("europe_time_slash")
        else:
            europe_time_slash = False

        df = func(*args, **kwargs)
        assert "file_name" in kwargs.keys(), "Specify filname as kwargs"

        file_name = kwargs["file_name"]
        if file_name.split('.')[-1] in ["pkl"]:
            return df

        unknown_cols = [item for item in df.columns if item not in cast_dict.keys()]
        exceptions = []
        for col in df.columns:
            try:
                if cast_dict[col] == "date":
                    df[col] = df[col].apply(lambda x: apply_datetime_format(x, europe_time_slash=europe_time_slash))
                    check_datetime_sanity(arr=df[col], order="past_to_future")
                else:
                    df[col] = df[col].astype(cast_dict[col])
            except Exception as e:
                exceptions.append(e)
                pass

        if len(unknown_cols) > 0:
            print("Unknown columns found, columns were not casted")
            print(unknown_cols)
        if len(exceptions) > 0:
            print("Exceptions were found")
            print(exceptions)
        return df

    return wrapper


def apply_textmonth_to_nummonth(x):
    month_dict = {
        "Dec": 12,
        "Nov": 11,
        "Oct": 10,
        "Sep": 9,
        "Aug": 8,
        "Jul": 7,
        "Jun": 6,
        "May": 5,
        "Apr": 4,
        "Mar": 3,
        "Feb": 2,
        "Jan": 1
    }

    x = x.split(" ")
    x[0] = int(month_dict[x[0]])
    x[1] = int(x[1][:-1])
    x[2] = int(x[2])

    return dt.datetime(month=x[0], day=x[1], year=x[2])
