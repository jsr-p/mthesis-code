import datetime
import operator
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from numba.core import types
from numba.typed import Dict

TIME_COL_TYPE = np.int32
TIME_COL_TYPE_NB = types.int32


def mock_dst_dates(df: pd.DataFrame):
    mask = df.VTIL == pd.Timestamp("2020-12-31")
    idc = np.random.choice(mask[mask.values].index, size=int(mask.sum() * 0.1))
    df.loc[idc, "VTIL"] = datetime.datetime(9999, 12, 31, 0, 0)
    return df


def astype_dict(cols: list[str], dtype: Any):
    if isinstance(dtype, list):
        return dict(zip(cols, dtype))
    return {col: dtype for col in cols}


def cols_to_numpy(df: pd.DataFrame, cols: list[str], dtype: Any = "float64"):
    return df.astype(astype_dict(cols, dtype))


def date_cols_to_int(df: pd.DataFrame, cols: list[str]):
    """Transforms date columns to integer dates as represented by pyarrow."""
    if (df[cols].dtypes == TIME_COL_TYPE).all():
        return df
    return df.astype(astype_dict(cols, "date32[pyarrow]")).astype(
        astype_dict(cols, TIME_COL_TYPE)
    )


def date_cols_to_dates(df: pd.DataFrame, cols: list[str]):
    """Transforms date columns to integer dates as represented by pyarrow."""
    if (df[cols].dtypes == TIME_COL_TYPE).all():
        return df.astype(astype_dict(cols, "date32[pyarrow]"))
    return df


def date_diffs_to_days(date_diffs):
    return pd.Series(date_diffs, dtype="duration[s][pyarrow]")


def cap_time_column(
    df: pd.DataFrame, time_col: str, year: int, upper: bool = True
) -> pd.DataFrame:
    """Cap time column to a given year."""
    if upper:
        compare_fn = operator.le
    else:
        compare_fn = operator.ge
    ts = pd.Timestamp(f"{year}-01-01")
    df[time_col] = df[time_col].where(compare_fn(df[time_col], ts), ts)
    return df


# --------------------- Mappings --------------------- #


def feat_map_type():
    # I -> F
    return Dict.empty(key_type=types.int64, value_type=types.float64)


def addr_to_neigh_type():
    # A -> I
    return Dict.empty(key_type=types.int64, value_type=types.int64[:])


def addr_pid_time_map_type():
    # A x I -> T_1 x T_2
    addr_pid = types.UniTuple(types.int64, 2)
    dates = TIME_COL_TYPE_NB[:, :]
    return Dict.empty(key_type=addr_pid, value_type=dates)


def nb_map(map_type: str):
    if map_type == "feat":
        return feat_map_type()
    if map_type == "addr_to_neigh":
        return addr_to_neigh_type()
    if map_type == "addr_pid_time":
        return addr_pid_time_map_type()
    raise ValueError


def addr_to_neigh_map(
    neighbors: pd.DataFrame,
    adr_col: str = "ADRESSE_ID",
    id_col: str = "PERSON_ID",
    as_nb_dict: bool = False,
):
    addr_to_neigh = defaultdict(set)
    for addr, pid in zip(neighbors[adr_col], neighbors[id_col]):
        addr_to_neigh[addr].add(pid)
    if as_nb_dict:
        _map = addr_to_neigh_type()
        for add in addr_to_neigh:  # Convert to numpy array for numba
            _map[add] = np.array(list(addr_to_neigh[add]), dtype=np.int64)
        return _map
    return dict(addr_to_neigh)


def neigh_addr_time_map(
    neighbors: pd.DataFrame,
    id_col: str,
    adr_col: str,
    time_cols: list[str] = ["VFRA", "VTIL"],
    as_nb_dict: bool = False,
):
    mapping = defaultdict(set)
    col_from, col_to = time_cols
    for addr, neighbor, st, et in zip(
        neighbors[adr_col],
        neighbors[id_col],
        neighbors[col_from],
        neighbors[col_to],
    ):
        mapping[(addr, neighbor)].update([tuple([st, et])])
    if as_nb_dict:
        _map = addr_pid_time_map_type()
        for val in mapping:
            _map[val] = np.array(list(mapping[val]), dtype=TIME_COL_TYPE)
        return _map
    else:
        return dict(mapping)


def feature_map(
    neighbors: pd.DataFrame, feature: str, id_col: str, as_nb_dict: bool = False
):
    # TODO: ensure that id_col has unique feature (i.e. if we have aggregated)
    if as_nb_dict:
        _map = feat_map_type()
        for _id, feat in zip(neighbors[id_col], neighbors[feature]):
            _map[_id] = feat
        return _map
    return dict(zip(neighbors[id_col], neighbors[feature]))
