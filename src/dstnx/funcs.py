from functools import partial
from typing import Optional

import pandas as pd

from dstnx.features import ID_COL


def group_avg(df, rel_col):
    return df.groupby("group_id")[rel_col].transform("mean")


def leave_one_out_mean(df: pd.DataFrame, rel_col: str):
    r"""\bar{x}_{-i} := (\bar{x} - x_i / N) * (N / (N - 1))"""
    return (df.group_avg - (df[rel_col] / df["group_count"])) * (
        df["group_count"] / (df["group_count"] - 1)
    )


def leave_one_out(
    df: pd.DataFrame,
    rel_col: str,
    group_id_col: str,
    name: Optional[str] = None,
    id_col=ID_COL,
):
    df = df.assign(
        group_avg=partial(group_avg, rel_col=rel_col),
        group_count=lambda df: df.groupby(group_id_col)[id_col].transform("count"),
    )
    df[name if name else "classmates_avg"] = leave_one_out_mean(df, rel_col)
    return df.drop(["group_avg", "group_count"], axis=1)
