import itertools as it
from functools import partial
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from dstnx import data_utils, features

INVALID_INSTNR = 999999


def create_edge_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs groups of student finishing the same education."""
    nx_cols = ["INSTNR", "ELEV3_VTIL", "UDEL", "AUDD", "AFG_ART"]
    edge_groups = (
        df.query(f"INSTNR != '{INVALID_INSTNR}'")
        .query("AFG_ART == '11'")  # Afsluttende uddannelse
        .groupby(nx_cols)
        .PERSON_ID.apply(set)
        .to_frame("groups")
        .assign(
            group_id=lambda df: np.arange(df.shape[0]),
            group_count=lambda df: df.groups.apply(len),
        )
        .sort_values(by="group_count", ascending=False)
    )
    return edge_groups


def create_nodes(edge_groups: pd.DataFrame) -> pd.DataFrame:
    """Constructs map from each student to its group in `edge_groups`"""
    node_metadata = (
        edge_groups.drop(["group_count"], axis=1)
        .explode("groups")
        .reset_index()
        .rename(columns={"groups": "PERSON_ID"})
        .filter(["PERSON_ID", "group_id", "YEAR"])
    )
    return node_metadata


def edges_from_iterable(iterable: Iterable) -> list[tuple[str, str]]:
    return list(it.combinations(iterable, r=2))


def construct_edge_groups_dict(
    node_metadata: pd.DataFrame, id_map: Optional[dict[str, int]] = None
) -> dict[str]:
    if id_map:
        print(f"#Obs: {node_metadata.shape[0]}")
        node_metadata["PERSON_ID"] = node_metadata["PERSON_ID"].map(id_map)
        print(f"#NaNs: {node_metadata.PERSON_ID.isna().sum()}")
    return (
        node_metadata.dropna(subset=["PERSON_ID"])
        .groupby("group_id")[features.ID_COL]
        .apply(set)
        .apply(edges_from_iterable)
        .to_dict()
    )


def edges_from_df(
    node_metadata: pd.DataFrame,
    with_info: bool = False,
    id_map: Optional[dict[str, int]] = None,
) -> list[tuple[str, str]]:
    edge_dict = construct_edge_groups_dict(node_metadata, id_map)
    if not with_info:
        return [pair for edge_group in edge_dict.values() for pair in edge_group]
    return [
        pair + ({"group": edge_group_num},)
        for edge_group_num in edge_dict
        for pair in edge_dict[edge_group_num]
    ]
