from dataclasses import dataclass

import numpy as np
import pandas as pd

from dstnx import fp
from dstnx.utils.ids import DataIds


@dataclass
class SiblingIds:
    nodes: DataIds
    siblings: DataIds
    df: pd.DataFrame


def _mapper(val):
    return "SIBLING_" + val if not ("SIBLING" in val or "PERSON_ID" in val) else val


def intersect_siblings(
    sibling_dtype: str = "full_siblings", start: int = 1985, end: int = 1990
) -> SiblingIds:
    nodes = DataIds(data_type="node_metadata", start=start, end=end)
    siblings = DataIds(data_type=sibling_dtype, start=start, end=end)
    df_nodes = nodes.cat_data().astype({"PERSON_ID": "int64[pyarrow]"})
    sibling_instnr = df_nodes[["PERSON_ID", "INSTNR"]].rename(
        columns={"INSTNR": "SIBLING_INSTNR", "PERSON_ID": "SIBLING_PERSON_ID"}
    )
    df_siblings = (
        siblings.cat_data()
        .drop(["SIBLING_MOR_PID", "SIBLING_FAR_PID"], axis=1)
        .rename(mapper=_mapper, axis=1)
    )
    # Merge nodes on PERSON_ID -> left gives nodes and right gives siblings
    # Next merge on SIBLING_PERSON_ID -> instnr for sibling
    df = (
        df_nodes.merge(df_siblings, on="PERSON_ID", how="inner")
        .merge(sibling_instnr, on="SIBLING_PERSON_ID", how="left")
        .drop_duplicates(
            subset=[
                "INSTNR",
                "PERSON_ID",
                "SIBLING_INSTNR",
                "SIBLING_PERSON_ID",
                "MOR_PID",
                "FAR_PID",
                "SIBLING_FAR_PID",
                "SIBLING_MOR_PID",
            ]
        )
        .pipe(lambda df: df.loc[df.INSTNR == df.SIBLING_INSTNR])
        .reset_index(drop=True)
        .assign(age_diff=lambda df: (df.ALDER - df.SIBLING_ALDER))
    )
    return SiblingIds(nodes, siblings, df)


def matching_parents(df):
    return (df.MOR_PID == df.SIBLING_MOR_PID) & (df.FAR_PID == df.SIBLING_FAR_PID)


def construct_school_by_family(
    sibling_dtype: str = "full_siblings",
    start: int = 1985,
    end: int = 1990,
    suffix: str = "",
):
    sibling_ids = intersect_siblings(sibling_dtype=sibling_dtype, start=start, end=end)
    sibs = sibling_ids.df

    cols = sibs.columns[sibs.columns.str.contains("mor|far", case=False)]
    sibs[cols].isna().sum(axis=0)  # log
    mask = matching_parents(sibs)
    sibs.loc[~mask].FAMILIE_ID.isna().sum()

    fam_cols = ["INSTNR", "MOR_PID", "FAR_PID"]
    sib_cols = ["PERSON_ID", "SIBLING_PERSON_ID", "INSTNR"]
    sibs = sibs.assign(
        # Some obs have FAM_ID = None (and thus parents) in some years
        # but we can find the parents anyway in the later years through
        # the siblings.
        MOR_PID=lambda df: df.MOR_PID.combine_first(df.SIBLING_MOR_PID),
        FAR_PID=lambda df: df.FAR_PID.combine_first(df.SIBLING_FAR_PID),
    ).drop_duplicates(subset=sib_cols)
    family = (
        sibs.filter(["MOR_PID", "FAR_PID"])
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(family=lambda df: np.arange(df.shape[0]))
    )
    school_by_family = (
        sibs.filter(fam_cols)
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(school_by_family=lambda df: np.arange(df.shape[0]))
    )
    cols = ["PERSON_ID", "INSTNR", "school_by_family", "family"]
    school_by_family = (
        sibs.merge(school_by_family, how="left", on=fam_cols)
        .merge(family, how="left", on=["MOR_PID", "FAR_PID"])[cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    school_by_family.to_parquet(fp.REG_DATA / f"school_by_family{suffix}.gzip.parquet")
    return school_by_family


if __name__ == "__main___":
    intersect_siblings()
