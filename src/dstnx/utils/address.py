from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import dstnx
from dstnx import data_utils, db, log_utils
from dstnx.utils import address_comp

LOGGER = log_utils.get_logger(name=__name__)
YOUTH_LENGTH = 18

ADRESSE_QUERY = (
    """
    select distinct B.PERSON_ID, B.ADRESSE_ID, B.ALDER, B.KOM,
    BEFBOP.BOP_VFRA, BEFBOP.BOP_VTIL
    from BEF{year}12 B
    inner join TMPIDS TMP on B.PERSON_ID = TMP.PERSON_ID
    inner join BEFBOP202012 BEFBOP on B.ADRESSE_ID = BEFBOP.ADRESSE_ID 
    AND B.PERSON_ID = BEFBOP.PERSON_ID
    where (
        BEFBOP.BOP_VFRA <= TO_DATE('{year_plus_one}-01-01', 'YYYY-MM-DD')
    """
    + f"""
        and B.ALDER < {YOUTH_LENGTH + 1}
    )
    """
)


def _insert_ids_table(dst_db, unq_ids: np.ndarray):
    name = "tmpids"
    dst_db.reset_global_tmp_table(name=name)
    dtypes = ["person_id NUMBER(16)"]
    dst_db.create_global_tmp_table(name=name, dtypes=dtypes)
    dst_db.insert_global_tmp_table(
        statement=f"insert into {name}(person_id) values (:1)",
        rows=unq_ids.reshape(-1, 1).tolist(),
    )


def _max_year(years: list[int]):
    return min(max(years) + YOUTH_LENGTH, 2020)  # cap at 2020


def get_years(year_born: pd.DataFrame):
    years = np.array(sorted(year_born.year_born.unique().astype(int)))
    years_fetch = range(min(years), _max_year(years))
    LOGGER.debug(f"years:\n{years}\nyears_fetch:{list(years_fetch)}")
    return list(years_fetch)


def construct_address_data(suffix: str) -> pd.DataFrame:
    year_born = data_utils.load_reg(f"year_born{suffix}")
    dst_db = db.DSTDB()
    unq_ids = np.array(df.PERSON_ID.unique())  # pandas 2.0
    LOGGER.debug(
        f"#{unq_ids.shape[0]} ids to get addresses for before age of {YOUTH_LENGTH + 1}"
    )
    _insert_ids_table(dst_db, unq_ids)

    adrs = dict()
    for year in get_years(year_born):  # Get address before start
        LOGGER.info(f"Fetching addresse in {year}...")
        query = ADRESSE_QUERY.format(year=year, year_plus_one=year + 1)
        adrs[year] = dst_db.extract_data(query)

    df = (
        pd.concat(list(adrs.values())).drop_duplicates(
            subset=["PERSON_ID", "ADRESSE_ID", "KOM", "BOP_VFRA", "BOP_VTIL"]
        )
    ).convert_dtypes(dtype_backend="pyarrow")
    mask = df[["BOP_VFRA", "BOP_VTIL"]].isna()
    LOGGER.debug(f"#Nans bop cols: {mask.sum(axis=0)}")
    return df.loc[~mask.any(axis=1)].reset_index(drop=True)


# --------------------- Array utils --------------------- #


def save_arrays(arrays: dict[str, np.ndarray], name: str, fp: Path = dstnx.fp.REG_DATA):
    fname = (fp / name).with_suffix(".npy")
    with open(fname, "wb") as f:
        np.savez(f, **arrays)
    print(f"Saved array to: {fname}")


def load_array(name: str, fp: Path = dstnx.fp.REG_DATA):
    with open((fp / name).with_suffix(".npy"), "rb") as f:
        arrays = np.load(f)
        return {f: arrays[f] for f in arrays.files}


def save_edges(edges: dict[int, np.ndarray], name: str, fp: Path = dstnx.fp.REG_DATA):
    save_arrays(
        arrays={str(node_id): arr for node_id, arr in edges.items()}, name=name, fp=fp
    )


def load_edges(name: str, fp: Path = dstnx.fp.REG_DATA):
    return {
        int(node_id): edges for node_id, edges in load_array(name=name, fp=fp).items()
    }


# --------------------- Time utils --------------------- #


def subset_years_mask(
    start: pd.Timestamp,
    end: pd.Timestamp,
    df: pd.DataFrame,
    col1: str = "BOP_VFRA",
    col2: str = "BOP_VTIL",
) -> pd.Series:
    """Subset by time interval.

    See:
        https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap
    """
    assert (df[col1] <= df[col2]).all()
    assert start <= end
    return (df[col1] < end) & (df[col2] >= start)


def subset_years(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    col1: str = "BOP_VFRA",
    col2: str = "BOP_VTIL",
) -> pd.DataFrame:
    return df.loc[subset_years_mask(start, end, df, col1, col2)]


# --------------------- Array functions --------------------- #


def years_overlap(
    start: np.ndarray,
    end: np.ndarray,
    arr_start: np.ndarray,
    arr_end: np.ndarray,
) -> np.ndarray:
    """Includes the case where intervals overlaps exactly.

    See:
        https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap
    """
    return (arr_start <= end) & (arr_end >= start)


# --------------------- Transformations --------------------- #


def construct_address_map(
    loc_geo: pd.DataFrame, adr_col: str = "ADRESSE_ID"
) -> dict[str, int]:
    """Constructs map from address id to unique number"""
    # TODO: Should be concatenation of nodes and neighbor addresses
    unqs = loc_geo[adr_col].unique()
    return dict(zip(unqs, np.arange(unqs.shape[0])))


def transform_address(df: pd.DataFrame, addr_map: dict[str, int]):
    return df.assign(ADRESSE_ID=lambda df: df.ADRESSE_ID.map(addr_map))


def transform_data_neigh(
    nodes: pd.DataFrame,
    neighbors: pd.DataFrame,
    loc_neighbors: pd.DataFrame,
    feat_cols: Optional[list[str]],
    addr_map: Optional[dict[str, int]] = None,
    coord_cols: list[str] = ["ETRS89_EAST", "ETRS89_NORTH"],
    time_cols: list[str] = ["BOP_VFRA", "BOP_VTIL"],
    id_col: str = "PERSON_ID",
    adr_col: str = "ADRESSE_ID",
):
    LOGGER.info("Transforming data for neighborhood measures")
    if not addr_map:
        addr_map = construct_address_map(loc_neighbors)
    cols_types = address_comp.astype_dict(
        [*coord_cols, adr_col], ["float64", "float64", "int64"]
    )
    LOGGER.info("Transforming data for nodes ...")
    nodes = (
        nodes.pipe(transform_address, addr_map=addr_map)
        .pipe(address_comp.date_cols_to_int, time_cols)
        .astype(cols_types)
        .astype({id_col: np.int64})
    )
    LOGGER.info("Transforming data for neighbors ...")
    neighbors = (
        neighbors.pipe(address_comp.cols_to_numpy, feat_cols, dtype="float64")
        .pipe(transform_address, addr_map=addr_map)
        .pipe(address_comp.date_cols_to_int, time_cols)
        .astype(cols_types)
        .astype({id_col: np.int64})
    )
    LOGGER.info("Transforming location data ...")
    loc_neighbors = loc_neighbors.pipe(transform_address, addr_map=addr_map).astype(
        cols_types
    )
    return nodes, neighbors, loc_neighbors


# --------------------- Other --------------------- #


def describe_address_data(df: pd.DataFrame) -> None:
    gp = df.groupby("PERSON_ID").apply(lambda val: val.KOM.unique().shape[0])
    gp_adr = df.groupby("PERSON_ID").apply(lambda val: val.ADRESSE_ID.unique().shape[0])
    print(df.groupby("PERSON_ID").ADRESSE_ID.count().value_counts())
    print(df.groupby("PERSON_ID").KOM.count().value_counts())
    print(gp.value_counts())
    print(gp_adr.value_counts())
    ax = df["diff"].hist()
    ax.set(xlabel="years")
    print(df["diff"].describe())


def check_no_overlap(df1, df2, cols):
    """
    Checks that none of the columns in 'cols' of DataFrame df1 contains
    values that are also in the corresponding columns of DataFrame df2.

    Returns:
        pandas.Series: Boolean mask where True indicates no overlap, False otherwise
    """
    overlap_mask = pd.Series(True, index=df1.index)
    for col in cols:
        overlap_mask &= ~df1[col].isin(df2[col])
    return overlap_mask


ID_COLS = ["ADRESSE_ID", "BOPIKOM", "OPGIKOM", "KOM"]
GRID_COLS = ["DDKN_M100", "DDKN_KM1", "DDKN_KM10"]
COORD_COLS = ["ETRS89_EAST", "ETRS89_NORTH"]


def process_duplicates(
    df: pd.DataFrame, id_cols: list[str], grid_cols: list[str], coord_cols: list[str]
) -> pd.DataFrame:
    """Fix duplicate entries in geodata based on the id_cols.

    Those of the duplicates that have duplicates in grid_cols are averaged
    over the coord_cols.
    These others are randomly picked.
    """
    # Find all duplicated entries based on id_cols
    duplicates = df.duplicated(subset=id_cols, keep=False)
    print(f"#{duplicates.sum()} duplicated ids in geotable!")

    # Subset the duplicated entries
    duplicated_df = df[duplicates].copy()

    # Find the duplicated entries that also have duplicates in grid_cols
    duplicated_with_grid_duplicates = duplicated_df.duplicated(
        subset=id_cols + grid_cols, keep=False
    )

    # Group by on id_cols and grid_cols for the duplicates that have grid duplicates
    group_cols = id_cols + grid_cols
    grouped_duplicates = (
        duplicated_df[duplicated_with_grid_duplicates]
        .groupby(group_cols)[coord_cols]
        .mean()
        .reset_index()
    )

    # Pick a random entry for the duplicates that do not have grid duplicates
    mask_remaining_ids = check_no_overlap(duplicated_df, grouped_duplicates, id_cols)
    random_duplicates = (
        duplicated_df[(~duplicated_with_grid_duplicates) & mask_remaining_ids]
        .groupby(id_cols)
        .sample(n=1)
    )

    # Concatenate the grouped and random duplicates
    result = pd.concat([grouped_duplicates, random_duplicates], axis=0)

    # Append the non-duplicated entries
    non_duplicates = df[~duplicates]
    result = pd.concat([result, non_duplicates], axis=0)

    return result
