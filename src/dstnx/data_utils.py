import json
import re
from functools import cached_property, reduce
from lib2to3.pytree import convert
from pathlib import Path
from typing import Any, Optional, overload

import numpy as np
import pandas as pd
import polars as pl
from tabulate import tabulate

import dstnx
from dstnx import fp, log_utils

LOGGER = log_utils.get_logger(name=__name__)


def pd_to_pl(df: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(df)


def filter_df(df: pd.DataFrame, col: str, values: list):
    return df.pipe(lambda df: df.loc[df[col].isin(values)].reset_index(drop=True))


def load_filter_values(name: str, col: str = "value") -> list:
    return pd.read_csv(fp.FILTER_MAPS / f"{name}-filtered.csv")[col].tolist()


def _assign_category_cols(
    disced: pd.DataFrame, col: str = "TITEL", convert_cols: bool = False
) -> pd.DataFrame:
    num_cats = 5
    col_suffix = col.lower()
    for i in reversed(range(1, num_cats + 1)):
        mask = disced.NIVEAU == i
        disced.loc[mask, f"niveau{i}_{col_suffix}_name"] = disced.loc[mask, col]
        disced[f"niveau{i}_{col_suffix}_name"] = disced[
            f"niveau{i}_{col_suffix}_name"
        ].fillna(method="ffill")
        for j in range(i + 1, max(num_cats, i) + 1):
            disced.loc[mask, f"niveau{j}_{col_suffix}_name"] = np.nan
    if convert_cols:
        cols = [f"niveau{i}_{col_suffix}_name" for i in range(1, num_cats + 1)]
        dtypes = ["Int64" for _ in range(num_cats)]
        return disced.astype(dict(zip(cols, dtypes)))
    return disced


def assign_category_cols(
    disced: pd.DataFrame, convert_cols: bool = True
) -> pd.DataFrame:
    return disced.pipe(_assign_category_cols, col="TITEL").pipe(
        _assign_category_cols, col="KODE", convert_cols=convert_cols
    )


DST_FILES = ["audd", "udd", "branche", "disco"]


def get_dst_nested_mapping(table_name: str = "audd", convert_cols: bool = True):
    """See downloaded files in nsdata/other-dst."""
    if table_name not in DST_FILES:
        raise ValueError
    file = fp.DATA / "other-dst" / f"disced_{table_name}.csv"
    rel_cols = ["KODE", "NIVEAU", "TITEL"]
    if table_name == "udd":
        df = pd.read_csv(file, engine="python", encoding="utf-8", sep=";")
    elif table_name == "branche":
        file = fp.DATA / "other-dst" / "Dansk_Branchekode_DB07_v3_2014.csv"
        df = pd.read_csv(file, engine="python", encoding="utf-8", sep=";")
    elif table_name == "disco":
        file = fp.DATA / "other-dst" / "disco.csv"
        df = pd.read_csv(file, engine="python", encoding="utf-8", sep=";")
    else:
        df = pd.read_csv(file, engine="python", encoding="latin", sep="\t")
    return df.filter(rel_cols).pipe(assign_category_cols, convert_cols=convert_cols)


class DSTNestedMapping:
    def __init__(self, table_name: str = "audd", convert_cols: bool = True):
        self.df = get_dst_nested_mapping(table_name, convert_cols=convert_cols)
        self.disced_map = self.df.query("NIVEAU == 5").set_index("KODE")

    def get_mapping(
        self,
        niveau: int = 1,
        col: str = "titel",
        q: Optional[str] = None,
        keys_to_int: bool = False,
        keys_to_str: bool = False,
    ):
        if niveau not in range(1, 5 + 1):
            raise ValueError
        if col not in ["titel", "kode"]:
            raise ValueError
        value_col = f"niveau{niveau}_{col}_name"
        if q:
            _map = self.disced_map.query(q)[value_col].to_dict()
        else:
            _map = self.disced_map[value_col].to_dict()
        if keys_to_int:
            return {int(k): v for k, v in _map.items()}
        if keys_to_str:
            return {str(k): v for k, v in _map.items()}
        return _map

    def get_map_values(self, q: Optional[str] = None):
        """Get values for specific category.

        Notes:
        For e.g. AUDD get values for specific education category.
        See categories here:
            https://www.dst.dk/da/Statistik/dokumentation/nomenklaturer/disced15-audd?id=bc145b5a-c843-473c-ada8-fee6be0fa77a
        For branchekoder see:
            https://www.dst.dk/da/Statistik/dokumentation/nomenklaturer/db07?
        """
        return self.disced_map.query(q).index

    def show_categories(self, niveau: int = 1):
        if niveau == 1:
            return self.df.query(f"NIVEAU == {niveau}").TITEL.tolist()
        cols = [f"niveau{i}_titel_name" for i in range(1, niveau + 1)]
        return self.df.query(f"NIVEAU == {niveau}")[cols]


class DISCEDMapping(DSTNestedMapping):
    def __init__(self, edu_type: str = "audd", convert_cols: bool = True):
        super().__init__(edu_type, convert_cols)


# --------------------- File utils --------------------- #


def load(name: str, folder: Path, as_pl: bool = False, **kwargs) -> pd.DataFrame:
    if as_pl:

        def load_fn(filename: Path):
            return pl.read_parquet(filename)

    else:
        if not kwargs:
            kwargs["dtype_backend"] = "pyarrow"

        def load_fn(filename: Path):
            return pd.read_parquet(filename, **kwargs)

    try:
        return load_fn(folder / f"{name}.gzip.parquet")
    except FileNotFoundError:  # Handle two ways of writing the suffix
        pass
    try:
        return load_fn(folder / f"{name}.parquet.gzip")
    except FileNotFoundError:  # Handle two ways of writing the suffix
        pass
    return load_fn(folder / f"{name}.parquet")


@overload
def load_reg(name: str, fp: Path = fp.REG_DATA, as_pl=True, **kwargs) -> pl.DataFrame:
    ...


@overload
def load_reg(name: str, fp: Path = fp.REG_DATA, as_pl=False, **kwargs) -> pd.DataFrame:
    ...


def load_reg(
    name: str, fp: Path = fp.REG_DATA, as_pl=False, **kwargs
) -> pd.DataFrame | pl.DataFrame:
    return load(name, fp, as_pl=as_pl, **kwargs)


def load_test(name: str) -> pd.DataFrame:
    return load(name, fp.TEST_DATA)


def show_reg_files(glob: str = None) -> list[Path]:
    if glob:
        return sorted(fp.REG_DATA.glob(glob), key=lambda file: file.stem)
    return sorted(fp.REG_DATA.glob("*"), key=lambda file: file.stem)


def load_json(file: Path):
    with open(file, "r") as f:
        return json.load(f)


def save_json(obj: Any, file: Path, indent: int = 1):
    with open(file, "w") as f:
        json.dump(obj, f, indent=indent)


def load_nx_period(
    start: int, end: int, period: int, name: str, method: str
) -> pd.DataFrame:
    return pd.concat(
        (
            load_reg(f"network_measures_{name}_{method}_{st}_{st+period}").add_suffix(
                f"{st}_{st+period}"
            )
            for st in range(start, end, period)
        ),
        axis=1,
    )


def identity_print_df(df: pd.DataFrame) -> pd.DataFrame:
    print(df)
    return df


# --------------------- Feature files --------------------- #


RE_YEAR = re.compile(r"(?P<start>\d{4})-(?P<end>\d{4})")
RE_BATCH = re.compile(r"batch_(?P<num>\d+)")
RE_AGE = re.compile(r"youth(?P<num>\d+)")


def extract_feature_file(file_info: dict):
    col_info = {"neighbors": file_info["neighbors"]}
    if file_info["neighbors"] == "youth":
        col_info["age"] = int(file_info["age"])
    return col_info


def adult_comparer(file):
    return int(RE_BATCH.search(file)["num"])


def youth_comparer(file):
    return (int(RE_AGE.search(file)["num"]), adult_comparer(file))


RE_FEATS = re.compile(
    r"features(?P<suffix>\w+)?_(?P<neighbors>\D+)(?P<age>\d+)?_"
    r"(?P<start>\d+)-(?P<end>\d+)_(?P<dist>\D+)_batch_(?P<num>\d+)"
)


def get_files_features(
    start: int, end: int, suffix: str, neighbors: str, dist: str
) -> tuple[list[Path], list[dict]]:
    """Finds the batch files for a given neighborhood computation.

    Args:
        start: start of period
        end: end of period
        suffix: custom suffix specified when doing the computation
        neighbors: the neighbor type
        dist: k or radius

    Returns:
        tuple of the files and file info
    """
    matches = []
    files = []
    for f in (dstnx.fp.REG_DATA / "batches").glob("*"):
        if match := RE_FEATS.search(f.name):
            files_info = match.groupdict()
            if (
                int(files_info["start"]) == start
                and int(files_info["end"]) == end
                and files_info["dist"] == dist
                and files_info["suffix"] == suffix
                and files_info["neighbors"] == neighbors
            ):
                matches.append(files_info)
                files.append(f)
    batch_nums = np.array([int(m["num"]) for m in matches])
    idc = np.argsort(batch_nums)
    if not (np.diff(batch_nums[idc]) == 1).all():  # Multiple years for each batch
        ages = set([match["age"] for match in matches])
        for age in ages:
            age_batch_nums = np.array(
                [int(m["num"]) for m in matches if m["age"] == age]
            )
            _idc = np.argsort(age_batch_nums)
            assert (np.diff(age_batch_nums[_idc]) == 1).all()
        # Sort by age
        batch_nums = np.array([int(m["age"]) for m in matches])
        idc = np.argsort(batch_nums)
    return [files[i] for i in idc], [matches[i] for i in idc]


def polars_assign(mapping: dict):
    return [pl.lit(v).alias(k) for k, v in mapping.items()]


def drop_pandas_index_col(data: pl.DataFrame):
    return data.drop("__index_level_0__")


def load_features(
    start: int,
    end: int,
    suffix: str,
    neighbors: str,
    dist: str,
    polars: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    files, file_infos = get_files_features(start, end, suffix, neighbors, dist)
    if polars:
        return pl.concat(
            pl.read_parquet(file).with_columns(
                polars_assign(extract_feature_file(file_info))
            )
            for file, file_info in zip(files, file_infos)
        ).pipe(drop_pandas_index_col)
    return pd.concat(
        pd.read_parquet(file).assign(**extract_feature_file(file_info))
        for file, file_info in zip(files, file_infos)
    ).reset_index(drop=True)


def load_reg_period(start: int, end: int, name: str, as_pl: bool):
    dfs = [
        load_reg(f"{name}_{year}-{year+1}", as_pl=as_pl) for year in range(start, end)
    ]
    if as_pl:
        return pl.concat(dfs)
    return pd.concat(dfs)


# --------------------- Log save utils --------------------- #


def extensive_describe(df):
    """Describes dataframe & counts nan values (very inefficient :)"""
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    return pd.concat(
        (df.describe(), pd.Series(df.isna().sum(axis=0), name="#Nans").to_frame().T),
        axis=0,
    )


def log_save_txt(
    filename: str, text: str, fp: Path = dstnx.fp.REG_OUTPUT, suffix: str = ".txt"
):
    if not fp.exists():
        Path.mkdir(fp, exist_ok=True)
    outfile = (fp / filename).with_suffix(suffix)
    with open(outfile, "w") as file:
        file.write(text + "\n")
    LOGGER.info(f"Saved textfile to {outfile}")


def log_save_json_append(
    filename: str,
    obj: Any,
    key: str,
    fp: Path = dstnx.fp.REG_OUTPUT,
):
    if not fp.exists():
        Path.mkdir(fp, exist_ok=True)
    outfile = (fp / filename).with_suffix(".json")
    if outfile.exists():
        _dict = load_json(outfile)
        _dict[key] = obj
    else:
        _dict = dict()
        _dict[key] = obj
    save_json(obj=_dict, file=outfile, indent=4)
    LOGGER.info(f"Saved json to {outfile}")


def log_save_json(
    filename: str,
    obj: dict,
    fp: Path = dstnx.fp.REG_OUTPUT,
):
    if not fp.exists():
        Path.mkdir(fp, exist_ok=True)
    outfile = (fp / filename).with_suffix(".json")
    save_json(obj=obj, file=outfile, indent=4)
    LOGGER.info(f"Saved json to {outfile}")


def log_save_fig(
    filename: str,
    fig,
    fp: Path = dstnx.fp.REG_PLOTS,
):
    if not fp.exists():
        Path.mkdir(fp, exist_ok=True)
    outfile = (fp / filename).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(outfile)
    LOGGER.info(f"Saved figure to {outfile}")


def log_save_tabulate(
    filename: str,
    df: pd.DataFrame,
    fp: Path = dstnx.fp.REG_TABLES,
):
    log_save_txt(
        filename=filename, text=tabulate(df, headers=df.columns), fp=fp  # type: ignore
    )


def log_save_pq(
    filename: str,
    df: pd.DataFrame | pl.DataFrame,
    fp: Path = dstnx.fp.REG_DATA,
    verbose: bool = False,
    describe: bool = False,
):
    if not fp.exists():
        Path.mkdir(fp, exist_ok=True)
    outfile = (fp / filename).with_suffix(".parquet")
    if isinstance(df, pd.DataFrame):
        df.to_parquet(outfile)
    elif isinstance(df, pl.DataFrame):
        df.write_parquet(outfile)
    if verbose:
        LOGGER.info(
            f"Saved dataframe to {filename} of {df.shape=} with columns:\n{df.columns}"
        )
    else:
        LOGGER.info(f"Saved dataframe to {filename}")
    if describe:
        log_save_tabulate(
            f"{filename}_desc",
            fp=dstnx.fp.REG_TABLES / "data_desc",
            df=extensive_describe(df),
        )


def save_array(array: np.ndarray, file_path: Path):
    np.save(file_path, array)


def load_array(filename: str, fp: Path = dstnx.fp.REG_DATA / "edges"):
    return np.load((fp / filename).with_suffix(".npy"))


def log_save_array(
    filename: str,
    array: np.ndarray,
    fp: Path = dstnx.fp.REG_DATA,
    verbose: bool = False,
    logger: Optional[log_utils.logging.Logger] = None,
):
    if not fp.exists():
        Path.mkdir(fp, exist_ok=True)
    if not isinstance(logger, log_utils.logging.Logger):  # For multiprocessing
        logger = LOGGER
    outfile = (fp / filename).with_suffix(".npy")
    save_array(array, outfile)
    if verbose:
        logger.info(f"Saved array to {outfile} with shape {array.shape[0]}")
    else:
        logger.debug(f"Saved array to {outfile} with shape {array.shape[0]}")


# --------------------- Reconstructs features --------------------- #


def construct_features(
    year_ranges: list[tuple[int, int]],
    suffix: str,
    neighbors: str,
    polars: bool = False,
):
    return pd.concat(
        (
            load_features(start, end, suffix, neighbors, polars)
            for start, end in year_ranges
        ),
        axis=0,
    ).reset_index(drop=True)


def year_averages(year_ranges: list[tuple[int, int]], df: pd.DataFrame):
    # TODO: Should use the counts from the averages to compute the measures
    measures = dict()
    for start, end in year_ranges:
        measures[f"{start}-{end}"] = (
            df.query(f"{start} <= age <= {end}").groupby("PERSON_ID").mean()
        )
    return measures


# --------------------- Uddannelsesregister --------------------- #
class UddReg:
    def __init__(self, in_years: bool = False) -> None:
        self.in_years = in_years
        self.data = pd.read_csv(
            dstnx.fp.DATA / "other-dst" / "uddannelsesregistret.csv",
            engine="python",
            encoding="latin",
        )

    @cached_property
    def udd_pria_map(self):
        return self.pria_map("UDD")

    @cached_property
    def audd_pria_map(self):
        return self.pria_map("AUDD")

    def pria_map(self, key: str = "UDD"):
        if key not in ["UDD", "AUDD"]:
            raise ValueError("key must be one of ['UDD', 'AUDD']")
        if self.in_years:
            return self.data.set_index(key)["PRIA"].div(12).to_dict()
        return self.data.set_index(key)["PRIA"].to_dict()

    @property
    def names_overview(self):
        return self.data[["UDD", "AUDD", "TEXT", "ATEXT"]]


# --------------------- Networks utils  --------------------- #


def arrays_union(arrays: list[np.ndarray]):
    def _union(arr1: np.ndarray, arr2: np.ndarray):
        return np.union1d(arr1, arr2)

    return reduce(_union, arrays)


def load_multiple_reg(name: str, start: int, end: int):
    return {year: load_reg(f"{name}_{year}") for year in range(start, end + 1)}


def filter_categorical_values(
    df: pd.DataFrame, col: str, min_count: int = 1
) -> np.ndarray:
    vc = df[col].value_counts()
    return vc[vc > min_count].index.values


def filter_categorical(df: pd.DataFrame, col: str, min_count: int = 1) -> pd.DataFrame:
    values = filter_categorical_values(df, col, min_count)
    return df[df[col].isin(values)].reset_index(drop=True)


def cat_lists(lists: list[list]):
    return reduce(lambda l1, l2: l1 + l2, lists)


def count_values_col_groups(
    dataframe: pd.DataFrame, column_groups: list[list[str]], fn_name: str = "nan"
):
    """
    Counts the number of NaN values in a Pandas DataFrame for a given list of column names.

    Parameters:
    dataframe (Pandas DataFrame): The DataFrame to check for NaN values.
    column_groups (list of lists of strings): A list of groups of column names to count NaN values for.

    Returns:
    A dictionary with the count of NaN values for each column group.
    """

    fn_names = {
        "nan": lambda df: df.isnull().sum().sum(),
        "dup": lambda df: df.duplicated().sum(),
    }
    count_fn = fn_names[fn_name]

    counts = {}

    for i, group in enumerate(column_groups):
        if i > 0:
            group = cat_lists(column_groups[: i + 1])
        count = count_fn(dataframe[group])
        counts[i] = count
        print(f"Col group:\n{group}")
        print(f"\t#{fn_name.capitalize()}Vals: {count}")
    return counts
