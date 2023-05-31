import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from dstnx import db, log_utils

LOGGER = log_utils.get_logger(name=__name__)


ID_COL = "PERSON_ID"


class FeatureIDs:
    """Class for handling temporary IDS to query features for.

    Attributes:
        dst_db:
    """

    def __init__(self, dst_db: Optional[db.DSTDB] = None):
        self.dst_db = dst_db if dst_db else db.DSTDB()

    def set_ids(self, ids: list[list[int]]):
        name = "tmpfeatids"
        self.dst_db.reset_global_tmp_table(name=name)
        dtypes = ["person_id NUMBER(16)"]
        self.dst_db.create_global_tmp_table(name=name, dtypes=dtypes)
        self.dst_db.insert_global_tmp_table(
            statement=f"insert into {name}(person_id) values (:1)",
            rows=ids,
        )
        # Sanity check
        assert (
            self.get_features("select * from tmpfeatids") == np.array(ids)
        ).PERSON_ID.all()

    def get_features(self, query: str) -> pd.DataFrame:
        return self.dst_db.extract_data(query)


@dataclass
class DSTVariableRange:
    year_start: int
    year_end: int
    col: str


class DSTVariable:
    def __init__(self, name: str, variable_ranges: list[DSTVariableRange]) -> None:
        self.name = name
        self._construct_map(variable_ranges)

    def _construct_map(self, variable_ranges: list[DSTVariableRange]):
        self._map = dict()
        for variable_range in variable_ranges:
            for year in range(variable_range.year_start, variable_range.year_end + 1):
                self._map[year] = variable_range.col

    def __getitem__(self, year: int) -> str:
        return self._map[year]

    @classmethod
    def simple(cls, year_start: int, year_end: int, col: str) -> "DSTVariable":
        return DSTVariable(
            name=col, variable_ranges=[DSTVariableRange(year_start, year_end, col)]
        )


@dataclass
class DSTColumns:
    cols: list[DSTVariable]

    def __getitem__(self, year: int) -> list[str]:
        return [dst_col[year] for dst_col in self.cols]

    def __len__(self):
        return len(self.cols)

    def rename_map(self, year: int):
        return {dst_col[year]: dst_col.name for dst_col in self.cols}


def check_cols_exist(
    dst_db: db.DSTDB, year_start: int, year_end: int, cols: DSTColumns, query: str
):
    for year in range(year_start, year_end):
        _cols = cols[year]
        len_cols = len(_cols)
        set_cols = set(_cols)
        table_cols = dst_db.extract_data(query.format(year=year)).columns
        if not len(set(table_cols) & set_cols) == len_cols:
            LOGGER.debug(f"Error in year {year}:")
            LOGGER.debug(f"Diff1:\n{set_cols - set(table_cols)}")
            LOGGER.debug(f"Diff2:\n{set(table_cols) - set_cols}")
            return False
    return True


def subset_cols_l(
    cols: list[str],
    re_match: re.Pattern,
    exclude_cols: list[str] = [],
    include_cols: list[str] = [],
    re_exclude: Optional[re.Pattern] = None,
):
    found_cols = [
        col
        for col in cols
        if (re_match.search(col) or col in include_cols) and col not in exclude_cols
    ]
    if re_exclude:
        found_cols = [col for col in found_cols if not re_exclude.search(col)]
    return found_cols


def subset_cols(
    df,
    re_match: re.Pattern,
    exclude_cols: list[str] = [],
    include_cols: list[str] = [],
    re_exclude: Optional[re.Pattern] = None,
):
    cols = subset_cols_l(df.columns, re_match, exclude_cols, include_cols, re_exclude)
    if isinstance(df, pd.DataFrame):
        return df[cols]
    return df.select(cols)


# --------------------- Subset different columns --------------------- #


RE_SES_VARS = re.compile(r"(inc_avg|inc_kont|crimes|highest_edu_pria|arblos)")
RE_ADULT_OTHER = re.compile(r"highest_(eu|gs)")
RE_SES = re.compile(r"(ses)")


def filter_cols(cols, regex):
    return [col for col in cols if regex.search(col)]


def cols_picker(
    cols: dict[str, list[str]],
    case: str,
    exclude_family: bool = False,
    exclude_adults: bool = False,
    exclude_youth: bool = False,
):
    match case:
        case "reduced":
            adults_ses = [col for col in cols["adult"] if RE_SES.search(col)]
            adults_other = [col for col in cols["adult"] if RE_ADULT_OTHER.search(col)]
            adult = adults_ses + adults_other
            parent = [col for col in cols["parent"] if RE_SES.search(col)]
        case "all":
            adults_ses = [col for col in cols["adult"] if RE_SES_VARS.search(col)]
            adults_other = [col for col in cols["adult"] if RE_ADULT_OTHER.search(col)]
            adult = adults_ses + adults_other
            parent = [col for col in cols["parent"] if RE_SES_VARS.search(col)]
        case _:
            raise ValueError(f"Case {case} not supported")
    out_cols = []
    if not exclude_family:
        out_cols.extend(parent)
        out_cols.extend(cols["familystatus"])
    if not exclude_adults:
        out_cols.extend(adult)
    if not exclude_youth:
        out_cols.extend(cols["youth"])
    return out_cols
