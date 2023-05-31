from typing import Optional

import pandas as pd
import polars as pl

from dstnx import db, log_utils
from dstnx.features.utils import FeatureIDs

LOGGER = log_utils.get_logger(name=__name__)


ID_COL = "PERSON_ID"
BEF_FEATS_CHILD = [
    "BOP_VFRA",
    "FAMILIE_TYPE",
    "FOED_DAG",
    "FM_MARK",
    "HUSTYPE",
    "IE_TYPE",
    "KOEN",
    "KOM",
    "OPR_LAND",
    "PLADS",
    "STATSB",
    "PERSON_ID",
]

QUERY_BEF = """
select * from TMPFEATIDS
left join (select {cols_str} from BEF{year}12) A using (PERSON_ID)
"""


class BEFFeatures:
    """Constructs features from the BEF table."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_ids: FeatureIDs,
        year: int,
        cols: Optional[list[str]] = None,
    ) -> None:
        self.df = df
        self.feature_ids = feature_ids
        self.year = year
        self.cols = cols
        self._set_ids()
        self.features = self._get_features()

    def _set_ids(self):
        self.feature_ids.set_ids(self.df["PERSON_ID"].values.reshape(-1, 1).tolist())

    def merge(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(self.features, how="left", on="PERSON_ID")

    def _get_features(self) -> pd.DataFrame:
        if not self.cols:
            self.cols = BEF_FEATS_CHILD
        cols_str = db.join_cols_sql(self.cols)
        query = QUERY_BEF.format(cols_str=cols_str, year=self.year)
        return self.feature_ids.get_features(query)


def is_desc(df: pd.DataFrame) -> pd.Series:
    return (df.IE_TYPE == 3).astype(int)


def is_imm(df: pd.DataFrame) -> pd.Series:
    return ((df.IE_TYPE == 2) | (df.IE_TYPE == 3)).astype(int)


def assign_imm(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(imm=is_imm)


def assign_imm_pl(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("IE_TYPE") == 2) | (pl.col("IE_TYPE") == 3)).cast(pl.Int8).alias("imm")
    ).drop("IE_TYPE")


def assign_koen(df: pl.DataFrame) -> pl.DataFrame:
    """Assigns gender column.

    https://www.dst.dk/da/TilSalg/Forskningsservice/Dokumentation/hoejkvalitetsvariable/folketal/koen
    """
    return df.with_columns((pl.col("KOEN") == 2).cast(pl.Int8).alias("female")).drop(
        "KOEN"
    )
