from abc import abstractmethod
import re
from typing import Literal, Optional, overload

import pandas as pd
import polars as pl

from dstnx import data_utils, db, log_utils
from dstnx.features import akm, bef, income, utils as feat_utils
from dstnx.features.utils import DSTColumns, FeatureIDs

LOGGER = log_utils.get_logger(name=__name__)


QUERY_EDU = """
select * from TMPFEATIDS
left join (select * from UDDF202009) A using (PERSON_ID)
where (
    HF_VTIL = TO_DATE('9999-12-31', 'YYYY-MM-DD')
)
"""
QUERY_TIME = """select * from TMPFEATIDS
left join (select * from UDDF202009) A using (PERSON_ID)
where (
    HF_VTIL = TO_DATE('9999-12-31', 'YYYY-MM-DD')
    and HF_VFRA <= TO_DATE('1986-01-01', 'YYYY-MM-DD')
)
"""


disced_map = data_utils.DISCEDMapping(edu_type="audd")


def clean_parent_col(ser: pd.Series):
    return ser.dropna().drop_duplicates().astype(int)


def add_parent_cols(df: pd.DataFrame, cols: list[str], keep_individual: bool = False):
    def _get_parent_cols(col):
        return [f"{col}_{parent}" for parent in ["mor", "far"]]

    all_par_cols = []
    for col in cols:
        if col == "PERSON_ID":
            continue
        par_cols = _get_parent_cols(col)
        df[f"{col}_parent_avg"] = df[par_cols].mean(axis=1)
        all_par_cols.extend(par_cols)
    if not keep_individual:
        return df.drop(all_par_cols, axis=1)
    return df


class ParentIDs:
    def __init__(self, node_metadata: pd.DataFrame):
        self.node_metadata = node_metadata
        self._get_parent_ids()

    def _get_parent_ids(self):
        self.mor_pids = clean_parent_col(self.node_metadata["MOR_PID"])
        self.far_pids = clean_parent_col(self.node_metadata["FAR_PID"])

    def get_ids_tup(self) -> tuple[pd.Series, pd.Series]:
        return self.mor_pids, self.far_pids

    def get_ids(self, parent: str):
        return getattr(self, f"{parent.lower()}_ids")

    @overload
    def get_concat(self, as_df: Literal[False]) -> list[list[int]]:
        ...

    @overload
    def get_concat(self, as_df: Literal[True]) -> pd.DataFrame:
        ...

    def get_concat(self, as_df: bool = False) -> pd.DataFrame | list[list[int]]:
        df = pd.concat((self.mor_pids, self.far_pids)).reset_index(drop=True)
        if as_df:
            return df.to_frame("PERSON_ID")
        return df.values.reshape(-1, 1).tolist()  # type: ignore


class ParentsFeatures:
    def __init__(
        self,
        node_metadata: pd.DataFrame,
        feature_ids: FeatureIDs,
        parent_ids: ParentIDs,
    ) -> None:
        self.node_metadata = node_metadata
        self.feature_ids = feature_ids
        self.parent_ids = parent_ids
        self._set_ids()
        self._get_parent_ids()

    def _get_parent_ids(self):
        self.mor_pids, self.far_pids = self.parent_ids.get_ids_tup()

    def _set_ids(self):
        self.feature_ids.set_ids(self.parent_ids.get_concat(as_df=False))

    def merge(self, node_metadata: pd.DataFrame):
        return node_metadata.merge(
            self.get_parent_feats("mor"), how="left", on="MOR_PID"
        ).merge(self.get_parent_feats("far"), how="left", on="FAR_PID")

    @abstractmethod
    def _get_features(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_parent_feats(self, parent: str) -> pd.DataFrame:
        ...


class ParentEducation(ParentsFeatures):
    def __init__(
        self,
        node_metadata: pd.DataFrame,
        feature_ids: FeatureIDs,
        parent_ids: ParentIDs,
    ) -> None:
        super().__init__(
            node_metadata=node_metadata, feature_ids=feature_ids, parent_ids=parent_ids
        )
        self.features = self._get_features()

    def _get_features(self):
        return (
            self.feature_ids.get_features(QUERY_EDU)
            .drop_duplicates(subset=["PERSON_ID", "HF_VTIL", "HFAUDD"])
            .drop(["AAR"], axis=1)
            .assign(hfaudd_name=lambda df: df.HFAUDD.map(disced_map.get_mapping()))
        )

    def get_parent_feats(self, parent: str):
        rel_cols = ["PERSON_ID", "HFAUDD", "hfaudd_name"]
        return (
            self.features.query(f"PERSON_ID in @self.{parent.lower()}_pids")
            .filter(rel_cols)
            .rename(
                columns={
                    "PERSON_ID": f"{parent.upper()}_PID",
                    "hfaudd_name": f"{parent.lower()}_hfaudd_name",
                    "HFAUDD": f"{parent.upper()}_HFAUDD",
                }
            )
            .reset_index(drop=True)
        )


class ParentIncome(ParentsFeatures):
    def __init__(
        self,
        feature_ids: FeatureIDs,
        node_metadata: pd.DataFrame,
        parent_ids: ParentIDs,
        start_year: int,
        end_year: int,
        inc_cols: Optional[list[str]] = None,
        avg: bool = False,
        keep_individual: bool = False,
    ) -> None:
        super().__init__(
            node_metadata=node_metadata, feature_ids=feature_ids, parent_ids=parent_ids
        )
        self.income_feats = income.IncomeFeatures(
            feature_ids=feature_ids,
            start_year=start_year,
            end_year=end_year,
            inc_cols=inc_cols,
        )
        self.avg = avg
        self.keep_individual = keep_individual

    def get_parent_feats(self, parent: str) -> pd.DataFrame:
        def _rename_cols(parent: str) -> dict[str, str]:
            return {col: f"{col}_{parent}" for col in self.income_feats.inc_cols}

        if not isinstance(self.income_feats.income, pd.DataFrame):
            raise ValueError
        cols = ["PERSON_ID"] + self.income_feats.inc_cols
        return (
            self.income_feats.income.query(
                f"PERSON_ID in @self.parent_ids.{parent.lower()}_pids"
            )
            .filter(cols)
            .rename(columns={"PERSON_ID": f"{parent.upper()}_PID"})
            .rename(columns=_rename_cols(parent))
        )

    def merge(self, node_metadata: pd.DataFrame) -> pd.DataFrame:
        return (
            node_metadata.merge(
                self.get_parent_feats("mor"), how="left", on="MOR_PID"
            ).merge(self.get_parent_feats("far"), how="left", on="FAR_PID")
            # Add average income of parents
            .pipe(
                lambda df: add_parent_cols(
                    df, self.income_feats.merge_cols, self.keep_individual
                )
                if self.avg
                else df
            )
        )


class ParentsAKM(ParentsFeatures):
    def __init__(
        self,
        node_metadata: pd.DataFrame,
        feature_ids: FeatureIDs,
        parent_ids: ParentIDs,
        cols: DSTColumns,
        year: int,
    ) -> None:
        super().__init__(
            node_metadata=node_metadata, feature_ids=feature_ids, parent_ids=parent_ids
        )
        self.cols = cols
        self.year = year
        self.features = self._get_features()

    def _get_features(self):
        cols_str = db.join_cols_sql(self.cols[self.year])
        query = akm.QUERY_AKM.format(cols_str=cols_str, year=self.year)
        return self.feature_ids.get_features(query).rename(
            columns=self.cols.rename_map(year=self.year)
        )

    def get_parent_feats(self, parent: str):
        rel_cols = ["PERSON_ID", "SOCIO", "BESKST"]
        return (
            self.features.query(f"PERSON_ID in @self.{parent.lower()}_pids")
            .filter(rel_cols)
            .rename(
                columns={
                    "PERSON_ID": f"{parent.upper()}_PID",
                    "BESKST": f"{parent.lower()}_BESKST",
                    "SOCIO": f"{parent.upper()}_SOCIO",
                }
            )
            .reset_index(drop=True)
        )


class ParentBEF(ParentsFeatures):
    def __init__(
        self,
        node_metadata: pd.DataFrame,
        feature_ids: FeatureIDs,
        parent_ids: ParentIDs,
        year: int,
    ) -> None:
        self.year = year
        self.bef_feats = bef.BEFFeatures(
            df=parent_ids.get_concat(as_df=True),
            feature_ids=feature_ids,
            year=year,
            cols=[
                "PERSON_ID",
                "FOED_DAG",
            ],
        )
        super().__init__(
            node_metadata=node_metadata, feature_ids=feature_ids, parent_ids=parent_ids
        )
        self.features = self.bef_feats.features

    def get_parent_feats(self, parent: str):
        return (
            self.features.query(f"PERSON_ID in @self.{parent.lower()}_pids")
            .rename(
                columns={
                    "PERSON_ID": f"{parent.upper()}_PID",
                    "FOED_DAG": f"{parent.lower()}_FOED_DAG",
                }
            )
            .reset_index(drop=True)
        )


PARENT_COLS = ["MOR_PID", "FAR_PID"]


def construct_parent_map(df: pd.DataFrame):
    df = df.drop_duplicates(subset=["PERSON_ID", *PARENT_COLS])
    parent_dict = {}
    for person_id, mor_pid, far_pid in zip(
        df["PERSON_ID"], df["MOR_PID"], df["FAR_PID"]
    ):
        parents = tuple()
        if pd.notnull(mor_pid):
            parents += (int(mor_pid),)
        if pd.notnull(far_pid):
            parents += (int(far_pid),)
        parent_dict[person_id] = parents
    return parent_dict


# --------------------- Upbringing parent features --------------------- #


class ParentYouthUpbringing:
    def __init__(
        self,
        node_metadata: pd.DataFrame,
        neighbors: pd.DataFrame,
        parent_feats: list[str],
    ) -> None:
        self.node_metadata = node_metadata
        self.neighbors = neighbors
        self.parent_feats = parent_feats

        # Set ids
        par_ids = ParentIDs(node_metadata)
        self.mor_pids = par_ids.mor_pids
        self.far_pids = par_ids.far_pids

        self.merged_feats = []

    def add_parent_features(self, node_metadata: pd.DataFrame):
        return node_metadata.merge(
            self.get_parent_feats("mor"), how="left", on="MOR_PID"
        ).merge(self.get_parent_feats("far"), how="left", on="FAR_PID")

    def get_parent_feats(self, parent: str):
        """Gets parents features by querying parents from the large table with population."""
        rel_cols = ["PERSON_ID"]
        return (
            self.neighbors.query(f"PERSON_ID in @self.{parent.lower()}_pids")
            .filter(rel_cols + self.parent_feats)
            .rename(
                columns={
                    "PERSON_ID": f"{parent.upper()}_PID",
                }
                | self._rename_dict(parent)
            )
            .reset_index(drop=True)
        )

    def _rename_dict(self, parent: str) -> dict[str, str]:
        _map = {col: f"{parent.lower()}_{col}" for col in self.parent_feats}
        for col in _map.values():
            self.merged_feats.append(col)
        return _map


def parents_imm(par: pl.DataFrame):
    dad_cols = feat_utils.subset_cols(
        par,
        re_match=re.compile(r"far_imm"),
        re_exclude=re.compile(r"nan_fm"),
        exclude_cols=["PERSON_ID"],
    ).columns
    mom_cols = feat_utils.subset_cols(
        par,
        re_match=re.compile(r"mor_imm"),
        re_exclude=re.compile(r"nan_fm"),
        exclude_cols=["PERSON_ID"],
    ).columns
    return par.with_columns(
        [
            par.select(pl.col(dad_cols)).max(axis=1).alias("dad_imm"),
            par.select(pl.col(mom_cols)).max(axis=1).alias("mom_imm"),
        ]
    ).drop(dad_cols + mom_cols)
