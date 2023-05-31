import operator
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import dstnx
from dstnx import data_utils, db, log_utils, models
from dstnx.data import kraf
from dstnx.features import bef, education, geo, income, parents, trans
from dstnx.features.trans import SES_PCA_COLS
from dstnx.queries import geo as geo_query

LOGGER = log_utils.get_logger(name=__name__)

# Features neighbors

YOUTH_FEAT_COLS = [
    "GRADAT1",
    "IE_TYPE",
    "HFAUDD",
    "PERINDKIALT_13",
    "ERHVERVSINDK_13",
    "ALDER",
]

FM_MARK_COLS_1 = [
    "with_parents",
    "with_mom_alone",
    "with_dad_alone",
    "not_lives_with_parents",
]

FM_MARK_COLS = FM_MARK_COLS_1 + [
    "with_mom_new_couple",
    "with_dad_new_couple",
    "nan_fm_mark",
]


YOUTH_AGE_SPAN = 1  # Consider +-1 youth neighbors


def geo_file(year: int):
    return dstnx.fp.REG_DATA / f"neighbors_{year}.parquet.gzip"


def construct_youth_ses(geo_data: "GeoData", lb_age=14, ub_age=18) -> pd.DataFrame:
    neighbors = geo_data.neighbors
    youth = geo_data.load_youth_neighbors(lb_age=lb_age, ub_age=ub_age)
    parents_youth = parents.ParentYouthUpbringing(
        youth, neighbors, parent_feats=["ses"]
    )
    LOGGER.debug(f"Loaded youth with {lb_age=} and {ub_age=}")
    return parents_youth.add_parent_features(
        # Fill nan for those without nan-values for parents
        youth.fillna(dict(zip(["MOR_PID", "FAR_PID"], [-99] * 2)))
    ).assign(ses=lambda df: df[["mor_ses", "far_ses"]].mean(axis=1))


def query_age(df: pd.DataFrame, lb_age: int, ub_age: int):
    return df.query(  # numxpr doesn't work with pyarrow
        f"{lb_age} <= ALDER <= {ub_age}"
    )


def fam_mark(df: pd.DataFrame):
    """Assign columns of who individual lives with.

    See docs:
    https://www.dst.dk/da/Statistik/dokumentation/Times/moduldata-for-befolkning-og-valg/fm-mark
    """
    LOGGER.debug(f"#Nans in FM_MARK column: {df.FM_MARK.isna().sum()}")
    return df.fillna({"FM_MARK": 7}).assign(
        with_parents=lambda df: (df.FM_MARK == 1).astype(int),
        with_mom_new_couple=lambda df: (df.FM_MARK == 2).astype(int),
        with_mom_alone=lambda df: (df.FM_MARK == 3).astype(int),
        with_dad_new_couple=lambda df: (df.FM_MARK == 4).astype(int),
        with_dad_alone=lambda df: (df.FM_MARK == 5).astype(int),
        not_lives_with_parents=lambda df: (df.FM_MARK == 6).astype(int),
        nan_fm_mark=lambda df: (df.FM_MARK == 7).astype(int),
    )


class GeoData:
    def __init__(
        self,
        year: int,
        force: bool = False,
        columns_geo: Optional[list[str]] = None,
        load_neighbors: bool = True,
    ) -> None:
        self.year = year
        self.force = force
        self.columns_geo = columns_geo
        self.kraf = kraf.load_kraf(year)
        self._full_pop = self.load_geo()
        if load_neighbors:
            self._neighbors = self.load_neighbors()

    @property
    def neighbors(self) -> pd.DataFrame:
        return self._neighbors

    @property
    def full_pop(self) -> pd.DataFrame:
        return self._full_pop

    @classmethod
    def from_file(cls, year: int, columns: list[str]):
        df = pd.read_parquet(geo_file(year), dtype_backend="pyarrow", columns=columns)
        obj = cls.__new__(cls)
        obj._full_pop = df
        obj._neighbors = obj.load_neighbors()
        return obj

    def load_geo(self):
        """Loads geodata for the full population."""
        if (file := geo_file(self.year)).exists() and not self.force:
            LOGGER.debug(f"Geo file already exists for {self.year}")
            if self.columns_geo is not None:
                df = pd.read_parquet(
                    file, dtype_backend="pyarrow", columns=self.columns_geo
                )
            else:
                df = pd.read_parquet(file, dtype_backend="pyarrow")
        else:
            self.dst_db = db.DSTDB(database=None, proxy=False)
            LOGGER.debug(f"Fetching neighborhood data from database for {self.year=}")
            df = geo_query.extract(self.dst_db, self.year)
            df.to_parquet(file)
            LOGGER.debug(f"Saved neighborhood data to {file}")
        # Extra other relevant data
        df = df.pipe(self._merge_kraf)
        return df

    def load_neighbors(self) -> pd.DataFrame:
        neighbors = (
            self.full_pop.astype(
                {"ALDER": np.int32}
            )  # numxpr doesn't work with pyarrow
            .query("20 <= ALDER <= 64")
            .pipe(self._adult_trans, fit_pca=True)
        )
        return neighbors

    def load_fullpop_trans(self) -> pd.DataFrame:
        """Load the full population transformed.

        Some parents might not be between 20 and 64; thus
        when we need the parents SES measure they will be excluded.
        Here we transform all but do not recompute e.g. the SES score.
        """
        return self.full_pop.pipe(self._adult_trans, fit_pca=False)

    def _adult_trans(self, df, fit_pca: bool = True):
        def _save_nan_condition(df, col, cond_fn, compare_val):
            barr = df[col].isna()
            LOGGER.debug(f"#Nans {col}: {barr.sum()}")
            return cond_fn(df[col].fillna(0), compare_val).astype(int)

        _arblos_map = {
            "working_grad": partial(
                _save_nan_condition, col="GRADAT1", cond_fn=operator.ge, compare_val=0
            ),
            "socio_grad": partial(
                _save_nan_condition, col="SOCIO", cond_fn=operator.eq, compare_val=210
            ),
        }

        if self.year >= 1991:
            _arblos = _arblos_map["socio_grad"]
        else:
            _arblos = _arblos_map["working_grad"]

        return (
            df.pipe(education.assign_highest_pria)
            .pipe(education.assign_highest_hfaudd_cat)
            .assign(
                arblos=_arblos,
                imm=bef.is_imm,
                inc=lambda df: income.inc_rank(df, inc_col="PERINDKIALT_13"),
                inc_erhv=lambda df: income.inc_rank(df, inc_col="ERHVERVSINDK_13"),
                inc_kont=lambda df: income.inc_rank(df, inc_col="KONTANTHJ"),
                inc_off=lambda df: income.inc_rank(df, inc_col="OFF_OVERFORSEL_13"),
                ses=lambda df: trans.construct_SES(
                    df,
                    cols=SES_PCA_COLS,
                    year=self.year,
                    fit_pca=fit_pca,
                    quantiles=True,
                ),
                highest_gs=lambda df: (df.cat_audd == "Grundskole").astype(int),
                highest_eu=lambda df: (
                    df.cat_audd == "Erhvervsfaglige uddannelser"
                ).astype(int),
            )
        )

    def load_youth_neighbors(self, lb_age: int, ub_age: int) -> pd.DataFrame:
        """Load youth inside age interval"""
        return (
            self.full_pop.astype({"ALDER": np.int32})
            .pipe(query_age, lb_age=lb_age, ub_age=ub_age)
            .pipe(fam_mark)
        )

    def _merge_kraf(self, df) -> pd.DataFrame:
        return (
            df.merge(self.kraf, how="left", on="PERSON_ID")
            # Fill no crimes with 0
            .assign(crimes=lambda df: df.crimes.fillna(0))
        )


def _get_bounds(age_node: int):
    """Gets the interval that we consider other youth for.

    The age is lower bounded by 0.
    """
    return max(0, age_node - YOUTH_AGE_SPAN), age_node + YOUTH_AGE_SPAN


class GeoYears:
    def __init__(self, start, end, current_year: Optional[int] = None):
        self.start = start
        self.end = end
        self.current_year = current_year
        self.geo_years: dict[int, GeoData] = {}
        self.load()

    def load(self):
        for year in tqdm(range(self.start, self.end)):
            self.geo_years[year] = GeoData(year, columns_geo=None)

    def load_period(
        self, start_period: int, end_period: int, neighbor_type: str = "neighbors"
    ):
        if neighbor_type == "neighbors":
            load_fn = self.load_neighbors
        elif neighbor_type == "youth":
            load_fn = self.load_youth
            if not self.current_year:
                raise AssertionError("No node year placed!")
        else:
            raise ValueError
        neighbors = pd.concat(
            load_fn(year)
            for year in tqdm(
                range(start_period, end_period), desc=f"Loading {neighbor_type}"
            )
        ).reset_index(drop=True)
        mask = neighbors.BOP_VFRA > pd.Timestamp(f"{end_period+1}")
        assert mask.sum() == 0
        return neighbors

    def load_youth_interval(self, year: int, min_age: int, max_age: int):
        geo_data = self.geo_years[year]
        return construct_youth_ses(geo_data, lb_age=min_age, ub_age=max_age).pipe(
            self._geo_dropna
        )

    def load_neighbors(self, year: int):
        return self.geo_years[year].neighbors.pipe(self._geo_dropna)

    def load_specific(self, year: int, pnrs: np.ndarray, cols: list[str]):
        return self.geo_years[year].full_pop.query("PERSON_ID in @pnrs")[cols]

    def load_youth(self, year: int):
        geo_data = self.geo_years[year]
        lb, ub = _get_bounds(self.current_year, year)
        return construct_youth_ses(geo_data, lb_age=lb, ub_age=ub).pipe(
            self._geo_dropna
        )

    def _geo_dropna(self, df):
        return df.dropna(subset=geo.addr_col + geo.coord_cols)


def construct_geo(year: int, force: bool = False, geo_only: bool = True):
    LOGGER.info(f"Constructing geo features for: {year}")
    LOGGER.info("Fetching neighbors from database...")
    geo_data = GeoData(year, force=force)

    if geo_only:
        return geo_data

    neighbors = geo_data.neighbors

    LOGGER.info("Constructing model features...")
    full_data = models.load_full_data(year)

    LOGGER.info("Constructing geo features...")
    geo_feats = geo.compute_geo_measures(neighbors)

    coord_cols = ["ETRS89_EAST", "ETRS89_NORTH"]
    geo_cols = ["DDKN_M100", "DDKN_KM1"]
    adresse_data = (
        full_data.merge(
            # Merge own coordinates onto
            geo_data.full_pop[["PERSON_ID", *geo_cols, *coord_cols]],
            how="left",
            on="PERSON_ID",
        )
        .dropna()
        .reset_index(drop=True)
    )
    LOGGER.debug(f"adresse_data:\n{adresse_data.head()}")

    LOGGER.info("Constructing KDTree features")
    cols = ["arblos", "PERINDKIALT_13", "highest_edu_pria", "imm"]
    coord_cols = ["ETRS89_EAST", "ETRS89_NORTH"]
    geo_kd_tree = geo.GeoKDTree(neighbors, coord_cols)
    neighbor_feats = geo_kd_tree.find_all(adresse_data, cols, as_df=True, k=11)
    LOGGER.debug(f"neighbor_feats:\n{neighbor_feats.head()}")

    data_reg = (
        adresse_data.join(neighbor_feats)
        .merge(geo_feats["DDKN_M100"].reset_index(), how="left", on=["DDKN_M100"])
        .merge(geo_feats["DDKN_KM1"].reset_index(), how="left", on=["DDKN_KM1"])
        .pipe(education.assign_highest_edu)
        .pipe(data_utils.filter_categorical, "INSTNR", 10)
        .pipe(data_utils.filter_categorical, "KOM", 10)
        .dropna()
        .reset_index(drop=True)
    )
    LOGGER.debug(f"data_reg:\n{data_reg.head()}")

    data_reg.to_parquet(dstnx.fp.REG_DATA / f"reg_{year}.parquet.gzip")


if __name__ == "__main__":
    construct_geo(1985, geo_only=True)
