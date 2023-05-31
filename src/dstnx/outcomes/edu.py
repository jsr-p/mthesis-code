from dataclasses import dataclass
import click

from dstnx import db, log_utils

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from dstnx import data_utils, db, funcs, log_utils, sql_queries
from dstnx.outcomes import utils

LOGGER = log_utils.get_logger(name=__name__)


# --------------------- CATEGORIES --------------------- #


REL_CATS = [
    "Grundskole",
    "Adgangsgivende uddannelsesforløb",
    "Forberedende uddannelser",
    "Erhvervsfaglige grundforløb",
    "Erhvervsfaglige uddannelser",
    "Gymnasiale uddannelser",
    "Korte videregående uddannelser, KVU",
    "Bacheloruddannelser, BACH",
    "Mellemlange videregående uddannelser, MVU",
    "Lange videregående uddannelser, LVU",
    "Ph.d. og forskeruddannelser",
]
REL_CATS_ABV = [
    "gs",
    "adg",
    "fu",
    "eg",
    "eu",
    "gym",
    "kvu",
    "bach",
    "mvu",
    "lvu",
    "phd",
]
REL_CATS_ABV_OC = [col for col in REL_CATS_ABV if col != "gs"]
REL_CATS_IDX = list(range(len(REL_CATS)))
REL_CATS_TO_IDX = {k: v for k, v in zip(REL_CATS_IDX, REL_CATS)}
REL_CATS_TO_ABV = {k: v for k, v in zip(REL_CATS, REL_CATS_ABV)}

EDU_CATS_ORD = CategoricalDtype(categories=REL_CATS, ordered=True)
EDU_CATS_ABV_ORD = CategoricalDtype(categories=REL_CATS_ABV, ordered=True)

NO_EDU = "noedu"
EDU_CATS_ABV_OC_ORD = CategoricalDtype(
    categories=[
        # We condition on graduating grundskole; thus NO_EDU is no education
        # after graduating grundskole.
        NO_EDU,
        "adg",
        "fu",
        "eg",
        "eu",
        "gym",
        "kvu",
        "bach",
        "mvu",
        "lvu",
        "phd",
    ],
    ordered=True,
)


# --------------------- GLOBALS --------------------- #

LOGGER = log_utils.get_logger(name=__name__)


disced_udd_maps = data_utils.DISCEDMapping(edu_type="udd")
disced_audd_maps = data_utils.DISCEDMapping(edu_type="audd")

audd_pria_map = data_utils.UddReg(in_years=True).audd_pria_map
udd_pria_map = data_utils.UddReg(in_years=True).udd_pria_map

# --------------------- Functions --------------------- #


def set_school_class_db_outcome(dst_db: db.DSTDB):
    """Excludes all grades from grundskole but 9th and 10th."""
    rel_cols = ["Grundskole 7.-9. klasse", "Grundskole 10. klasse"]
    rel_audd = (
        data_utils.DISCEDMapping("audd")
        .df.query(expr="niveau2_titel_name in @rel_cols")
        .dropna(subset=["niveau5_titel_name"])
        .query("niveau4_titel_name not in ['8.klasse', '7. klasse']")
        .query("niveau5_titel_name not in ['7-8 år ivu']")
    )
    audd_values = rel_audd.KODE.values.tolist()
    dst_db.create_collection_table(name="audd", dtype="integer", values=audd_values)
    return dst_db


def construct_edu_overview(df: pd.DataFrame, cohort_year: int) -> pd.DataFrame:
    udd_col = "cat_udd"  # Startende uddannelse
    rel_cols = ["PERSON_ID", udd_col, "grad"]
    gp = (
        df[rel_cols]
        .groupby(["PERSON_ID", udd_col])
        .grad.any()
        .astype(int)
        .to_frame("did_graduate")
        .assign(did_apply=1)
        .reset_index()
    )
    gp_ongoing = (
        df[["PERSON_ID", udd_col, "ongoing"]]
        .groupby(["PERSON_ID", udd_col])
        .ongoing.any()
        .astype(int)
        .to_frame("is_ongoing")
        .reset_index()
    )
    gp = gp.merge(gp_ongoing, how="left", on=["PERSON_ID", udd_col]).fillna(
        {"is_ongoing": 0}
    )
    LOGGER.debug(f"{gp.columns=}")
    outcomes = gp.pivot_table(
        index=["PERSON_ID"],
        columns=[udd_col],
        values=["did_graduate", "did_apply", "is_ongoing"],
        fill_value=0,
    )
    # Log summary stats
    for col in ["did_graduate", "did_apply", "is_ongoing"]:
        summary = outcomes[col].mean(axis=0)
        LOGGER.debug(f"Outcome summary ({col=}, {cohort_year=}):\n{summary}")
    return outcomes


def oc_classmate_col(oc_col: str) -> str:
    """Renames oc col to class mate avg."""
    return f"{oc_col}_classmates_avg"


def assign_classmates_outcome(df: pd.DataFrame, oc_col: str):
    return df.pipe(
        funcs.leave_one_out,
        rel_col=oc_col,
        group_id_col="group_id",
        name=oc_classmate_col(oc_col),
    )


def oc_classmate_cols() -> list[str]:
    return [oc_classmate_col(col) for col in REL_CATS_ABV_OC]


## High school filter
REL_AUDD = [
    "3.g, matematisk-sproglig fælleslinie",
    "Anden dansk studentereksamen",
    "3.g",
    "3.g, matematisk linie una",
    "3.g, matematisk-fysisk linie",
    "3.g, matematisk-naturfaglig linie",
    "3.g, matematisk-samfundsfaglig linie",
    "3.g, matematisk linie",
    "3.g, matematisk-musisk linie",
    "3.g, matematisk forsøgslinie",
    "3.g, sproglig linie una",
    "3.g, nysproglig linie",
    "3.g, klassisksproglig linie",
    "3.g, musiksproglig linie",
    "3.g, sproglig linie",
    "3.g, samfundssproglig linie",
    "3.g, sproglig forsøgslinie",
    "2. hf",
    "3-årig hf",
    "3. hf.",
    "3. hf",
    "2. år, studenterkursus",
    "2. år, studenterkursus matematik-fysik",
    "2. år, studenterkursus matematik-naturfag",
    "2. år, studenterkursus matematik-samfundsfag",
    "2. år, studenterkursus matematik",
    "2. år, studenterkursus nysproglig",
    "2. år, studenterkursus klassisksproglig",
    "2. år, studenterkursus samfundsfag-sproglig",
    "2. år, studenterkursus sproglig",
    "2.år, HHX kursus",
    "3. år, hhx",
    "Htx, 3-årig",
    "3. år, htx",
    "3.g, IB år 2",
    "3. år, anden international studentereksamen",
    "Studentereksamen mv., ivu",
    "Udenlandsk studentereksamen",
]


def disced_hs():
    return disced_audd_maps.df.dropna(subset=["niveau5_titel_name"]).query(
        "niveau1_titel_name == 'Gymnasiale uddannelser'"
    )


def get_hs_values() -> pd.DataFrame:
    return disced_hs().query("niveau5_titel_name in @REL_AUDD").KODE.unique()


def filter_highschool(df: pd.DataFrame):
    """Filters finished high school entries of KOTRE.

    Selects only the entries for the last year of high school.
    """
    hs_all = disced_hs().KODE.unique()
    hs_finish = get_hs_values()
    hs_other = np.setdiff1d(hs_all, hs_finish)

    mask = df.AUDD.isin(hs_other)
    vc = (
        df.loc[mask]
        .AUDD.map(disced_audd_maps.get_mapping(5))
        .value_counts(dropna=False)
    )
    LOGGER.debug(f"Non-finished high-school cats:\n{vc}")

    mask_hs = df.AUDD.isin(hs_finish)
    vc = (
        df.loc[mask_hs]
        .AUDD.map(disced_audd_maps.get_mapping(5))
        .value_counts(dropna=False)
    )
    LOGGER.debug(f"Finished high-school cats:\n{vc}")

    return df.loc[~mask].reset_index(drop=True)


## Data


def _log_not_finished(df: pd.DataFrame, cap_year: pd.Timestamp) -> pd.DataFrame:
    # We cap at the year and consider it as still ongoing even though they might
    # drop out later
    mask = df.ELEV3_VTIL < cap_year
    LOGGER.debug(f"{(~mask).sum()} observations which are still ongoing")
    return df.assign(
        # Need to cap ongoing at 25 years after cohort and replace
        # with 9999 in order to make the fair comparison of still
        # being enrolled in a education at 25
        ELEV3_VTIL=lambda df: df.ELEV3_VTIL.where(mask, cap_year),
        AUDD=lambda df: np.where(df.ELEV3_VTIL < cap_year, df.AUDD, 9999),
    )


def get_all_post_edus(
    dst_db: db.DSTDB,
    node_metadata: pd.DataFrame,
    cohort_year: int,
    max_age_edu: int = 25,
):
    """
    See also:
        https://www.dst.dk/da/TilSalg/Forskningsservice/Dokumentation/hoejkvalitetsvariable/elevregister-3/audd
    """
    # Need to exclude 8th grade and below from the list of
    # completed educations while some people don't finish 9th grade
    set_school_class_db_outcome(dst_db)

    # Ids
    ids = node_metadata.PERSON_ID.unique().astype("int").tolist()
    dst_db.create_collection_table(name="id", dtype="integer", values=ids)
    query = sql_queries.OUTCOME_QUERY.format(end_year=cohort_year + max_age_edu)
    cap_year = pd.Timestamp(f"{cohort_year + max_age_edu}-01-01")
    LOGGER.debug(f"Capping at {cap_year=} for {cohort_year=} with {max_age_edu=}")
    df = (
        dst_db.extract_data(
            query,
            parameters=dict(
                id=dst_db.get_collection_table("id"),
                # Exclude 8th grade
                audd=dst_db.get_collection_table("audd"),
            ),
        )
        .pipe(_log_not_finished, cap_year=cap_year)
        .assign(
            grad=lambda df: (df.AUDD != 0) & (df.AUDD != 9999),
            ongoing=lambda df: df.AUDD == 9999,
            cat_udd=lambda df: df.UDD.map(disced_udd_maps.get_mapping()),
            cat_audd=lambda df: df.AUDD.map(disced_audd_maps.get_mapping()),
            cat_4_audd=lambda df: df.AUDD.map(disced_audd_maps.get_mapping(niveau=5)),
            cat_4_udd=lambda df: df.UDD.map(disced_udd_maps.get_mapping(niveau=5)),
        )
        .pipe(filter_highschool)
    )
    ids_not_in_df = np.setdiff1d(ids, df.PERSON_ID)
    LOGGER.debug(
        f"#IDs for {cohort_year=} with no further educational outcomes in KOTRE: {ids_not_in_df.shape[0]}"
    )
    if ids_not_in_df.size > 0:
        filler = np.zeros(shape=(ids_not_in_df.shape[0], df.shape[1]))
        filler_date = pd.Timestamp(f"{cohort_year}-01-01")
        random_cat = "Gymnasiale uddannelser"  # Filler category for later groupby
        LOGGER.debug(f"Adding random category: {random_cat=}")
        no_outcomes = pd.DataFrame(filler, columns=df.columns.tolist()).assign(
            PERSON_ID=ids_not_in_df,
            ELEV3_VFRA=filler_date,
            ELEV3_VTIL=filler_date,
            cat_audd=random_cat,
            cat_udd=random_cat,
        )
        LOGGER.debug(f"cols of filler df: {no_outcomes.columns}")
        df = pd.concat((df, no_outcomes), axis=0).reset_index(drop=True)
    return df


class EducationOutcomes:
    def __init__(
        self,
        outcomes: pd.DataFrame,
        node_metadata: pd.DataFrame,
    ):
        self.outcomes = outcomes
        self.node_metadata = node_metadata
        self.rel_udd_cols = [col for col in REL_CATS_ABV]

    def construct_outcomes(self, bool_col: str = "did_apply"):
        LOGGER.debug(f"Constructing {bool_col} features...")
        return (
            self.outcomes[bool_col]
            .reset_index()
            .rename(columns=REL_CATS_TO_ABV)
            .pipe(self._assign_upper_secondary_any, bool_col=bool_col)
        )

    def _assign_upper_secondary_any(self, df, bool_col: str):
        """Boolean indicating if any of the three upper secondary is true.

        The abbv. becomes `us` for Upper Secondary.
        """
        cols = ["eu", "gym"]
        df = df.assign(us=df[cols].any(axis=1).astype(int))
        LOGGER.debug(f"Upper secondary (eu, gym) avg for ({bool_col=}):{df.us.mean()=}")
        return df

    def _check_any_cols(self, df):
        return [col for col in self.rel_udd_cols if col not in df.columns]


def highest_pria_grad(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.query("grad == True")
        .assign(audd_pria=lambda df: df.AUDD.map(audd_pria_map))
        .groupby("PERSON_ID")
        .audd_pria.max()
        .to_frame("highest_pria_grad")
        .reset_index()
    )


def highest_pria_apply(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(udd_pria=lambda df: df.UDD.map(udd_pria_map))
        .groupby("PERSON_ID")
        .udd_pria.max()
        .to_frame("highest_pria_apply")
        .reset_index()
    )


def highest_pria_outcomes(df: pd.DataFrame, fill_na: bool = True) -> pd.DataFrame:
    df = pd.merge(
        highest_pria_grad(df), highest_pria_apply(df), how="outer", on="PERSON_ID"
    )
    if fill_na:
        return df.assign(
            # We condition on completing grundskole (10 years); thus all those
            # without any further educational outcomes have highest_pria_grad = 10 years
            highest_pria_grad=lambda df: df.highest_pria_grad.fillna(10),
            highest_pria_apply=lambda df: np.where(
                df.highest_pria_apply < 10, 10, df.highest_pria_apply
            ),
        )
    return df


def outcomes_cohort(dst_db: db.DSTDB, cohort, cohort_year: int, max_age_edu: int):
    df = get_all_post_edus(
        dst_db, cohort, cohort_year=cohort_year, max_age_edu=max_age_edu
    )
    outcomes = construct_edu_overview(df, cohort_year=cohort_year)
    education_outcomes = EducationOutcomes(outcomes, cohort)
    did_apply = education_outcomes.construct_outcomes("did_apply")
    did_graduate = education_outcomes.construct_outcomes("did_graduate")
    is_ongoing = education_outcomes.construct_outcomes("is_ongoing")
    highest_pria = highest_pria_outcomes(df, fill_na=False)
    LOGGER.debug(f"Highest pria: #obs{highest_pria.shape[0]}")
    return did_apply, did_graduate, is_ongoing, highest_pria


@dataclass
class EduData:
    did_apply: pd.DataFrame
    did_graduate: pd.DataFrame
    is_ongoing: pd.DataFrame
    highest_pria: pd.DataFrame


def concat_edus(edu_datas: list[EduData], attribute: str) -> pd.DataFrame:
    return pd.concat(
        [getattr(edu_data, attribute) for edu_data in edu_datas]
    ).reset_index(drop=True)


def construct_outcomes(dst_db: db.DSTDB, suffix: str, max_age_edu: int = 25):
    cohorts = data_utils.load_reg(f"cohorts{suffix}")
    cohort_years = cohorts.cohort.unique()

    edu_datas = []
    for year in cohort_years:
        edu_datas.append(
            EduData(
                *outcomes_cohort(
                    dst_db,
                    cohorts.loc[cohorts.cohort == year].drop_duplicates(
                        subset="PERSON_ID"
                    )[["PERSON_ID", "cohort"]],
                    year,
                    max_age_edu=max_age_edu,
                )
            )
        )

    # Save outcomes
    extra_suffix = utils.outcome_file_name_extra_suffix(max_age_edu)
    data_utils.log_save_pq(
        filename=f"outcomes_apply{suffix}{extra_suffix}",
        df=concat_edus(edu_datas, attribute="did_apply"),
        verbose=True,
    )
    data_utils.log_save_pq(
        filename=f"outcomes_grad{suffix}{extra_suffix}",
        df=concat_edus(edu_datas, attribute="did_graduate"),
        verbose=True,
    )
    data_utils.log_save_pq(
        filename=f"outcomes_ongoing{suffix}{extra_suffix}",
        df=concat_edus(edu_datas, attribute="is_ongoing"),
        verbose=True,
    )
    data_utils.log_save_pq(
        filename=f"outcomes_pria{suffix}{extra_suffix}",
        df=concat_edus(edu_datas, attribute="highest_pria"),
        verbose=True,
    )
    LOGGER.info(f"Finished constructing outcomes for {suffix}")


@click.command()
@click.option("--suffix", default="", help="File suffix")
@click.option("--max-age-edu", default=25, help="Cutoff age to consider educations for")
def proc_all(suffix: str, max_age_edu: int):
    dst_db = db.DSTDB()
    LOGGER.info(f"Constructing outcomes for ({suffix=}, {max_age_edu=})")
    construct_outcomes(dst_db, suffix, max_age_edu)


def main():
    proc_all()


if __name__ == "__main__":
    main()
