import click
import numpy as np
import pandas as pd

import dstnx
from dstnx import data_utils, db, log_utils, mappings, network

LOGGER = log_utils.get_logger(name=__name__)

QUERY_COHORT = """
select * from D222202.BEF{year}12
where (
      BEF{year}12.FOED_DAG >= TO_DATE('{year}-01-01', 'YYYY-MM-DD')
  and
      BEF{year}12.FOED_DAG < TO_DATE('{year_next}-01-01', 'YYYY-MM-DD')
)
"""
QUERY_SCHOOL = """
select kotre.*
FROM D222202.KOTRE2020 kotre
where (
    kotre.PERSON_ID in (select * from table(:pids_cohort))
    and
    kotre.ELEV3_VTIL < TO_DATE('{year_end}-01-01', 'YYYY-MM-DD')
    and 
    kotre.AUDD in (select * from table(:audd_values))
)
"""
QUERY_COHORT_CLASSMATES = """
SELECT kotre.*
FROM (
    SELECT EXTRACT(YEAR FROM ELEV3_VTIL) AS YEAR, kotre.*
    FROM KOTRE2020 kotre
) kotre
INNER JOIN (
    SELECT *
    FROM groups
) groups
ON kotre.AUDD = groups.AUDD
AND kotre.INSTNR = groups.INSTNR 
AND kotre.YEAR = groups.YEAR 
WHERE kotre.PERSON_ID NOT IN (SELECT * FROM table(:pids_cohorts))
"""

QUERY_BEF = """
select ALDER, FOED_DAG, PERSON_ID, FAMILIE_ID, MOR_PID, FAR_PID 
from BEF{year}12
where (
    PERSON_ID in (select * from table(:pids_cohort))
)
"""
NUM_YEARS = 25  # Query all grundskole-educations until age 25


def set_school_class_db(dst_db: db.DSTDB):
    rel_cols = ["Grundskole 7.-9. klasse", "Grundskole 10. klasse"]
    rel_audd = (
        data_utils.DISCEDMapping("audd")
        .df.query(expr="niveau2_titel_name in @rel_cols")
        .dropna(subset=["niveau5_titel_name"])
        .query("niveau5_titel_name != '7. klasse'")
    )
    audd_values = rel_audd.KODE.values.tolist()
    dst_db.create_collection_table(
        name="audd_values", dtype="integer", values=audd_values
    )


def query_cohort(year_born: int, dst_db: db.DSTDB):
    cohort = dst_db.extract_data(
        QUERY_COHORT.format(year=year_born, year_next=year_born + 1)
    )
    LOGGER.info(f"Cohort {year_born=} of shape {cohort.shape=}...")
    if (mask := cohort.PERSON_ID.isna()).any():
        LOGGER.warning(f"BEF query data contains nan ids {mask.sum()=}")
        return cohort.loc[~mask].reset_index(drop=True).astype({"PERSON_ID": np.int64})
    return cohort


def query_school_cohort(cohort: pd.DataFrame, year_born: int, dst_db: db.DSTDB):
    pids_cohort = cohort.PERSON_ID.dropna().unique().tolist()
    dst_db.create_collection_table(
        name="pids_cohort", dtype="integer", values=pids_cohort
    )
    return dst_db.extract_data(
        QUERY_SCHOOL.format(year_end=year_born + NUM_YEARS),
        parameters=dict(
            pids_cohort=dst_db.get_collection_table("pids_cohort"),
            audd_values=dst_db.get_collection_table("audd_values"),
        ),
    )


def extract_classmates(cohorts: pd.DataFrame, dst_db: db.DSTDB):
    pids_cohorts = cohorts.PERSON_ID.dropna().unique().tolist()
    dst_db.create_collection_table(
        name="pids_cohorts", dtype="integer", values=pids_cohorts
    )
    df = dst_db.extract_data(
        QUERY_COHORT_CLASSMATES,
        parameters=dict(
            pids_cohorts=dst_db.get_collection_table("pids_cohorts"),
        ),
    )
    return df


def get_cohort(year_start, year_end, dst_db: db.DSTDB):
    col_mapper = mappings.ColMapper()
    school_dfs = dict()
    cohort_dfs = dict()
    for year_born in range(year_start, year_end + 1):

        def _write_desc(df):
            desc = df.pipe(col_mapper.map_all)
            LOGGER.debug(f"{year_born:-^80}")
            LOGGER.debug(desc.audd_name.value_counts())
            LOGGER.debug(desc.udd_name.value_counts())

        cohort = query_cohort(year_born, dst_db=dst_db)
        school = query_school_cohort(cohort, year_born, dst_db=dst_db)
        _write_desc(school)
        cohort_dfs[year_born] = cohort
        school_dfs[year_born] = school
    all_school = (
        pd.concat(df.assign(cohort=year) for year, df in school_dfs.items())
        .reset_index(drop=True)
        .assign(YEAR=lambda df: df.ELEV3_VTIL.dt.year)
    )
    cohorts = pd.concat(
        cohort.assign(cohort=year) for year, cohort in cohort_dfs.items()
    ).reset_index(drop=True)
    LOGGER.debug(f"Num persons cohort: {cohorts.PERSON_ID.unique().shape}")
    return cohorts, all_school


def fetch_bef_info_nodes(node_metadata: pd.DataFrame, year: int, dst_db: db.DSTDB):
    pids_cohort = node_metadata.PERSON_ID.dropna().unique().tolist()
    dst_db.create_collection_table(
        name="pids_cohort", dtype="integer", values=pids_cohort
    )
    return dst_db.extract_data(
        QUERY_BEF.format(year=year),
        parameters=dict(
            pids_cohort=dst_db.get_collection_table("pids_cohort"),
        ),
    )


def fetch_bef_info_by_years(
    node_metadata: pd.DataFrame, dst_db: db.DSTDB
) -> pd.DataFrame:
    def _get(year):
        # NOTE: Somehow duplicate birth ids get introduced here
        node_metadata_year = node_metadata.query(f"YEAR == {year}")
        return node_metadata_year.merge(
            fetch_bef_info_nodes(
                node_metadata_year, year, dst_db=dst_db
            ).drop_duplicates(subset=["PERSON_ID"]),
            how="left",
            on="PERSON_ID",
        )

    LOGGER.debug("Fetching BEF for nodes...")
    years = sorted(node_metadata.YEAR.unique())
    return pd.concat((_get(year) for year in years))


@click.command()
@click.option("--start", default=1992)
@click.option("--end", default=1996)
@click.option("--suffix", default="")
def construct_groups(start: int, end: int, suffix: str):
    dst_db = db.DSTDB()
    # Get cohorts
    set_school_class_db(dst_db)
    cohorts, all_school = get_cohort(year_start=start, year_end=end, dst_db=dst_db)

    # Get groups from all school observations for the cohort
    group_cols = ["AUDD", "INSTNR", "YEAR"]
    group_cols = (
        all_school.assign(year=lambda df: df.ELEV3_VTIL.dt.year)
        .drop_duplicates(subset=group_cols)[group_cols]
        .reset_index(drop=True)
    )
    dtypes = ["audd NUMBER(16)", "instnr NUMBER(16)", "year NUMBER(16)"]
    name = "groups"
    dst_db.reset_global_tmp_table("groups")
    dst_db.create_global_tmp_table(name=name, dtypes=dtypes)
    dst_db.insert_global_tmp_table(
        statement=f"insert into {name}(audd, instnr, year) values (:1, :2, :3)",
        rows=group_cols.values.tolist(),
    )

    # Extract all classmates from the cohort
    cohort_classmates = extract_classmates(all_school, dst_db=dst_db).assign(
        cohort=99
    )  # 99 for classmate
    assert not cohort_classmates.PERSON_ID.isin(all_school.PERSON_ID).any()

    # Constuct groups
    full_class = pd.concat((all_school, cohort_classmates), axis=0)
    group_cols = ["AUDD", "INSTNR", "YEAR"]
    gp = full_class.groupby(group_cols).PERSON_ID.apply(list)
    LOGGER.debug(gp.apply(len).describe())

    groups = (
        gp.to_frame("groups")
        .reset_index()
        .assign(
            group_id=lambda df: np.arange(df.shape[0]),
            group_count=lambda df: df.groups.apply(len),
        )
        .sort_values(by="group_count", ascending=False)
    )

    # Get node metdata; fetch BEF for classmates and fix year born
    node_metadata = (
        network.create_nodes(groups)
        .pipe(fetch_bef_info_by_years, dst_db=dst_db)
        .assign(year_born=lambda df: df.FOED_DAG.dt.year)
    )
    LOGGER.debug(f"#Born before 1985: {(node_metadata.year_born < 1985).sum()}")
    LOGGER.debug(f"#Born too late: {(node_metadata.year_born > 2003).sum()}")
    LOGGER.debug(
        f"Overview year born: {node_metadata.query('year_born >= 1985').groupby('year_born').PERSON_ID.count()}"
    )
    node_metadata = node_metadata.query("1985 <= year_born <= 2003").reset_index(
        drop=True
    )

    data_utils.log_save_pq(filename=f"edge_group_metadata{suffix}", df=groups)
    data_utils.log_save_pq(filename=f"node_metadata{suffix}", df=node_metadata)
    data_utils.log_save_pq(filename=f"cohorts{suffix}", df=cohorts)
    data_utils.log_save_pq(filename=f"all_school{suffix}", df=all_school)

    # Year born for all individuals in the cohort and their classmates; we use this
    # for querying e.g. neighbors for all
    year_born = node_metadata.drop_duplicates(subset=["PERSON_ID", "year_born"])
    mask = year_born.duplicated(subset=["PERSON_ID"])
    if mask.any():
        LOGGER.debug(
            f"#{mask.sum()}/{year_born.shape=} duplicates with different year born"
        )
        year_born = year_born.loc[
            year_born.groupby("PERSON_ID").year_born.idxmax()
        ].reset_index(drop=True)
        LOGGER.debug(
            f"Removed duplicates by picking last year, shape of df now {year_born.shape=}"
        )
    data_utils.log_save_pq(
        filename=f"year_born{suffix}",
        df=year_born[["PERSON_ID", "year_born", "FOED_DAG"]],
    )


if __name__ == "__main__":
    construct_groups()
