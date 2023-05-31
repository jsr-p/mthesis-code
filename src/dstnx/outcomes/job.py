import click
import pandas as pd
import polars as pl

from dstnx import data_utils, db, log_utils, mappings
from dstnx.outcomes import utils

LOGGER = log_utils.get_logger(name=__name__)

QUERY = """
select A.PERSON_ID, A.AAR, A.SOCIO13, A.BESKST13, R.DISCO_KODE, R.ARB_HOVED_BRA_DB07
from AKM{year} A
inner join RAS{year} R on A.PERSON_ID = R.PERSON_ID 
where (
    A.PERSON_ID in (select * from table(:id))
)
"""


def extract_jobs(
    dst_db: db.DSTDB,
    node_metadata: pd.DataFrame,
    cohort_year: int,
    max_age_job: int = 25,
) -> pd.DataFrame:
    ids = node_metadata.PERSON_ID.unique().astype("int").tolist()
    dst_db.create_collection_table(name="id", dtype="integer", values=ids)
    df = dst_db.extract_data(
        QUERY.format(year=cohort_year + max_age_job),
        parameters=dict(
            id=dst_db.get_collection_table("id"),
        ),
    )
    LOGGER.debug(f"#Dups job dataframe: {df.PERSON_ID.duplicated().sum()}")
    return df


@click.command()
@click.option("--suffix", default="", help="Suffix for data files")
@click.option("--max-age-job", default=24, help="Cutoff age to consider educations for")
def get_all_post_jobs(suffix: str, max_age_job: int):
    LOGGER.info(f"Getting all post jobs for {suffix}")
    dst_db = db.DSTDB()
    cohorts = data_utils.load_reg(f"cohorts{suffix}")
    cohort_years = cohorts.cohort.unique()

    job_datas = []
    for year in sorted(cohort_years):
        job_datas.append(
            extract_jobs(
                dst_db=dst_db,
                node_metadata=cohorts.loc[cohorts.cohort == year].drop_duplicates(
                    subset="PERSON_ID"
                )[["PERSON_ID", "cohort"]],
                cohort_year=year,
                max_age_job=max_age_job,
            ),
        )

    extra_suffix = utils.outcome_file_name_extra_suffix(max_age_job)
    data_utils.log_save_pq(
        filename=f"outcomes_jobs{suffix}{extra_suffix}",
        df=pd.concat(job_datas).reset_index(drop=True),
        verbose=True,
    )


def get_jobs_df(
    suffix: str, extra_suffix: str = "", as_pl: bool = False
) -> pd.DataFrame | pl.DataFrame:
    """
    Notes:
        - The jobs data contains several observations for each
        individual
        - We groupby and take the max for each relevant category
    """
    jobs = data_utils.load_reg(f"outcomes_jobs{suffix}{extra_suffix}", as_pl=False)
    jobs = jobs.astype({"BESKST13": "int32[pyarrow]"}).pipe(mappings.map_job_cols)
    jobs = (
        jobs.loc[:, ["PERSON_ID", "AAR"] + mappings.JOB_COLS]
        .groupby("PERSON_ID")[mappings.JOB_COLS]
        .max()
        .add_suffix(extra_suffix)
        .reset_index()
    )
    if as_pl:
        return data_utils.pd_to_pl(jobs)
    return jobs


def filter_fortidspension(df: pl.DataFrame) -> pl.DataFrame:
    LOGGER.debug(
        f"Removed #Fortidspension: {df.select(pl.col('fortidspension') == 1).sum().item()}"
    )
    return df.filter(pl.col("fortidspension") != 1)


if __name__ == "__main__":
    get_all_post_jobs()
