import click
import pandas as pd

from dstnx import data_utils, db
from dstnx.data import neighbors
from dstnx.features import geo

# --------------------- Youth geo --------------------- #


def load(nodes: pd.DataFrame, cohort_year: int) -> pd.DataFrame:
    cols = ["PERSON_ID", "ADRESSE_ID", "KOM", "POSTNR", "SOGNEKODE", "BOP_VFRA"]
    geo_data = neighbors.GeoData(cohort_year, load_neighbors=False)
    pnrs = nodes.PERSON_ID.unique()
    addr = geo_data.full_pop.query("PERSON_ID in @pnrs")[cols]
    return addr.loc[addr.groupby("PERSON_ID").BOP_VFRA.idxmax()].reset_index(drop=True)


# --------------------- BEF KOM kode --------------------- #

QUERY_KOM = """
select PERSON_ID, KOM from BEF{year}12
where (
    PERSON_ID in (select * from table(:id))
)
"""


def query_kom(
    dst_db: db.DSTDB,
    node_metadata: pd.DataFrame,
    cohort_year: int,
    max_year_kom: int,
) -> pd.DataFrame:
    ids = node_metadata.PERSON_ID.unique().astype("int").tolist()
    dst_db.create_collection_table(name="id", dtype="integer", values=ids)
    df = dst_db.extract_data(
        QUERY_KOM.format(year=cohort_year + max_year_kom),
        parameters=dict(
            id=dst_db.get_collection_table("id"),
        ),
    )
    return df


@click.command()
@click.option("--suffix", default="", help="Suffix for data files")
@click.option("--max-year-kom", default=15, help="Suffix for data files")
def query_all_kom(suffix: str, max_year_kom: int = 15):
    dst_db = db.DSTDB()
    cohorts = data_utils.load_reg(f"cohorts{suffix}")
    cohort_years = cohorts.cohort.unique()

    kom_datas = []
    for year in sorted(cohort_years):
        kom_datas.append(
            query_kom(
                dst_db=dst_db,
                node_metadata=cohorts.loc[cohorts.cohort == year].drop_duplicates(
                    subset="PERSON_ID"
                )[["PERSON_ID", "cohort"]],
                cohort_year=year,
                max_year_kom=max_year_kom,
            ),
        )
    df = pd.concat(kom_datas).reset_index(drop=True)
    assert df.duplicated(subset="PERSON_ID").sum() == 0
    data_utils.log_save_pq(filename=f"kom{suffix}", df=df)


def get_kom(suffix: str, as_pl: bool):
    return data_utils.load_reg(f"kom{suffix}", as_pl=as_pl)


if __name__ == "__main__":
    query_all_kom()
