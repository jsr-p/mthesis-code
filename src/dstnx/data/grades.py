import click
import polars as pl

from dstnx import data_utils, db
from dstnx.features.agg_neighbors import pd_to_pl

QUERY_GRAN = """
SELECT PERSON_ID, BEDOEMMELSESFORM, FAGDISCIPLIN, INSTNR, KLTRIN, SKOLEAAR, AVG(GRUNDSKOLEKARAKTER) AS AVERAGE_GRADE
FROM UDFK2021
WHERE (
    PERSON_ID IN (SELECT * FROM table(:pids_cohorts))
    )
GROUP BY PERSON_ID, BEDOEMMELSESFORM, FAGDISCIPLIN, INSTNR, KLTRIN, SKOLEAAR
"""

QUERY = """
SELECT PERSON_ID, INSTNR, KLTRIN, SKOLEAAR, AVG(GRUNDSKOLEKARAKTER) AS AVERAGE_GRADE
FROM UDFK2021
WHERE (
    PERSON_ID IN (SELECT * FROM table(:pids_cohorts))
    and  BEDOEMMELSESFORM = 'Afgangsprøve'
    and KLTRIN in (9, 10)
    )
GROUP BY PERSON_ID, INSTNR, KLTRIN, SKOLEAAR
"""


def query_grades(ids: list[int], gran: bool = False):
    if gran:
        query = QUERY_GRAN
    else:
        query = QUERY
    dst_db = db.DSTDB()
    dst_db.create_collection_table(name="pids_cohorts", dtype="integer", values=ids)
    return dst_db.extract_data(
        query,
        parameters=dict(
            pids_cohorts=dst_db.get_collection_table("pids_cohorts"),
        ),
    )


def load_unique(
    suffix: str, as_pl: bool = False, pivot: bool = True, one_avg: bool = False
):
    gpa = (
        data_utils.load_reg(f"grades{suffix}", as_pl=True)
        .groupby(["PERSON_ID", "KLTRIN"])
        .agg(pl.col("AVERAGE_GRADE").mean().alias("gpa"))
        .to_pandas()
    )
    if pivot:
        gpa = (
            gpa.pivot(columns="KLTRIN", index="PERSON_ID", values="gpa")
            .add_prefix("gpa_")
            .reset_index()
            .rename_axis(None, axis="columns")
        )
    if one_avg:
        gpa = gpa.set_index("PERSON_ID").mean(axis=1).to_frame("gpa").reset_index()
    if as_pl:
        return pd_to_pl(gpa)
    return gpa


@click.command()
@click.option("--suffix", default="")
def main(suffix: str):
    nodes = data_utils.load_reg(f"node_metadata{suffix}", as_pl=False)
    data_utils.log_save_pq(
        filename=f"grades_gran{suffix}",
        df=query_grades(nodes.PERSON_ID.unique().tolist(), gran=True),
    )
    data_utils.log_save_pq(
        filename=f"grades{suffix}", df=query_grades(nodes.PERSON_ID.unique().tolist())
    )


if __name__ == "__main__":
    main()
