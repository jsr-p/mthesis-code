import click
import numpy as np
import pandas as pd

import dstnx
from dstnx import data_utils, db, log_utils, network

LOGGER = log_utils.get_logger(name=__name__)

QUERY = """
SELECT K.PERSON_ID,
       EXTRACT(YEAR FROM REFERENCETID) as REFERENCEAAR,
       EXTRACT(YEAR FROM REFERENCETID)+1 AS YEAR_NEXT,
       K.KLASSEID,
       K.SKL_VFRA, K.UDD, K.INSTNR, K.KL_BETEGNELSE, 
       K.KL_TYPE, K.KLASSETRIN, K.UDEL from xl2.GRUNDSKOLE_KLASSER K
       where (
    PERSON_ID in (select * from table(:pids))
    and K.KLASSETRIN > 7
    )
"""


@click.command()
@click.option("--suffix", default="")
def construct_class_ids(suffix: str):
    LOGGER.info(f"Constructing school class data for {suffix}...")
    nodes = data_utils.load_reg(f"node_metadata{suffix}", as_pl=False)
    metadata = data_utils.load_reg(
        "edge_group_metadata_new", as_pl=False, dtype_backend="numpy_nullable"
    ).astype({"INSTNR": int})

    nodes_m = nodes.merge(
        metadata[
            ["group_id", "INSTNR", "AUDD", "YEAR", "group_count"]
        ].drop_duplicates(),
        how="left",
        on=["group_id", "YEAR"],
    )

    dst_db = db.DSTDB(proxy=False, dsn="DB_PSD.world")
    dst_db.create_collection_table(
        name="pids", dtype="integer", values=nodes.PERSON_ID.unique().tolist()
    )
    school = dst_db.extract_data(
        QUERY,
        parameters=dict(
            pids=dst_db.get_collection_table("pids"),
        ),
    )

    merged = (
        nodes_m.merge(
            school.rename(columns={"YEAR_NEXT": "YEAR"}),
            how="left",
            on=["PERSON_ID", "YEAR", "INSTNR"],
        )
        .assign(
            non_nan_klasseid=lambda df: df.groupby("group_id").KLASSEID.transform(
                lambda _df: _df.notna().sum()
            )
        )
        .assign(diff=lambda df: df.group_count - df.non_nan_klasseid)
    )
    gp_non = merged[merged.non_nan_klasseid != 0]
    gp_nan = merged[merged.non_nan_klasseid == 0]
    LOGGER.debug(f"#{gp_non.shape[0]} obs with class ids")
    LOGGER.debug(f"#{gp_nan.shape[0]} obs without class ids")

    new_groups = gp_non.groupby("group_id").apply(
        lambda df: df.KLASSEID.dropna().unique()
    )

    # Allocate randomly those who have nan class ids
    group_ids = gp_non.loc[gp_non.KLASSEID.isna(), "group_id"]
    randomly_allocated = new_groups.loc[group_ids].apply(np.random.choice).values
    gp_non.loc[gp_non.KLASSEID.isna(), "KLASSEID"] = randomly_allocated
    assert gp_non.KLASSEID.isna().sum() == 0

    new_groups = gp_non.groupby(["group_id", "KLASSEID"]).PERSON_ID.apply(list)

    groups_klasseids = (
        gp_non.groupby(["group_id", "KLASSEID", "YEAR", "INSTNR", "AUDD"])
        .PERSON_ID.apply(list)
        .reset_index()
        .drop(["group_id", "KLASSEID"], axis=1)
        .rename(columns={"PERSON_ID": "groups"})
        .assign(
            group_id=lambda df: np.arange(df.shape[0]),
            group_count=lambda df: df.groups.apply(len),
        )
    )
    # Construct the other groups starting from the highest new
    max_group_id = groups_klasseids.group_id.max()
    groups_other_ids = (
        gp_nan.groupby(["group_id", "YEAR", "INSTNR", "AUDD"])
        .PERSON_ID.apply(list)
        .reset_index()
        .rename(columns={"PERSON_ID": "groups"})
        .drop(["group_id"], axis=1)
        .assign(
            group_id=lambda df: np.arange(
                max_group_id + 1, df.shape[0] + max_group_id + 1
            ),
            group_count=lambda df: df.groups.apply(len),
        )
    )

    all_new = pd.concat((groups_klasseids, groups_other_ids)).reset_index(drop=True)
    data_utils.log_save_pq(filename=f"edge_group_metadata_class{suffix}", df=all_new)
    # New node metadata
    data_utils.log_save_pq(
        network.create_nodes(all_new).merge(
            filename=f"node_metadata_class{suffix}",
            df=nodes.drop(["group_id"], axis=1).drop_duplicates(
                subset=["PERSON_ID", "YEAR"]
            ),
            how="left",
            on=["PERSON_ID", "YEAR"],
        )
    )


if __name__ == "__main__":
    construct_class_ids()
