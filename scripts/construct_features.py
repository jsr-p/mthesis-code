import click

from dstnx import data_utils, features, fp, log_utils
from dstnx.features import bef, education, parents, siblings

LOGGER = log_utils.get_logger(name=__name__)


@click.command()
@click.option("--start", default=1985, help="Start year")
@click.option("--end", default=2000, help="End year")
def proc_all(start: int, end: int):
    feat_ids = features.FeatureIDs()

    socio_var = features.DSTVariable(
        name="SOCIO",
        variable_ranges=[
            features.DSTVariableRange(1985, 1990, "SOCIO_GL"),
            features.DSTVariableRange(1991, 2005, "SOCIO13"),
        ],
    )
    besk_var = features.DSTVariable(
        name="BESKST",
        variable_ranges=[
            features.DSTVariableRange(1985, 1990, "BESKST"),
            features.DSTVariableRange(1991, 2005, "BESKST13"),
        ],
    )
    id_var = features.DSTVariable.simple(1985, 2005, "PERSON_ID")
    akm_cols = features.DSTColumns([socio_var, besk_var, id_var])

    for year in range(start, end + 1):
        LOGGER.info(f"Constructing features: {year}")
        node_metadata = data_utils.load_reg(f"node_metadata_{year}").drop_duplicates()
        LOGGER.debug(f"#ObsPre ({year}): {node_metadata.shape[0]}")

        # Parents features
        parent_ids = parents.ParentIDs(node_metadata)
        parent_education = parents.ParentEducation(node_metadata, feat_ids, parent_ids)
        parent_akm = parents.ParentsAKM(
            node_metadata, feat_ids, parent_ids, akm_cols, year
        )
        parent_income = parents.ParentIncome(
            feat_ids,
            node_metadata,
            parent_ids,
            start_year=year - 4,
            end_year=year,
            avg=True,
        )
        parent_bef = parents.ParentBEF(node_metadata, feat_ids, parent_ids, year)

        # BEF & Siblings
        bef_features = bef.BEFFeatures(node_metadata, feat_ids, year)
        sibling_feats = siblings.SiblingFeatures(node_metadata, year)

        # Merge features
        full_features = (
            node_metadata.pipe(parent_education.merge)
            .pipe(parent_income.merge)
            .pipe(parent_akm.merge)
            .pipe(parent_bef.merge)
            .pipe(bef_features.merge)
            .pipe(sibling_feats.merge)
            .pipe(education.assign_parents_highest_pria, in_years=True)
            .pipe(education.assign_classmate_parents_edu)
            .drop(["FAMILIE_ID", "MOR_PID", "FAR_PID"], axis=1)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        LOGGER.debug(f"#ObsPost ({year}): {full_features.shape[0]}")

        # Save
        file = fp.REG_DATA / f"features_{year}.gzip.parquet"
        full_features.to_parquet(file)
        LOGGER.info(f"Finished constructing features; file saved to: {str(file)}")
        LOGGER.info(f"#NaNs:\n{full_features.isna().sum(axis=0)}")


def main():
    proc_all()


if __name__ == "__main__":
    main()
