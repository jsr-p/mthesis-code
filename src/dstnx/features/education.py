from tokenize import group
import numpy as np
import pandas as pd

from dstnx import data_utils, edu_outcomes, funcs, mappings
from dstnx.features.utils import ID_COL
from dstnx.outcomes import edu


# Extra feature
def assign_highest_edu(merged: pd.DataFrame) -> pd.DataFrame:
    highest_edu = (
        merged[[ID_COL, *edu_outcomes.REL_CATS_ABV_OC]]
        .melt(id_vars=ID_COL, var_name="udd_cat", value_name="grad")
        .query("grad == 1")
    )
    highest_edu_map = (
        highest_edu.astype({"udd_cat": edu_outcomes.EDU_CATS_ABV_OC_ORD})
        .groupby("PERSON_ID")
        .udd_cat.max()
        .to_dict()
    )
    return merged.assign(
        highest_edu=lambda df: df.PERSON_ID.map(highest_edu_map)
        .fillna(edu_outcomes.NO_EDU)
        .astype(edu_outcomes.EDU_CATS_ABV_OC_ORD),
        highest_edu_num=lambda df: df.highest_edu.cat.codes,
    )


def assign_highest_edus(merged: pd.DataFrame, in_years: bool = False) -> pd.DataFrame:
    """Assign highest education achieved for parents & self.

    The value are categorical ordered & numerical for pria.
    """
    highest_edu = (
        merged[[ID_COL, *edu_outcomes.REL_CATS_ABV_OC]]
        .melt(id_vars=ID_COL, var_name="udd_cat", value_name="grad")
        .query("grad == 1")
    )
    highest_edu_map = (
        highest_edu.astype({"udd_cat": edu_outcomes.EDU_CATS_ABV_OC_ORD})
        .groupby("PERSON_ID")
        .udd_cat.max()
        .to_dict()
    )
    udd_reg = data_utils.UddReg(in_years=in_years)
    audd_pria_map = udd_reg.audd_pria_map
    return (
        merged.assign(
            highest_edu=lambda df: df.PERSON_ID.map(highest_edu_map)
            .fillna(edu_outcomes.NO_EDU)
            .astype(edu_outcomes.EDU_CATS_ABV_OC_ORD)
        )
        .astype(
            {
                "mor_hfaudd_name": edu_outcomes.EDU_CATS_ORD,
                "far_hfaudd_name": edu_outcomes.EDU_CATS_ORD,
            }
        )
        .assign(
            highest_edu_pria_mor=lambda df: df.MOR_HFAUDD.map(audd_pria_map),
            highest_edu_pria_far=lambda df: df.FAR_HFAUDD.map(audd_pria_map),
        )
        .assign(
            highest_edu_parent=lambda df: pd.Series(
                np.where(
                    df.mor_hfaudd_name > df.far_hfaudd_name,
                    df.mor_hfaudd_name,
                    df.far_hfaudd_name,
                )
            ).astype(edu_outcomes.EDU_CATS_ORD),
            highest_edu_audd_parent=lambda df: pd.Series(
                np.where(
                    df.mor_hfaudd_name > df.far_hfaudd_name,
                    df.MOR_HFAUDD,
                    df.FAR_HFAUDD,
                )
            ),
            highest_edu_num_parent=lambda df: df.highest_edu_parent.cat.codes,
            highest_edu_num=lambda df: df.highest_edu.cat.codes,
            highest_edu_num_mor=lambda df: df.mor_hfaudd_name.cat.codes,
            highest_edu_num_far=lambda df: df.far_hfaudd_name.cat.codes,
            highest_edu_pria_parent=lambda df: df.highest_edu_audd_parent.map(
                audd_pria_map
            ),
        )
    )


def assign_highest_pria(df: pd.DataFrame) -> pd.DataFrame:
    """Assign highest education achieved"""
    udd_reg = data_utils.UddReg(in_years=True)
    return df.assign(highest_edu_pria=lambda df: df.HFAUDD.map(udd_reg.audd_pria_map))


def assign_highest_hfaudd_cat(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        cat_audd=lambda df: df.HFAUDD.map(edu.disced_audd_maps.get_mapping()),
    )


def assign_parents_highest_pria(
    df: pd.DataFrame, in_years: bool = False
) -> pd.DataFrame:
    """Assign highest education achieved for parents & self.

    The value are numerical for pria.
    """
    udd_reg = data_utils.UddReg(in_years=in_years)
    audd_pria_map = udd_reg.audd_pria_map
    return df.assign(
        highest_edu_pria_mor=lambda df: df.MOR_HFAUDD.map(audd_pria_map),
        highest_edu_pria_far=lambda df: df.FAR_HFAUDD.map(audd_pria_map),
        highest_edu_parent=lambda df: pd.Series(
            np.where(
                df.highest_edu_pria_mor > df.highest_edu_pria_far,
                df.mor_hfaudd_name,
                df.far_hfaudd_name,
            )
        ),
        highest_edu_audd_parent=lambda df: pd.Series(
            np.where(
                df.mor_hfaudd_name > df.far_hfaudd_name,
                df.MOR_HFAUDD,
                df.FAR_HFAUDD,
            )
        ),
        highest_edu_pria_parent=lambda df: df.highest_edu_audd_parent.map(
            audd_pria_map
        ),
        avg_edu_pria_parent=lambda df: df[
            ["highest_edu_pria_mor", "highest_edu_pria_far"]
        ].mean(axis=1),
    )


def assign_classmate_parents_edu(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(
        funcs.leave_one_out,
        rel_col="highest_edu_pria_parent",
        group_id_col="group_id",
        name="classmate_avg_highest_edu_pria_parents",
    ).pipe(
        funcs.leave_one_out,
        rel_col="avg_edu_pria_parent",
        group_id_col="group_id",
        name="classmate_avg_edu_pria_parents",
    )


DUMMY_COLS = ["klasse_10", "klasse_11", "efterskole"]


def get_groups(
    suffix: str, full_info: bool = False, with_counts: bool = False, **kwargs
):
    """Returns dataframe with groups and variables to be used for peer effects comp."""
    groups = (
        data_utils.load_reg(f"edge_group_metadata{suffix}", **kwargs)
        .pipe(mappings.ColMapper().map_group_ids)
        .query("group_count > 2")
        .reset_index(drop=True)
        .assign(gcgeq2=1, post07=lambda df: (df.YEAR > 2007).astype(int))
        .assign(
            qs=lambda df: pd.qcut(
                df.group_count,
                q=[0, 0.9, 0.95, 0.99, 1],
                labels=["leq90", "leq95", "leq99", "top"],
            )
        )
    )
    group_cols = ["group_id", *mappings.SCHOOL_DUMMIES, "audd_name", "instname"]
    if full_info:
        group_cols.extend(["INSTNR", "YEAR"])
    if with_counts:
        group_cols.extend(["group_count", "gcgeq2", "post07", "qs"])  # Group info
    return groups.pipe(mappings.map_audd_cols)[group_cols]


def school_types(
    suffix: str, full_info: bool = False, with_counts: bool = False, as_pl: bool = False
):
    """Computes the school types attended for each person.

    Args:
        suffix: Suffix for the data files.

    Returns:
        DataFrame with PERSON_ID and school type dummies.
    """
    groups = get_groups(suffix, full_info, with_counts)
    nodes_with_groups = data_utils.load_reg(f"node_metadata{suffix}")[
        ["PERSON_ID", "group_id"]
    ].merge(
        groups,
        how="left",
        on="group_id",
    )
    if (
        as_pl
    ):  # Computation done in pandas above; thus cannot load as_pl from `load_reg`; convert instead
        return data_utils.pd_to_pl(nodes_with_groups)
    return nodes_with_groups


def school_type_dummies(suffix: str):
    return (
        school_types(suffix)
        .groupby("PERSON_ID")[mappings.SCHOOL_DUMMIES]
        .any()
        .astype(int)
        .reset_index()
    )
