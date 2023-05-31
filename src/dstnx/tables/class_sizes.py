from functools import partial

import pandas as pd

import dstnx
from dstnx import data_utils, mappings
from dstnx.features import education
from dstnx.tables import descriptive, interact
from dstnx.utils import tex

GRADES = ["8-9thGrade", "10thGrade", "BoardingSchool", "AllGrades"]
DESC = ["count", "mean", "std", "min", "50%", "90%", "95%", "99%", "max"]


def school_desc(year: str = "All"):
    """Helper function for `school_table`."""
    school = education.get_groups("_class_new", full_info=True, with_counts=True).query(
        "group_count > 2"
    )
    if year == "Post07":
        school = school.query("YEAR > 2007")
    grade10 = (
        school.loc[school.audd_name.isin(mappings.KLASSE_10)]
        .group_count.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_frame("10thGrade")
    )
    grade910 = (
        school.loc[school.audd_name.isin(mappings.KLASSE_89)]
        .group_count.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_frame("8-9thGrade")
    )
    gradees = (
        school.loc[school.audd_name.isin(mappings.EFTERSKOLE)]
        .group_count.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_frame("BoardingSchool")
    )
    grades_all = school.group_count.describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ).to_frame("AllGrades")
    df = (
        pd.concat((grade910, grade10, gradees, grades_all), axis=1)
        .reset_index(names="desc")
        .melt(id_vars="desc", var_name="grade")
        .assign(year=year)
        .assign(
            grade=lambda df: pd.Categorical(df.grade, categories=GRADES, ordered=True),
            desc=lambda df: pd.Categorical(df.desc, categories=DESC, ordered=True),
        )
    )
    return df


def school_table():
    """Constructs the table of class sizes for the paper."""
    table = (
        pd.concat((school_desc(year="All"), school_desc(year="Post07"))).pivot_table(
            index="desc", values="value", columns=["grade", "year"]
        )
        # .rename(columns={"All": r"\textit{All}$", "Post07": r"\textit{Post07}$"})
        .astype(int)
    )
    table = (
        table.droplevel(0, axis=1)
        .pipe(interact.strip_multiindex)
        .rename(descriptive.DESC_MAP, axis=0)
    )
    format_map = {
        (target_col, descriptive.DESC_MAP[desc_col]): int
        for desc_col in DESC
        for target_col in GRADES
    }
    num_idx = 1
    num_cols = len(GRADES)
    num_subcols = 2  # All and post07
    tab = table.style.format(format_map).to_latex(
        multirow_align="l",
        multicol_align="c",
        column_format=num_idx * "l" + num_cols * num_subcols * "c",
    )
    data_utils.log_save_txt(
        filename="desc_class_sizes",
        suffix=".tex",
        text=tex.compose(
            tab,
            tex_fns=[
                tex.add_rules,
                partial(
                    tex.add_column_midrules,
                    add_row_midrule=False,
                    num_idx=1,
                    num_subcols=num_subcols,
                    num_cols=num_cols,
                ),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def ses_class_size():
    """
    Constructs the table of descriptive statistics for the SES variable cond.
    on class sizes
    """
    suffix = "_new"
    ses_cols = ["all_ses_large", "all_ses_q90", "all_ses_q95", "all_ses_q99"]
    new_names = ["All", "q90", "q95", "q99"]
    col_map = dict(zip(ses_cols, new_names))
    data_utils.load_reg(f"cohorts{suffix}").PERSON_ID.unique()
    table = (
        data_utils.load_reg(f"peer_ses{suffix}")
        .query("PERSON_ID in @cohort_ids")
        .melt(id_vars="PERSON_ID")
        .assign(
            variable=lambda df: pd.Categorical(
                df.variable,
                categories=[
                    "all_ses_q90",
                    "all_ses_q95",
                    "all_ses_q99",
                    "all_ses_large",
                ],
                ordered=True,
            ),
        )
        .groupby("variable")
        .value.describe()
        .astype(int)
        .transpose()
        .rename(columns=col_map)
        .rename(descriptive.DESC_MAP, axis=0)
        .pipe(interact.strip_multiindex)
        .style.format({col: int for col in col_map.values()})
        .to_latex(column_format="l" + "c" * len(col_map))
    )
    data_utils.log_save_txt(
        filename="SES_desc_classsize",
        suffix=".tex",
        text=tex.compose(
            table,
            tex_fns=[
                tex.add_rules,
                partial(
                    tex.add_column_midrules,
                    add_row_midrule=False,
                    num_idx=1,
                    num_subcols=1,
                    num_cols=4,
                ),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def main():
    school_table()
    ses_class_size()


if __name__ == "__main__":
    main()
