from functools import partial
from tokenize import group

import numpy as np
import pandas as pd

import dstnx
from dstnx import data_utils
from dstnx.models import model_prep
from dstnx.plots import sesq
from dstnx.tables import interact
from dstnx.utils import results, tex

TARGET_MAP_SPEC = {
    "eu_grad": r"$\texttt{VocGrad}$",
    "gym_grad": r"$\texttt{HSGrad}$",
    "us_grad": r"$\texttt{USGrad}$",
    "eu_apply": r"$\texttt{VocApply}$",
    "gym_apply": r"$\texttt{HSApply}$",
    "us_apply": r"$\texttt{USApply}$",
    "real_neet": r"$\texttt{NoUSJob}$",
    "eu_grad_y20": r"$\texttt{VocGrad}$",
    "gym_grad_y20": r"$\texttt{HSGrad}$",
    "us_grad_y20": r"$\texttt{USGrad}$",
    "eu_apply_y20": r"$\texttt{VocApply}$",
    "gym_apply_y20": r"$\texttt{HSApply}$",
    "us_apply_y20": r"$\texttt{USGrad}$",
    "real_neet_y20": r"$\texttt{NoUSJob}$",
    "own_ses": "SES",
    "own_ses_14_17": "SES",
    "ses_1417": r"SES^{HS}",
}

TARGET_NONTT_MAP = {
    "eu_grad": "VocGrad",
    "gym_grad": "HSGrad",
    "us_grad": "USGrad",
    "eu_apply": "VocApply",
    "gym_apply": "HSApply",
    "us_apply": "USApply",
    "real_neet": "NoUSJob",
    "eu_grad_y20": "VocGrad",
    "gym_grad_y20": "HSGrad",
    "us_grad_y20": "USGrad",
    "eu_apply_y20": "VocApply",
    "gym_apply_y20": "HSApply",
    "us_apply_y20": "USGrad",
    "real_neet_y20": "NoUSJob",
}
_DESC_MAP = {
    "count": "N",
    "mean": "Mean",
    "std": "SD",
    "min": "Min",
    "max": "Max",
    "25%": "1. Quartile",
    "50%": "Median",
    "75%": "3. Quartile",
}
DESC_MAP = _DESC_MAP | {
    "75%": "3. Quartile",
    "90%": "q90",
    "95%": "q95",
    "99%": "q99",
}


def create_outcome_desc(data, age: str = "25"):
    if age == "25":
        suffix = ""
    elif age == "20":
        suffix = "_y20"
    else:
        raise ValueError
    gp = (
        data.groupby("cohort")
        .agg(
            {
                f"gym_grad{suffix}": "mean",
                f"gym_apply{suffix}": "mean",
                f"eu_grad{suffix}": "mean",
                f"eu_apply{suffix}": "mean",
                f"us_grad{suffix}": "mean",
                f"us_apply{suffix}": "mean",
                f"real_neet{suffix}": "mean",
            }
        )
        .mul(100)
        .astype(int)
        .pipe(interact.strip_multiindex)
    )
    n_map = data.groupby("cohort").PERSON_ID.count().to_dict()
    tab = (
        gp.rename(
            columns=TARGET_MAP_SPEC,
            index={k: f"{k} (N = {val})" for k, val in n_map.items()},
        )
        .style.format()
        .to_latex(
            column_format="l" + len(TARGET_MAP_SPEC) * "c",
        )
    )

    data_utils.log_save_txt(
        filename=f"general-descriptive{suffix}",
        suffix=".tex",
        text=tex.add_rules(tab),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def _id(df):
    print(df)
    return df


def multiply_cols(df, columns, factor):
    for col in columns:
        df[col] = df[col] * factor
    return df


def aux_desc(data: pd.DataFrame):
    variables = ["imm", "female", "klasse_10", "efterskole", "any_psyk", "own_crimes"]
    new_cols = ["Imm.", "Female", "10thGrade", "BoardingSchool", "Psych", "Crimes"]
    tab = (
        data.assign(own_crimes=lambda df: (df.own_crimes > 1).astype(int))[
            variables + ["cohort"]
        ]
        .rename(columns=dict(zip(variables, new_cols)))
        .groupby("cohort")
        .agg({var: "mean" for var in new_cols})
        .mul(100)
        .astype(int)
        .pipe(interact.strip_multiindex)
        .rename(columns=DESC_MAP)
    )

    n_map = data.groupby("cohort").PERSON_ID.count().to_dict()
    tab = (
        tab.rename(
            index={k: f"{k} (N = {val})" for k, val in n_map.items()},
        )
        .style.format()
        .to_latex(
            column_format="l" + len(TARGET_MAP_SPEC) * "c",
        )
    )

    data_utils.log_save_txt(
        filename=f"aux-descriptive",
        suffix=".tex",
        text=tex.compose(tab, [tex.add_rules]),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def _format_describe(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(interact.strip_multiindex).rename(columns=DESC_MAP)


def create_sesq_desc(
    data, group_ses_col="all_ses_small", suffix="small", group_q: str = "SES_q"
):
    """Creates table of SES desc. cond. on quantile groups."""
    feats = ["all_ses_small", "all_ses_large", "SES_q", "SES_q10"]
    tab = (
        data[feats]
        .pipe(model_prep.convert_non_nullable)
        .dropna()
        .assign(
            SES_q=lambda df: df.SES_q.map(sesq.SESQ_MAP),
            SES_q10=lambda df: df.SES_q10.map(sesq.SESQ_MAP),
        )
        .groupby(group_q)[group_ses_col]
        .describe()
        .pipe(_format_describe)
        .style.format({col: int for col in DESC_MAP.values()})
        .to_latex(
            column_format="l" + len(DESC_MAP) * "c",
        )
    )
    data_utils.log_save_txt(
        filename=f"sesq-descriptive-{suffix}-{group_q}",
        suffix=".tex",
        text=tex.compose(
            tab,
            [tex.add_rules]
            # partial(tex.add_column_midrules, add_row_midrule=False, num_idx=1, num_subcols=1, num_cols=len(DESC_MAP))]
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def desc_ses():
    cols = ["ses_1417", "own_ses"]
    descs = ["count", "mean", "std", "25%", "50%", "75%"]
    desc = (
        data_utils.load_reg("full_new_k30")
        .assign(
            ses_1417=lambda df: (df.own_mor_ses_14_17 + df.own_far_ses_14_17) / 2 + 1,
            own_ses=lambda df: df.own_ses + 1,
        )
        .groupby("cohort")[cols]
        .describe()
    )
    desc.columns = desc.columns.swaplevel(0, 1)
    desc.sort_index(axis=1, level=0, inplace=True)
    desc = desc[descs]
    format_map = {
        (DESC_MAP[desc_col], TARGET_MAP_SPEC[target_col]): int
        for desc_col in descs
        for target_col in cols
    }
    num_idx = 1
    num_cols = len(descs)
    num_subcols = len(cols)
    tab = (
        desc.pipe(_format_describe)
        .rename(columns=DESC_MAP)
        .rename(columns=TARGET_MAP_SPEC)
        .style.format(format_map)
        .to_latex(
            multirow_align="l",
            multicol_align="c",
            column_format=num_idx * "l" + num_cols * num_subcols * "c",
        )
    )
    data_utils.log_save_txt(
        filename=f"desc_ses_full_and_hs",
        suffix=".tex",
        text=tex.compose(
            tab,
            tex_fns=[
                tex.add_rules,
                partial(
                    tex.add_column_midrules,
                    add_row_midrule=False,
                    num_idx=1,
                    num_subcols=2,
                    num_cols=6,
                ),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def main():
    data = data_utils.load_reg("full_new_k50_all_all", dtype_backend="numpy_nullable")
    # Create outcome descriptive tables
    create_outcome_desc(data, age="25")
    create_outcome_desc(data, age="20")

    # Create class SES desc
    create_sesq_desc(data, group_ses_col="all_ses_small", suffix="small")
    create_sesq_desc(data, group_ses_col="all_ses_large", suffix="large")
    create_sesq_desc(
        data, group_ses_col="all_ses_small", suffix="small", group_q="SES_q10"
    )
    desc_ses()

    aux_desc(data)


if __name__ == "__main__":
    main()
