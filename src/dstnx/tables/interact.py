import re
from functools import partial

import pandas as pd

import dstnx
from dstnx import data_utils
from dstnx.utils import results, tex


def remap_index(df, mapping):
    df = df.copy()  # Copy df to allow reapplying fn when experimenting
    df.index = df.index.map(mapping)
    return df


def strip_names(df):
    df.index.name = ""
    df.columns.name = ""
    return df


def strip_multiindex(df):
    return df.rename_axis(columns=lambda x: None, index=lambda x: None)


def create_tables(subset, se_under: bool = True, t_under: bool = False) -> pd.DataFrame:
    def _create_table(value: str):
        table = (
            subset.reset_index(names="interaction")
            .pivot_table(values=value, index="target", columns="interaction")
            .round(4)
        )
        table.columns.name = value
        table.index.name = "Target"
        return table

    coef = _create_table("Estimate")
    tval = _create_table("t.value")
    se = _create_table("Std..Error")
    if se_under:
        return tex.underset_df(coef, se)
    if t_under:
        return tex.underset_df(coef, tval)
    return coef, tval, se


def format_multiindex_table(
    df: pd.DataFrame,
    num_cols: int = 4,
    num_subcols: int = 2,
    num_idx: int = 2,
    precision: int = 3,
) -> str:
    return (
        df.pipe(strip_multiindex)
        .style.format(
            {
                ("Numeric", "Integers"): "\${}",
                ("Numeric", "Floats"): "{:.2f}",
                ("Non-Numeric", "Strings"): str.upper,
            },
            precision=precision,
        )
        .to_latex(
            multirow_align="l",
            multicol_align="c",
            column_format=num_idx * "l" + num_cols * num_subcols * "c",
        )
    )


def load_res(filename: str = "fe_k50_all"):
    res = results.cat_res(dstnx.fp.REG_TABLES / filename).assign(
        peer_col=lambda df: df.peer_col.replace(
            dict(zip(["all_ses_large", "all_ses_small"], ["large", "small"]))
        )
    )
    sesvals = res.loc[res.index.str.contains("SES_Q\d+:|:SES_Q\d+")]
    return sesvals


REGEX_Q = re.compile(r"Q(\d+)")
REGEX_Q_NONCAP = re.compile(r"Q\d+")


def ses_mapping(sesvals, group_col: str = "all_ses_large"):
    unq_vals = sesvals.index.unique().tolist()
    mapping = dict()
    for val in unq_vals:
        match = REGEX_Q.search(val)
        if match and group_col in val:
            q_val = match.group(1)
            mapping[val] = rf"$Q^{{SES}}_{{{q_val}}} \times \overline{{SES}}_{{it}}$"
        elif match and group_col not in val:
            q_val = match.group(1)
            mapping[val] = rf"$Q_{{{q_val}}}$"
        else:
            mapping[val] = r"$\overline{{SES}}_{{it}}$"
    return mapping


def construct_simple():
    sesvals = load_res()
    mapping = ses_mapping(sesvals)
    subset = sesvals.query("peer_col == 'small' and weighted == 'nw'").copy()
    subset.index = subset.index.map(mapping)
    data_utils.log_save_txt(
        filename="interact-peer-small",
        suffix=".tex",
        text=create_tables(subset, se_under=True).style.to_latex(),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
    subset = sesvals.query("peer_col == 'large' and weighted == 'nw'").copy()
    subset.index = subset.index.map(mapping)
    data_utils.log_save_txt(
        filename="interact-peer-large",
        suffix=".tex",
        text=create_tables(subset, se_under=True).style.to_latex(),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def pivot_table(
    sesvals, mapping, value: str, index: list[str], columns: list[str], transpose=False
):
    gp = (
        sesvals.pipe(remap_index, mapping)
        .reset_index(names="interaction")
        .pivot_table(values=value, index=index, columns=columns)
        .round(4)
    )
    if transpose:
        return gp.transpose()
    return gp


MAP_DICT = {"eu_grad": "Voc.", "gym_grad": "HS", "us_grad": "US", "real_neet": "NEET"}


def load_multiindex(filename: str = "fe_k50_all"):
    return load_res(filename=filename).assign(
        # Convert to categorical for sorting :)
        target=lambda df: (
            pd.Categorical(
                df.target.replace(MAP_DICT),
                ordered=True,
                categories=["HS", "Voc.", "US", "NEET"],
            )
        ),
        weighted=lambda df: pd.Categorical(
            df.weighted.replace({"w": "W", "nw": "NW"}),
            ordered=True,
            categories=["NW", "W"],
        ),
        peer_col=lambda df: pd.Categorical(
            df.peer_col.str.capitalize(), ordered=True, categories=["Small", "Large"]
        ),
    )


def construct_multiindex():
    sesvals = load_multiindex()
    mapping = ses_mapping(sesvals)
    columns = ["interaction", "weighted"]
    index = ["target", "peer_col"]
    coef = pivot_table(sesvals, mapping, "Estimate", index, columns)
    se = pivot_table(sesvals, mapping, "Std..Error", index, columns)
    pvals = pivot_table(sesvals, mapping, "Pr...t..", index, columns)
    simple_tab = tex.underset_df(coef, se).pipe(format_multiindex_table)
    data_utils.log_save_txt(
        filename="interact-peer-large-multisimple",
        suffix=".tex",
        text=tex.compose(
            simple_tab,
            [tex.add_rules, partial(tex.add_column_midrules, add_row_midrule=False)],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
    new_tab = tex.underset_df(
        tex.assign_pval_starts(coef, pvals, in_math_mode=False), se
    ).pipe(format_multiindex_table)
    data_utils.log_save_txt(
        filename="interact-peer-large-multiadv",
        suffix=".tex",
        text=tex.compose(
            new_tab,
            [tex.add_rules, partial(tex.add_column_midrules, add_row_midrule=False)],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def construct_multiindex_woweight():
    sesvals = load_multiindex()
    mapping = ses_mapping(sesvals)
    columns = ["interaction"]
    index = ["target", "peer_col"]
    coef = pivot_table(sesvals, mapping, "Estimate", index, columns)
    se = pivot_table(sesvals, mapping, "Std..Error", index, columns)
    pvals = pivot_table(sesvals, mapping, "Pr...t..", index, columns)
    simple_tab = tex.underset_df(coef, se).pipe(format_multiindex_table)
    data_utils.log_save_txt(
        filename="interact-peer-large-multisimple-wo",
        suffix=".tex",
        text=tex.compose(
            simple_tab,
            [
                tex.add_rules,
                partial(tex.add_column_midrules, add_row_midrule=False, num_subcols=1),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
    new_tab = tex.underset_df(
        tex.assign_pval_starts(coef, pvals, in_math_mode=False), se
    ).pipe(format_multiindex_table)
    data_utils.log_save_txt(
        filename="interact-peer-large-multiadv-wo",
        suffix=".tex",
        text=tex.compose(
            new_tab,
            [
                tex.add_rules,
                partial(tex.add_column_midrules, add_row_midrule=False, num_subcols=1),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def construct_multiindex_dec():
    sesvals = load_multiindex(filename="fe_k50_all_all-deciles")
    mapping = ses_mapping(sesvals)
    columns = ["interaction", "weighted"]
    index = ["target", "peer_col"]
    coef = pivot_table(sesvals, mapping, "Estimate", index, columns, transpose=True)
    se = pivot_table(sesvals, mapping, "Std..Error", index, columns, transpose=True)
    pvals = pivot_table(sesvals, mapping, "Pr...t..", index, columns, transpose=True)
    print(coef)
    # 4 x 2 while we tranpose; 4 targets and two subcols; index have 2 by default
    simple_tab = tex.underset_df(coef, se).pipe(
        format_multiindex_table, num_cols=4, num_subcols=2
    )
    print(simple_tab)
    data_utils.log_save_txt(
        filename="interact-peer-large-multisimple",
        suffix=".tex",
        text=tex.compose(
            simple_tab,
            [
                tex.add_rules,
                partial(
                    tex.add_column_midrules,
                    add_row_midrule=False,
                    num_cols=4,
                    num_subcols=2,
                ),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
    new_tab = tex.underset_df(
        tex.assign_pval_starts(coef, pvals, in_math_mode=False), se
    ).pipe(format_multiindex_table, num_cols=4, num_subcols=2)
    data_utils.log_save_txt(
        filename="interact-peer-large-multiadv-dec",
        suffix=".tex",
        text=tex.compose(
            new_tab,
            [tex.add_rules, partial(tex.add_column_midrules, add_row_midrule=False)],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


if __name__ == "__main__":
    # construct_simple()
    # construct_multiindex()
    # construct_multiindex_woweight()
    construct_multiindex_dec()
