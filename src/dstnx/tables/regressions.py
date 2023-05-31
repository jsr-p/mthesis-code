import pandas as pd
import numpy as np
from functools import partial

import dstnx
from dstnx.tables import interact, name_mappings, descriptive
from dstnx.utils import results, tex
from dstnx import data_utils
from dstnx import data_utils, log_utils

LOGGER = log_utils.get_logger(name=__name__)


def create_tables(subset, group_var: str):
    df = pd.DataFrame(name_mappings.parse_cols(subset.index), index=subset.index)
    myres = pd.concat((subset, df), axis=1).assign(
        period=lambda df: pd.Categorical(
            df.period.map(name_mappings.PERIOD_MAP),
            ordered=True,
            categories=["EC", "PS", "ES", "MS", "HS"],
        )
    )
    LOGGER.debug(f"Col order: {myres.period}")
    mask = myres.isna().any(axis=1)
    table_coef = (
        myres.loc[~mask]
        .pivot(columns=["target", "period"], index="colname", values="Estimate")
        .mul(100)
        .round(1)
    )
    table_se = (
        myres.loc[~mask]
        .pivot(columns=["target", "period"], index="colname", values="Std..Error")
        .mul(100)
        .round(1)
    )
    table_pval = myres.loc[~mask].pivot(
        columns=["target", "period"], index="colname", values="Pr...t.."
    )
    table_tex = tex.underset_df(
        tex.assign_pval_starts(table_coef, table_pval, in_math_mode=False), table_se
    )

    # --------------------- interact --------------------- #

    other_vars = myres.loc[mask]
    q_mask = other_vars.index.str.contains(
        interact.REGEX_Q_NONCAP
    ) | other_vars.index.str.contains(group_var)
    q_vals = other_vars.loc[q_mask]
    mapping = interact.ses_mapping(q_vals, group_col=group_var)
    gp = q_vals.rename(mapping).reset_index(names="name")
    table_interact_coef = (
        gp.pivot(columns=["target"], index="name", values="Estimate").mul(100).round(2)
    )
    table_interact_se = (
        gp.pivot(columns=["target"], index="name", values="Std..Error")
        .mul(100)
        .round(2)
    )
    table_interact_pval = gp.pivot(columns=["target"], index="name", values="Pr...t..")
    table_interact_tex = tex.underset_df(
        tex.assign_pval_starts(
            table_interact_coef, table_interact_pval, in_math_mode=False
        ),
        table_interact_se,
    )
    # --------------------- controls --------------------- #

    control_mask = ~q_mask
    controls_coef = (
        other_vars.loc[control_mask]
        .reset_index(names="name")
        .pivot(columns=["target"], index="name", values="Estimate")
        .mul(100)
        .round(1)
    )
    controls_se = (
        other_vars.loc[control_mask]
        .reset_index(names="name")
        .pivot(columns=["target"], index="name", values="Std..Error")
        .mul(100)
        .round(2)
    )
    controls_pval = (
        other_vars.loc[control_mask]
        .reset_index(names="name")
        .pivot(columns=["target"], index="name", values="Pr...t..")
    )
    control_tex = tex.underset_df(
        tex.assign_pval_starts(controls_coef, controls_pval, in_math_mode=False),
        controls_se,
    )

    num_obs = other_vars.N.unique()
    assert num_obs.size == 1
    N = num_obs.item()
    print(f"{N=} for all regressions")
    return table_tex, table_interact_tex, control_tex


def export_tables(table_tex, table_interact_tex, control_tex, suffix, group_var):
    def _subset_main(table_tex, cols, name: str = "first"):
        num_idx = 1
        num_cols = 3
        num_subcols = 5
        tab = (
            table_tex[cols]
            .pipe(interact.strip_multiindex)
            .rename(columns=descriptive.TARGET_MAP_SPEC)
            .replace(rf"$\underset{{({np.nan})}}{{{np.nan}}}$", "-")
            .style.to_latex(
                multirow_align="l",
                multicol_align="c",
                column_format=num_idx * "l" + num_cols * num_subcols * "c",
            )
        )
        data_utils.log_save_txt(
            filename=f"results-main-{name}-{suffix}",
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

    cols = ["gym_grad", "eu_grad", "real_neet"]
    other_cols = ["us_grad", "gym_apply", "us_apply"]
    _subset_main(table_tex, cols, name="first")
    _subset_main(table_tex, other_cols, name="second")

    cols = cols + other_cols  # 6 in total
    tab = (
        table_interact_tex[cols]
        .pipe(interact.strip_multiindex)
        .rename(columns=descriptive.TARGET_MAP_SPEC)
        .replace(rf"$\underset{{({np.nan})}}{{{np.nan}}}$", "-")
        .style.to_latex(
            column_format="l" + len(cols) * "c",
        )
    )
    data_utils.log_save_txt(
        filename=f"results-interact-{suffix}",
        suffix=".tex",
        text=tex.compose(
            tab,
            tex_fns=[
                tex.add_rules,
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )

    tab = (
        control_tex[cols]
        .pipe(interact.strip_multiindex)
        .rename(columns=descriptive.TARGET_MAP_SPEC, index=name_mappings.CONTROL_MAP)
        .replace(rf"$\underset{{({np.nan})}}{{{np.nan}}}$", "-")
        .style.to_latex(
            column_format="l" + len(cols) * "c",
        )
    )
    data_utils.log_save_txt(
        filename=f"results-controls-{suffix}",
        suffix=".tex",
        text=tex.compose(
            tab,
            tex_fns=[
                tex.add_rules,
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


def construct_tables(
    res, feat_case: str, peer_col: str, weighted: str, extra_suffix=""
):
    subset = res.query(
        f"feat_case == '{feat_case}' and peer_col == '{peer_col}' and weighted == '{weighted}'"
    )
    table_tex, table_interact_tex, control_tex = create_tables(
        subset, group_var=peer_col
    )
    export_tables(
        table_tex,
        table_interact_tex,
        control_tex,
        suffix=f"{feat_case}-{peer_col}-{weighted}{extra_suffix}",
        group_var=peer_col,
    )


if __name__ == "__main__":
    # # r = 200
    res = results.cat_res(
        dstnx.fp.REG_TABLES / "fe_radius200_defaultradius_defaultradius200-quintiles"
    )
    construct_tables(
        res, "reduced", "all_ses_large", "w", extra_suffix="radius200default"
    )
    construct_tables(res, "all", "all_ses_large", "w", extra_suffix="radius200default")

    # # k = 30
    res = results.cat_res(dstnx.fp.REG_TABLES / "fe_k30_defaultk_defaultk30-quintiles")
    construct_tables(res, "reduced", "all_ses_large", "w", extra_suffix="k30default")
    construct_tables(res, "all", "all_ses_large", "w", extra_suffix="k30default")
    construct_tables(res, "reduced", "all_ses_q90", "w", extra_suffix="k30default")
    construct_tables(res, "all", "all_ses_q90", "w", extra_suffix="k30default")

    # Deciles
    res = results.cat_res(dstnx.fp.REG_TABLES / "fe_k30_defaultk_defaultk30-deciles")
    construct_tables(
        res, "all", "all_ses_large", "w", extra_suffix="k30default-deciles"
    )

    # k = 50
    res = results.cat_res(dstnx.fp.REG_TABLES / "fe_k50_defaultk_defaultk50-quintiles")
    construct_tables(res, "all", "all_ses_large", "w", extra_suffix="k50default")

    # r = 400
    res = results.cat_res(
        dstnx.fp.REG_TABLES / "fe_radius400_defaultradius_defaultradius400-quintiles"
    )
    construct_tables(res, "all", "all_ses_large", "w", extra_suffix="radius400default")
