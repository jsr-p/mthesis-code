import re
from collections.abc import Callable
from tabnanny import check

import numpy as np
import pandas as pd

RE_TABULAR = re.compile(r"\\(begin|end){tabular}")
RE_MULTICOL = re.compile(r"\\multicolumn")
RE_STRIP_MIDRULE = re.compile(r"\\midrule\n")


def strip_midrule(tab: str) -> str:
    return RE_STRIP_MIDRULE.sub("", tab)


def underset_se(se, est, in_math_mode: bool = True):
    """Helper function for `se_below_ests`"""
    if in_math_mode:
        return rf"$\underset{{({se})}}{{{est}}}$"
    return rf"\underset{{({se})}}{{{est}}}"


def check_se_est_nan(se, est):
    if isinstance(se, str) and isinstance(est, str):
        return False
    # elif isinstance(se, (int, float)) and isinstance(est, (int, float)):
    #     return False
    elif isinstance(se, (int, float)) and isinstance(
        est, str
    ):  # non-significant estimate as a str
        return False
    print(se, est, type(se), type(est))
    return np.isnan(se) | np.isnan(est)


def se_below_ests(ses, ests, in_math_mode: bool = True):
    """Function to put s.e.s underneath estimates"""
    vals = [
        underset_se(se, est, in_math_mode) if not check_se_est_nan(se, est) else np.nan
        for se, est in zip(ses, ests)
    ]
    return vals


def underset_df(coef: pd.DataFrame, se: pd.DataFrame, in_math_mode: bool = True):
    shape = coef.shape
    return pd.DataFrame(
        np.array(
            se_below_ests(se.values.ravel(), coef.values.ravel(), in_math_mode)
        ).reshape(*shape),
        columns=coef.columns,
        index=coef.index,
    )


def assign_pval(pval, est, in_math_mode: bool = True):
    if pval > 0.05:
        new_est = f"{est}"
    elif pval < 0.01:
        new_est = rf"{est}^{{**}}"
    elif pval <= 0.05:
        new_est = rf"{est}^{{*}}"
    elif np.isnan(pval):
        return np.nan
    else:
        raise ValueError
    if in_math_mode:
        return f"${new_est}$"
    return new_est


def pval_above_ests(pvals, ests, **kwargs):
    """Function to put s.e.s underneath estimates"""
    vals = [assign_pval(pval, est, **kwargs) for pval, est in zip(pvals, ests)]
    return vals


def assign_pval_starts(tab: pd.DataFrame, pvals: pd.DataFrame, **kwargs):
    shape = tab.shape
    return pd.DataFrame(
        np.array(
            pval_above_ests(pvals.values.ravel(), tab.values.ravel(), **kwargs)
        ).reshape(*shape),
        columns=tab.columns,
        index=tab.index,
    )


def find_line_match(rows: list[str], regex: re.Pattern):
    tabular_idcs = []
    for i, line in enumerate(rows):
        if regex.search(line):
            tabular_idcs.append(i)
    return tabular_idcs


def add_rules(table: str) -> str:
    rows = table.split("\n")
    tabular_idcs = find_line_match(rows, RE_TABULAR)
    assert len(tabular_idcs) == 2
    start, end = tabular_idcs
    new_tab = []
    for i, row in enumerate(rows):
        if i == end:
            new_tab.extend(["\\bottomrule"] * 2)
        new_tab.append(row)
        if i == start:
            new_tab.extend(["\\toprule"] * 2)
    return "\n".join(new_tab)


def get_midrule(start: int, end: int):
    return "\\cmidrule(lr){" + f"{start}-{end}" + "}"


def add_column_midrules(
    table: str,
    num_idx: int = 2,
    num_subcols: int = 2,
    num_cols: int = 4,
    add_row_midrule: bool = True,
) -> str:
    rows = table.split("\n")
    if num_subcols > 1:
        idcs = find_line_match(rows, RE_MULTICOL)
        assert len(idcs) == 1, f"`{idcs}` does not have length 1"
        (colidx,) = idcs
    else:
        idcs = find_line_match(rows, RE_TABULAR)
        assert len(idcs) == 2, "More than two tabular found"
        idc, _ = idcs
        colidx = idc + 1 + 2  # Cols start after \begin{tabular} + 2xtoprule
    midrules = " ".join(
        [
            get_midrule(s, s + num_subcols - 1)
            for s in range(
                num_idx + 1, num_idx + 1 + num_cols * num_subcols, num_subcols
            )
        ]
    )
    new_tab = []
    for i, row in enumerate(rows):
        new_tab.append(row)
        if i == colidx:
            new_tab.append(midrules)
        elif i == (colidx + 1) and add_row_midrule:  # Midrule for rows
            new_tab.append(get_midrule(1, 2))
    return "\n".join(new_tab)


def compose(table: str, tex_fns: list[Callable[[str], str]]):
    for tex_fn in tex_fns:
        table = tex_fn(table)
    return table
