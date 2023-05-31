import dstnx
import pandas as pd
import numpy as np

from dstnx.tables import interact
from dstnx.features import utils as feat_utils
from dstnx.tables import name_mappings, descriptive
from dstnx.models import model_prep
from dstnx.utils import tex
from dstnx import data_utils
from functools import partial


def compute_desc_radius():
    radius200 = data_utils.load_reg(
        "full_new_radius200_defaultradius_defaultradius200",
        dtype_backend="numpy_nullable",
    )
    cols = data_utils.load_json(
        dstnx.fp.DATA
        / "feature-columns"
        / "columns_w-radius200_defaultradius_defaultradius200.json"
    )
    feat_cols = feat_utils.cols_picker(cols, "all")
    subset = (
        radius200[feat_cols]
        .melt()
        .astype({"value": np.float64})
        .groupby("variable")
        .agg(("mean", "std"))  # Pandas describe does not handle nans ...
        .droplevel(0, axis=1)
        # radius200[feat_cols].describe().transpose()[["count", "mean", "std"]]
        # .pipe(model_prep.convert_non_nullable)
    )
    df = pd.DataFrame(name_mappings.parse_cols(subset.index), index=subset.index)
    myres = pd.concat((subset, df), axis=1).assign(
        period=lambda df: pd.Categorical(
            df.period.map(name_mappings.PERIOD_MAP),
            ordered=True,
            categories=["EC", "PS", "ES", "MS", "HS"],
        )
    )
    return myres.assign(d="radius").reset_index(drop=True)


def compute_desc_k():
    radius200 = data_utils.load_reg(
        "full_new_k50_defaultk_defaultk50", dtype_backend="numpy_nullable"
    )
    cols = data_utils.load_json(
        dstnx.fp.DATA / "feature-columns" / "columns_w-k50_defaultk_defaultk50.json"
    )
    feat_cols = feat_utils.cols_picker(cols, "all")
    subset = (
        radius200[feat_cols]
        .melt()
        .astype({"value": np.float64})
        .groupby("variable")
        .agg(("mean", "std"))  # Pandas describe does not handle nans ...
        .droplevel(0, axis=1)
        # radius200[feat_cols].describe().transpose()[["count", "mean", "std"]]
        # .pipe(model_prep.convert_non_nullable)
    )
    df = pd.DataFrame(name_mappings.parse_cols(subset.index), index=subset.index)
    myres = pd.concat((subset, df), axis=1).assign(
        period=lambda df: pd.Categorical(
            df.period.map(name_mappings.PERIOD_MAP),
            ordered=True,
            categories=["EC", "PS", "ES", "MS", "HS"],
        )
    )
    return myres.assign(d="k").reset_index(drop=True)


if __name__ == "__main__":
    desc_k = compute_desc_k()
    desc_radius = compute_desc_radius()
    all_desc = pd.concat((desc_k, desc_radius))

    tab = (
        all_desc.assign(
            period=lambda df: pd.Categorical(
                df.period, ordered=True, categories=["EC", "PS", "ES", "MS", "HS"]
            )
        )
        .melt(
            id_vars=["colname", "d", "period"],
            value_vars=["mean", "std"],
            var_name="desc",
            value_name="desc_values",
        )
        .pivot_table(
            index=["colname", "desc"], columns=["d", "period"], values=["desc_values"]
        )
        .droplevel(0, axis=1)
        .round(2)
        .pipe(interact.strip_multiindex)
        .rename(
            columns={"k": "$k = 50$", "radius": "$r = 200$"}, index=descriptive.DESC_MAP
        )
        .fillna("-")
        .style.format(precision=2)
        .to_latex(
            multirow_align="l",
            multicol_align="c",
            column_format=2 * "l" + 2 * 5 * "c",
        )
    )

    data_utils.log_save_txt(
        filename=f"desc-upbringing-k-radius",
        suffix=".tex",
        text=tex.compose(
            tab,
            tex_fns=[
                tex.add_rules,
                partial(
                    tex.add_column_midrules,
                    add_row_midrule=False,
                    num_idx=2,
                    num_subcols=5,
                    num_cols=2,
                ),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
