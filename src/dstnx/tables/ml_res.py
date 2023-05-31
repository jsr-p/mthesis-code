from dstnx.utils import results, tex
import pandas as pd
import dstnx
from dstnx import data_utils
from functools import partial
from dstnx.tables import interact


def create_table(ml_results, columns, num_subcols=2, num_cols=4):
    tab = ml_results.pivot_table(
        columns=columns, values="ROC AUC Score", index="target"
    ).pipe(interact.format_multiindex_table, num_idx=1, precision=2)
    return tex.compose(
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
    )


def convert_targets_cat(ser: pd.Series):
    return pd.Categorical(
        ser,
        ordered=True,
        categories=["HS", "Voc.", "US", "NEET"],
    )


TARGET_MAP = {
    "eu_grad": "Voc.",
    "gym_grad": "HS",
    "us_grad": "US",
    "eu_apply": "Voc.",
    "gym_apply": "HS",
    "us_apply": "US",
    "real_neet": "NEET",
}

MODEL_MAP = {"logreg": "LogReg", "lgbm": "LGBM"}


def convert_cols(df):
    return df.assign(
        target=lambda df: convert_targets_cat(df.target.replace(TARGET_MAP)),
        name=lambda df: df.name.replace(MODEL_MAP),
    )


if __name__ == "__main__":
    ml_results = results.read_ml_results()
    print(ml_results)
    data_utils.log_save_txt(
        filename="auc-simple-grad",
        suffix=".tex",
        text=create_table(
            ml_results.query("target.str.contains('grad')").pipe(convert_cols),
            columns=["fs", "name"],
            num_cols=4,
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
    data_utils.log_save_txt(
        filename="auc-simple-apply",
        suffix=".tex",
        text=create_table(
            ml_results.query("target.str.contains('apply')").pipe(convert_cols),
            columns=["fs", "name"],
            num_cols=4,
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )
