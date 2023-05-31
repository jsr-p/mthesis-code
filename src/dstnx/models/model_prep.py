import re
from dataclasses import dataclass

import formulaic
import numpy as np
import pandas as pd

from dstnx import log_utils

LOGGER = log_utils.get_logger(name=__name__)


@dataclass
class ModelConfig:
    features: list[str]
    target: str
    group_col: str = ""
    interaction: bool = ""
    const: bool = False
    trend: bool = True


def get_form(conf: ModelConfig):
    form = f"{conf.target} ~ " + " + ".join(conf.features)
    if conf.group_col:
        form += f" + C({conf.group_col})"
    if conf.const:
        form += "+ 1"
    if conf.interaction:
        form += " + C(SES_q, contr.treatment(base='SES_Q5'))*all_ses_small"
    if conf.trend:
        form += " + trend"
    return form


def prepare_sm(data, conf: ModelConfig, as_arrays: bool = False):
    form = get_form(conf)
    LOGGER.debug(f"Estimating model for formula: {form}")
    LOGGER.debug(f"{data.shape[0]}")
    endog, exog = formulaic.model_matrix(form, data, na_action="ignore")
    LOGGER.debug(f"{endog.shape[0]}")
    endog = endog.astype(int)
    exog = exog.astype(np.float64)
    if as_arrays:
        return endog.values, exog.values
    return endog, exog


def clean_arrow_na(data, features, _type=np.float64):
    """Pandas does not drop nan values for pyarrow Float64; fix this!"""
    data = data.astype(convert_types(features, _type=_type))
    mask = data[features].isna().any(axis=1)
    print(f"#NaNs: {mask.sum()}")
    data = data.loc[~mask]
    return data


def convert_types(cols, _type) -> dict:
    return {col: _type for col in cols}


def to_non_nullable(dtypes) -> dict:
    dtypes = dict(zip(dtypes.index, dtypes.values.astype(str)))
    return {
        k: v.lower() for k, v in dtypes.items() if any(d in v for d in ["Int", "Float"])
    }


def convert_non_nullable(df: pd.DataFrame) -> pd.DataFrame:
    """Fix formulaic bug.

    Formulaic makes nan-values out of nullable dtypes.
    This function converts to usual numpy dtypes.
    """
    return df.astype(to_non_nullable(df.dtypes))


def prepare_data(df):
    """Prepares data for estimaiton.

    Categorical are converted; pyarrow dtypes are converted
    into numpy values.
    """
    return (
        df.astype({"KOM": "category", "INSTNR": "category"})
        .dropna()
        .pipe(convert_non_nullable)
        .dropna()
        .reset_index(drop=True)
    )


def desc_target(df, edu):
    return df[f"{edu}_grad"].mean()


RE_TARGET = re.compile("(eu|gym)_grad")


def cond_apply(conf: ModelConfig, df: pd.DataFrame):
    if not RE_TARGET.search(conf.target):
        raise ValueError(f"{conf.target=} is not a valid target for conditional trans.")
    edu, *_ = conf.target.split("_")
    trans_df = df.query(f"{edu}_apply == 1").reset_index(drop=True)
    LOGGER.debug(f"Distribution of target: {desc_target(df, edu)}")
    return trans_df


def transform(conf: ModelConfig, trans: str, df: pd.DataFrame):
    LOGGER.info(f"Transforming data with {trans=}")
    match trans:
        case "cond":
            return cond_apply(conf, df)
        case "impute":
            # Impute
            return df
        case _:
            raise ValueError
