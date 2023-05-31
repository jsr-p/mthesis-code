from multiprocessing.sharedctypes import Value
from pathlib import Path

import pandas as pd

import dstnx
from dstnx import data_utils


def read_fe_csv(file: Path):
    # {target_col}-{dist}{col_suffix}-{name}-{group_col}-{peer_col}-{extra_suffix}.txt  # Format of col
    try:
        (
            target,
            _,
            name,
            group_col,
            peer_col,
            extra_suffix,
            feat_case,
            percentile,
        ) = file.stem.split("-")
    except ValueError as e:
        print(f"{file} gave an valueerror")
        raise e
    weighted = name
    return pd.read_csv(file, index_col=0).assign(
        target=target,
        group_col=group_col,
        peer_col=peer_col,
        weighted=weighted,
        extra_suffix=extra_suffix,
        feat_case=feat_case,
        percentile=percentile,
    )


def cat_res(reg_folder: Path) -> pd.DataFrame:
    try:
        return pd.concat(read_fe_csv(file) for file in reg_folder.glob("*"))
    except ValueError as exc:
        print(f"Got ValueError for {reg_folder}")
        raise exc


MODEL_IDENTIFERS = (
    "name",
    "fs",
    "group_col",
    "trans",
    "data_suffix",
    "feature_suffix",
    "conduct_optuna_study",
)


def parse_model(model: str):
    # return dictionary
    return {k: v for k, v in zip(MODEL_IDENTIFERS, model.split("-"))}


def read_ml_results() -> pd.DataFrame:
    fp = dstnx.fp.REG_OUTPUT / "model-scores"
    data = []
    for target_file in fp.glob("*-scores.json"):
        scores = data_utils.load_json(target_file)
        for model, result in scores.items():
            data.append(
                parse_model(model) | {"target": target_file.stem.split("-")[0]} | result
            )
    return pd.DataFrame(data)


def read_featimps() -> pd.DataFrame:
    fp = dstnx.fp.REG_OUTPUT / "model-explain"
    data = []
    for target_file in fp.glob("*featimp.parquet"):
        target = target_file.stem.split("-")[0]
        model = "-".join(target_file.stem.split("-")[1:])
        attrs = parse_model(model) | {"target": target}
        df = pd.read_parquet(target_file).assign(**attrs)
        data.append(df)
    return pd.concat(data).reset_index(drop=True)
