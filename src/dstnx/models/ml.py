from dataclasses import dataclass

import click
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dstnx
from dstnx import data_utils, log_utils
from dstnx.models import cvoptuna, feat_sets, model_prep, model_utils

LOGGER = log_utils.get_logger(name=__name__)


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    features: list[str]
    target: str

    def extract(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


def split_data(full, X, y, target, features) -> DataSplit:
    X_train, X_test, y_train, y_test = model_utils.train_test_from_df(
        full, X.values, y.values.ravel()
    )
    LOGGER.info(f"{X_train.shape[0]=}, {X_test.shape[0]=}")
    return DataSplit(X_train, X_test, y_train, y_test, features, target)


def score_dict(y_test, y_pred, y_pred_prob):
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC Score": roc_auc_score(y_test, y_pred_prob),
    }
    return scores


@dataclass
class ModelIdentity:
    name: str
    target: str
    fs: str
    group_col: str
    interaction: bool
    trans: str
    data_suffix: str
    feature_suffix: str
    conduct_optuna_study: bool

    def get_identifier(self) -> str:
        identifiers = [
            self.name,
            self.fs,
            self.group_col,
            self.trans,
            self.data_suffix,
            self.feature_suffix,
            self.conduct_optuna_study,
        ]
        return "-".join(str(item) if item else "None" for item in identifiers)

    @property
    def column_file_suffix(self):
        return f"_{self.data_suffix}{self.feature_suffix}"

    @property
    def data_file_suffix(self):
        return f"{self.data_suffix}{self.feature_suffix}"


def ml_model(data_split: DataSplit, model: ModelIdentity, clf):
    X_train, X_test, y_train, y_test = data_split.extract()
    target, features = data_split.target, data_split.features
    clf.fit(X_train, y_train)  # Do cross-validation here

    # Save params if optuna
    if model.conduct_optuna_study:
        clf: cvoptuna.OptunaModel
        best_params = clf.best_params
        data_utils.log_save_json_append(
            filename=f"{model.target}-bestparams",
            obj=best_params,
            key=model.get_identifier(),
            fp=dstnx.fp.REG_OUTPUT / "model-scores",
        )

    # Make predictions on the test set
    y_pred_prob = clf.predict_proba(X_test)[:, 1].ravel()
    y_pred = clf.predict(X_test)
    scores = score_dict(y_test, y_pred, y_pred_prob)
    for k, v in scores.items():
        LOGGER.info(f"{k}: {v}")

    # Log output
    data_utils.log_save_json_append(
        filename=f"{target}-scores",
        obj=scores
        | {
            "N_train": X_train.shape[0],
            "N_test": X_test.shape[0],
            "y_train_avg": y_train.mean(),
            "y_test_avg": y_test.mean(),
        },
        key=model.get_identifier(),
        fp=dstnx.fp.REG_OUTPUT / f"model-scores",
    )

    clf_id = f"{target}-{model.get_identifier()}"

    # # Conf
    # fig, ax = plt.subplots()
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    # data_utils.log_save_fig(
    #     filename=f"{clf_id}-confmat", fig=fig, fp=dstnx.fp.REG_PLOTS / "mlplots"
    # )
    # plt.close()

    # fig, ax = plt.subplots()
    # RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=ax)
    # data_utils.log_save_fig(
    #     filename=f"{clf_id}-roccurve", fig=fig, fp=dstnx.fp.REG_PLOTS / "mlplots"
    # )
    # plt.close()

    # Featimp
    if isinstance(clf, Pipeline):
        clf = clf.named_steps[model.name]
    if model.name in ["xgb", "lgbm"]:
        scores = clf.feature_importances_
    elif model.name in ["logreg"]:
        scores = np.abs(clf.coef_.ravel())  # LogReg
    else:
        scores = np.zeros(len(features))
    feature_importance_df = pd.DataFrame({"model.name": features, "score": scores})
    data_utils.log_save_pq(
        filename=f"{clf_id}-featimp",
        df=feature_importance_df.sort_values(by="score", ascending=False),
        fp=dstnx.fp.REG_OUTPUT / "model-explain",
    )


def construct_pl(model: ModelIdentity, clf, clf_name: str):
    steps = [("scaler", StandardScaler())]
    if model.trans == "impute":
        steps.append(("imputer", SimpleImputer()))
    steps.append((clf_name, clf))
    return Pipeline(steps)


def fit_with_optuna(model: ModelIdentity, data_split: DataSplit):
    if model.name not in ["lgbm", "logreg"]:
        raise ValueError
    clf = cvoptuna.OptunaModel(model.name)
    clf = construct_pl(model, clf)
    ml_model(data_split, model=model, clf=clf)


def fit_model(model: ModelIdentity, data_split: DataSplit):
    if model.conduct_optuna_study:
        fit_with_optuna(model, data_split)
        return
    match model.name:
        case "lgbm":
            clf = lgb.LGBMClassifier(n_jobs=4)
        case "logreg":
            clf = LogisticRegression(n_jobs=4)
        case "logregl1":
            clf = LogisticRegression(n_jobs=4, penalty="l1", solver="liblinear")
        case "logregel":
            clf = LogisticRegression(n_jobs=4, penalty="elasticnet", solver="saga")
        case _:
            raise ValueError
    clf = construct_pl(model, clf, clf_name=model.name)
    ml_model(data_split, model=model, clf=clf)


def _to_non_nullable(full):
    mask = (full.dtypes == "object") | (full.dtypes == "category")
    cols = full.columns[~mask].tolist()
    full = full.astype({col: np.float64 for col in cols})
    return full


def fit(model: ModelIdentity):
    LOGGER.info(f"Training {model.name=} with {model.target=} and {model.fs=}")
    features = feat_sets.get(model.fs, model.data_file_suffix)
    full = (
        data_utils.load_reg(
            f"full_new_{model.data_file_suffix}",
            as_pl=False,
            dtype_backend="numpy_nullable",
        )
        .astype({"KOM": "category", "INSTNR": "category"})
        .assign(trend=lambda df: (df.cohort - 1991).astype(int))
    )
    LOGGER.debug(f"Full has columns: {full.columns.tolist()}")
    conf = model_prep.ModelConfig(
        features=features,
        target=model.target,
        group_col=model.group_col,
        interaction=model.interaction,
        trend=False,
    )
    if model.trans:
        full = model_prep.transform(conf, model.trans, full)
    if model.trans != "impute":
        LOGGER.info("Dropping nan values from dataframe")
        full = (
            full[features + [model.target] + ["cohort"]]
            .fillna(np.nan)
            .pipe(_to_non_nullable)
            .dropna()
            .reset_index(drop=True)
        )
    else:
        full = full.fillna(np.nan)[
            features + [model.target] + ["cohort", "KOM", "INSTNR"]
        ].pipe(_to_non_nullable)
        LOGGER.debug(f"Full df dtypes:{full.dtypes}")
    y, X = model_prep.prepare_sm(
        full, conf, as_arrays=False
    )  # Feature are changed inside here
    data_split = split_data(full, X, y, model.target, X.columns.tolist())
    fit_model(model, data_split)


@click.command()
@click.argument("model_name", default="gcn")
@click.option("--target", default="eu_grad")
@click.option("--fs", default="fs1")
@click.option("--group-col", default="")
@click.option("--interaction", default=False, is_flag=True)
@click.option("--trans", default="")
@click.option(
    "--data-suffix",
    default="",
    type=str,
    help="Suffix for data",
)
@click.option(
    "--feature-suffix",
    default="",
    type=str,
    help="File suffix for features",
)
@click.option(
    "--conduct-optuna",
    is_flag=True,
    default=False,
    help="Conduct optuna study",
)
def cli(
    model_name: str,
    target: str,
    fs: str,
    group_col: str,
    interaction: bool,
    trans: str,
    data_suffix: str,
    feature_suffix: str,
    conduct_optuna: bool,
):
    model = ModelIdentity(
        model_name,
        target,
        fs,
        group_col,
        interaction,
        trans,
        data_suffix,
        feature_suffix,
        conduct_optuna_study=conduct_optuna,
    )
    fit(model)


if __name__ == "__main__":
    cli()
