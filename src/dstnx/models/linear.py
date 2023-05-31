from functools import partial

import click
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from statsmodels.tools.sm_exceptions import MissingDataError

import dstnx
from dstnx import data_utils, log_utils
from dstnx.features import merge
from dstnx.models import feat_sets
from dstnx.models import output as m_output

LOGGER = log_utils.get_logger(name=__name__)


def linear_model(y, X, target, name):
    # Statsmodels throws an error if the datatypes aren't strict float64 (Not Float64)
    # for features and int (Not Int64) for binary outcome ...
    try:
        model = sm.OLS(
            y.astype(int), X.astype(np.float64), hasconst=True, missing="drop"
        )
    except MissingDataError as ex:
        gp = (np.isinf(X) | X.isna()).sum(axis=0)
        print(y.isna().sum())
        print(gp[gp > 0])
        print(gp.sum())
        raise ex
    results = model.fit()
    # Printing the OLS summary
    res = m_output.abs_sum2(results, by="t")
    print(f"{f'{target}-{name}':-^50}")
    print(res.iloc[:15][["Coef.", "t", "P>|t|"]])

    for sort_val in ["t", "coef"]:
        data_utils.log_save_tabulate(
            filename=f"{target}-{name}-{sort_val}",
            df=m_output.abs_sum2(results, by=sort_val),
            fp=dstnx.fp.REG_OUTPUT / "models",
        )


def score_dict(y_test, y_pred, y_pred_prob):
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC Score": roc_auc_score(y_test, y_pred_prob),
    }
    return scores


def ml_model(y, X, target, name, clf):
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )

    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_prob = clf.predict_proba(X_test)[:, 1].ravel()
    y_pred = clf.predict(X_test)

    clf_id = f"{target}-{name}-{type(clf).__name__}"

    scores = score_dict(y_test, y_pred, y_pred_prob)
    for k, v in scores.items():
        LOGGER.info(f"{k}: {v}")
    data_utils.log_save_json_append(
        filename=f"{target}", obj=scores, key=f"{name}-{type(clf).__name__}"
    )

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    data_utils.log_save_fig(filename=f"{clf_id}-confmat", fig=fig)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=ax)
    data_utils.log_save_fig(filename=f"{clf_id}-roccurve", fig=fig)

    feature_importance_df = pd.DataFrame(
        {"name": X.columns.tolist(), "score": clf.feature_importances_}
    )
    data_utils.log_save_tabulate(
        filename=f"{clf_id}-featimp",
        df=feature_importance_df.sort_values(by="score", ascending=False),
        fp=dstnx.fp.REG_OUTPUT / "models",
    )


def get_duplicates(lst):
    return list(set([x for x in lst if lst.count(x) > 1]))


def estimate_model(suffix: str, inner_fn, radius: int, k: int, col_suffix: str):
    LOGGER.info(f"Estimating for {suffix} with {inner_fn.__name__=}")
    table = merge.load_full(
        suffix=suffix,
        as_pl=False,
        radius=radius,
        k=k,
        col_suffix=col_suffix,
        dtype_backend="numpy_nullable",
    )
    dist, dist_val = merge.validate_args(radius, k)  # Metadata for files
    weighted_cols = data_utils.load_json(dstnx.fp.DATA / "columns_w.json")
    normal_cols = data_utils.load_json(dstnx.fp.DATA / "columns.json")

    features: list[str] = (
        sum([cols for cols in normal_cols.values()], []) + feat_sets.EXTRA_FEATS
    )
    weighted_features: list[str] = (
        sum([cols for cols in weighted_cols.values()], []) + feat_sets.EXTRA_FEATS
    )

    for _name, _feats in zip(["weighted", "normal"], [weighted_features, features]):
        for _extra_name, extra_feat in zip(
            ["small_ses", "large_ses"], feat_sets.EXTRA_SETS
        ):
            name = f"{_name}-{_extra_name}-{dist}{dist_val}{col_suffix}"
            for target in feat_sets.TARGETS_ESTIMATE:
                feats = _feats + extra_feat
                LOGGER.info(
                    f"Estimating for ({name=}, {target=}) with {inner_fn.__name__=}..."
                )

                if len(feats) != len(set(feats)):
                    raise AssertionError(
                        f"Duplicate featueres:\n{get_duplicates(feats)}"
                    )

                # Log nans
                mask = table[[target] + feats].isna()
                nans = mask.sum(axis=0)
                data_utils.log_save_tabulate(
                    filename=f"nans-table-{name}", df=nans[nans > 0].to_frame("#nans")
                )
                table_notna = table[[target] + feats].dropna()
                merge.overview_table(
                    table=table_notna, name=f"overview_table_notna-{name}"
                )

                # Adding a constant column to X for the intercept term
                y = table_notna[target].astype(int)
                X = table_notna.drop(target, axis=1)
                X = sm.add_constant(X)
                year_dummies = pd.get_dummies(
                    table_notna.cohort, drop_first=True
                ).astype(int)
                kom_dummies = (
                    pd.get_dummies(table_notna.KOM).astype(int).drop(["411"], axis=1)
                )
                X = pd.concat((X, year_dummies, kom_dummies), axis=1).drop(
                    ["cohort", "KOM"], axis=1
                )
                # Fitting the OLS model
                LOGGER.info(f"Data dims: ({X.shape=}, {y.shape=})")
                inner_fn(y, X, target, name)


def _partial(fn, **kwargs):
    new_fn = partial(fn, **kwargs)
    new_fn.__name__ = fn.__name__
    return new_fn


@click.command()
@click.option("--radius", default=None, type=int, help="Radius to get features for")
@click.option("--k", default=None, type=int, help="k values to get features for")
@click.option(
    "--col-suffix",
    default=None,
    type=str,
    help="File suffix for neighbor measures",
)
def estimate_all(radius: int, k: int, col_suffix: str):
    estimate_model(
        "_new", inner_fn=linear_model, radius=radius, k=k, col_suffix=col_suffix
    )
    estimate_model(
        "_new",
        inner_fn=_partial(ml_model, clf=lgb.LGBMClassifier(n_jobs=4)),
        radius=radius,
        k=k,
        col_suffix=col_suffix,
    )


if __name__ == "__main__":
    estimate_all()
