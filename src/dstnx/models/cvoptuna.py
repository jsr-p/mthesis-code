import json
from functools import partial
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

import dstnx
from dstnx import log_utils, data_utils

LOGGER = log_utils.get_logger(name=__name__)
CLASSIFIERS = {"logreg": LogisticRegression, "lgbm": lgb.LGBMClassifier}

lgb.register_logger(LOGGER)


class OptunaParams:
    def __init__(self) -> None:
        pass

    def get(self, clf_name: str, trial: optuna.Trial):
        match clf_name:
            case "logreg":
                return self.logreg(trial)
            case "lgbm":
                return self.lgbm(trial)
            case _:
                raise ValueError(f"Invalid classifier: {clf_name}")

    def lgbm(self, trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            # limit the max depth for tree model
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            # max number of leaves in one tree
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            # number of boosting iterations; alias is `num_iterations`
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            # minimal sum hessian in one leaf; to deal with over-fitting
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # Alias is `feature_fraction`
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
            # Alias is `bagging_fraction`; randomly select part of data without resampling
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            # L1 regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            # L2 regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_jobs": 4,
            # "device" : "gpu",
            "verbose": 1,  # info
        }

    def logreg(self, trial: optuna.Trial):
        return {
            # "tol": trial.suggest_float("tol", 1e-6, 1e-3),
            "C": trial.suggest_float("C", 1e-2, 1, log=True),
            "tol": 1e-4,  # Default
            "n_jobs": 4,
            "verbose": 0,
            "max_iter": 200,
        }


# --------------------- Optuna --------------------- #

OPTUNA_PARAMS = OptunaParams()


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    num_inner_folds: int = 5,
    n_jobs_cv: int = 4,
):
    clf = CLASSIFIERS[clf_name](**OPTUNA_PARAMS.get(clf_name, trial))
    # Stratified k-fold with k=`num_inner_folds`
    score = cross_val_score(
        clf, X, y, cv=num_inner_folds, n_jobs=n_jobs_cv, scoring="roc_auc"
    ).mean()
    return score


def _create_study(storage, study_name):
    """Fix for optuna not allowing rewriting an existing study.

    Assumes the `study_name` consists of a name and number with
    a `-` between.
    """
    try:
        return optuna.create_study(
            direction="maximize",  # Maximize ROC
            storage=storage,
            study_name=study_name,
        )
    except optuna.exceptions.DuplicatedStudyError:
        optuna.delete_study(study_name=study_name, storage=storage)
    return optuna.create_study(
        direction="maximize",  # Maximize ROC
        storage=storage,
        study_name=study_name,
    )


def optuna_single(
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    study_name: str = "lgbm-study",
    n_trials: int = 20,
    num_inner_folds: int = 5,
):
    file_path = "./journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    LOGGER.debug(f"Creating optuna study for {study_name=}")
    study = _create_study(storage, study_name)
    study.optimize(
        partial(
            objective,
            X=X_train,
            y=y_train,
            clf_name=clf_name,
            num_inner_folds=num_inner_folds,
        ),
        n_trials=n_trials,
    )
    best_params = study.best_params
    clf = CLASSIFIERS[clf_name](**best_params)
    clf.fit(X_train, y_train)  # Refit on whole training data

    return clf, best_params


class OptunaModel:
    def __init__(self, clf_name: str) -> None:
        self.clf_name = clf_name
        self.best_params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf, self.best_params = optuna_single(
            self.clf_name, X, y, study_name=f"{self.clf_name}-1"
        )
        if self.clf_name == "logreg":
            self.coef_ = self.clf.coef_
        elif self.clf_name == "lgbm":
            self.feature_importances_ = self.clf.feature_importances_

    def predict(self, X: np.ndarray):
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray):
        return self.clf.predict_proba(X)


def conduct_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    n_trials: int = 30,
    fp: Path = dstnx.fp.DATA,
    study_name: str = "study",
    filename: str = "optuna_scores.json",
    num_outer_folds: int = 5,
    num_inner_folds: int = 5,
):
    print(
        f"Conducting study {study_name} with {n_trials} trials and "
        f"({num_outer_folds=}, {num_inner_folds=})"
    )
    outer_cv = KFold(n_splits=num_outer_folds, shuffle=True, random_state=1)
    scores = []
    for i, (train_ix, test_ix) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        clf = optuna_single(
            clf_name,
            X_train,
            y_train,
            f"{clf_name}-{study_name}-fold{i}",
            n_trials,
            num_inner_folds,
        )
        score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"ROC AUC score: {score}")
        scores.append((score, clf.get_params()))

    with open(fp / filename, "w") as f:
        json.dump(scores, f, indent=4)


def load_scores(fp: Path = dstnx.fp.DATA, filename: str = "optuna_scores.json"):
    with open(fp / filename, "r") as f:
        return json.load(f)


def main():
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_redundant=1,
        n_informative=5,
        n_classes=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    conduct_optuna_study(
        X,
        y,
        clf_name="logreg",
        n_trials=10,
        num_outer_folds=2,
        num_inner_folds=2,
        study_name="logreg-new-all",
    )


if __name__ == "__main__":
    main()
