import json

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import dstnx
from dstnx import log_utils
from dstnx.utils.address_comp import cols_to_numpy

LOGGER = log_utils.get_logger(name=__name__)

PCA_FILE = dstnx.fp.REG_DATA / "ses_pca.json"
SES_PCA_COLS = ["highest_edu_pria", "arblos", "inc", "inc_kont", "crimes"]


def load_pca() -> dict:
    with open(PCA_FILE, "r") as file:
        return json.load(file)


def load_pca_df():
    pcas = load_pca()
    columns = ["year"] + SES_PCA_COLS + ["E"]  # + Explained variance ratio
    return pd.DataFrame(
        [
            [year, *pcas[year]["pca_vec"], pcas[year]["explained_variance_ratio"]]
            for year in pcas
        ],
        columns=columns,
    )


def save_pca(year: int, pca_info: dict):
    if PCA_FILE.exists():  # Read file and append year
        pca_dict = load_pca()
        pca_dict[year] = pca_info
    else:  # Construct the dictionary from scratch and insert first element
        pca_dict = {}
        pca_dict[year] = pca_info
    with open(PCA_FILE, "w") as file:  # Write back the new dictionary
        json.dump(pca_dict, file, indent=4)


class NanPCA:
    def __init__(self, df):
        self.df = df
        self.compute_indices()

    def compute_indices(self):
        nans = self.df.isna()
        LOGGER.debug(f"Nans in SES cols:\n{nans.sum(axis=0)}")
        LOGGER.debug(f"Nans in SES cols (across row):\n{nans.sum(axis=1).sum()}")

        self.mask = nans.any(axis=1)
        self.idc_nan = np.where(self.mask)[0]
        self.idc_notna = np.where(~self.mask)[0]

    def any_nans(self) -> bool:
        return self.mask.any()

    def fill_pca_array(self, pca_scores: np.ndarray) -> np.ndarray:
        """Fill the pca array with missing values for the ones.

        X is assumed to be (N, ).
        """
        scores = np.empty(self.df.shape[0], dtype=np.float64)
        scores[self.idc_notna] = pca_scores
        scores[self.idc_nan] = np.nan
        return scores

    def subset_notna(self) -> np.ndarray:
        return self.df.loc[~self.mask].values


def construct_SES(
    df: pd.DataFrame,
    cols: list[str],
    year: int = 2020,
    fit_pca: bool = False,
    quantiles: bool = True,
    save: bool = True,
) -> np.ndarray:
    """Constructs SES-score for a given year.

    The score is computed by projected the p variables in
    `cols` to a 1-dimensional space using PCA.
    Nan-values are not considered in the computation.
    """
    pca_df = df[cols].pipe(cols_to_numpy, cols=cols, dtype="float64")
    nan_pca = NanPCA(pca_df)  # Handle nans
    if data_contains_nans := nan_pca.any_nans():
        X = nan_pca.subset_notna()
    else:
        X = pca_df.values
    X = StandardScaler().fit_transform(X)
    LOGGER.debug(f"{X.mean(axis=0)}, {X.std(axis=0)}")
    if fit_pca:
        LOGGER.debug(f"Fitting PCA for {year=}")
        pca = PCA(n_components=1)
        pca.fit(X)
        LOGGER.debug(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        LOGGER.debug(f"Linear combination with {cols}:\n{pca.components_}")

        # Transform by flipping pca vector based on the education variable
        idx_highest_edu = 0
        pca_vec = pca.components_.ravel()[:, np.newaxis]  # From (D, p) -> (p, D)
        sign_vec = np.sign(pca_vec[idx_highest_edu, :])
        if not sign_vec == -1:
            LOGGER.warning("Sign of pca vector is not as expected!")
        else:
            # Flip sign s.t. high SES -> high status; from manual inspection but holds
            # through the years cf. Gandil & Bjerre-Nielsen
            pca_vec = sign_vec * pca_vec
        X_SES = X @ pca_vec  # Project
        pca_info = {
            "pca_vec": pca_vec.squeeze().tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.item(),
        }
        if save:
            save_pca(year, pca_info)
    else:
        LOGGER.debug(f"Computing linear combination with: {pca_vec}")
        pca_vec = np.array(load_pca()[year]["pca_vec"])[:, np.newaxis]
        assert pca_vec.shape[1] == 1, "PCA vec should be (D, 1)"
        assert cols == SES_PCA_COLS
        LOGGER.debug(f"Computing linear combination with: {pca_vec}")
        X_SES = X @ pca_vec
    if quantiles:
        scores = pd.qcut(MinMaxScaler().fit_transform(X_SES).ravel(), q=100).codes
    else:
        scores = MinMaxScaler().fit_transform(X_SES).ravel()
    if data_contains_nans:  # Reconstruct array and fill nans
        return nan_pca.fill_pca_array(scores)
    return scores


if __name__ == "__main__":
    ...
