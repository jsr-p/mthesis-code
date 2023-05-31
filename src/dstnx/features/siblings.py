import numpy as np
import pandas as pd

from dstnx import data_utils, log_utils

LOGGER = log_utils.get_logger(name=__name__)


def number_born(subset: pd.DataFrame) -> int:
    """Finds the number that the given person is born.

    Note:
        - `node_age` is constructed inside `_transform_siblings`
        - `ALDER` is the age of the sibling
    """
    (age,) = set(subset.node_age)
    number_born = np.searchsorted(subset.ALDER.sort_values(), age) + 1
    return number_born


class SiblingFeatures:
    def __init__(self, node_metadata: pd.DataFrame, year: int):
        self.node_metadata = node_metadata
        self.year = year
        self._load_siblings()
        self._transform_siblings()

    def _load_siblings(self):
        self.siblings = data_utils.load_reg(f"full_siblings_{self.year}").query(
            "SIBLING_PERSON_ID in @self.node_metadata.PERSON_ID"
        )
        assert self.siblings.SIBLING_PERSON_ID.isin(self.node_metadata.PERSON_ID).all()

    def _transform_siblings(self):
        age_map = self.node_metadata.set_index("PERSON_ID").ALDER.to_dict()
        # foed_dag_map = self.node_metadata.set_index("PERSON_ID").FOED_DAG.to_dict()
        self.siblings = self.siblings.assign(
            node_age=lambda df: df.SIBLING_PERSON_ID.map(age_map),
            age_diff=lambda df: df.ALDER - df.node_age,
            # has_twin=lambda df: df.SIBL
            number_born=lambda df: df.SIBLING_PERSON_ID.map(
                df.groupby("SIBLING_PERSON_ID").apply(number_born)
            ),
            # first_born=lambda df: np.where((df.number_born == 1) & (df.has_twin != 1), 1, 0),
            first_born=lambda df: np.where(df.number_born == 1, 1, 0),
            is_older=lambda df: df.age_diff < 0,
            is_younger=lambda df: df.age_diff > 0,
            any_older=lambda df: df.groupby("SIBLING_PERSON_ID")
            .is_older.transform("any")
            .astype(int),
            any_younger=lambda df: df.groupby("SIBLING_PERSON_ID")
            .is_younger.transform("any")
            .astype(int),
        )

    def merge(self, node_metadata: pd.DataFrame):
        sib_feat_cols = ["any_older", "any_younger", "number_born", "first_born"]
        node_metadata = node_metadata.merge(
            (
                self.siblings[["SIBLING_PERSON_ID", *sib_feat_cols]]
                .rename(columns={"SIBLING_PERSON_ID": "PERSON_ID"})[
                    ["PERSON_ID", *sib_feat_cols]
                ]
                .drop_duplicates()
            ),
            how="left",
            on="PERSON_ID",
        )
        for col in ["any_older", "any_younger"]:
            node_metadata[col].fillna(0, inplace=True)
        return node_metadata
