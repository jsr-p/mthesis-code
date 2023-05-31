import numpy as np
import pandas as pd

from dstnx import data_utils


class DataIds:
    def __init__(
        self, data_type: str = "node_metadata", start: int = 1985, end: int = 2000
    ):
        self.data_type = data_type
        self.start = start
        self.end = end
        self.data = dict()
        self.ids = dict()
        self.load()

    def load(self):
        for year in range(self.start, self.end + 1):
            self.load_year(year)
        self.all_ids = np.concatenate(list(self.ids.values()))

    def load_year(self, year: int):
        self.data[year] = (
            data_utils.load_reg(f"{self.data_type}_{year}")
            .drop_duplicates()
            .assign(year=year)
        )
        self.ids[year] = self.data[year]["PERSON_ID"].values

    def cat_data(self):
        return pd.concat(self.data.values()).reset_index(drop=True)
