import json

import pandas as pd

from dstnx import data_utils, fp


def col_info(df: pd.DataFrame):
    str_types = [str(_type) for _type in df.dtypes]
    return dict(zip(df.columns, str_types))


def construct_mock_regdata():
    mock_data = dict()
    for file in fp.REG_DATA.glob("*"):
        mock_data[file.name] = col_info(data_utils.load_reg(file.stem.split(".")[0]))

    with open(fp.DATA / "mock_data.json", "w") as f:
        json.dump(mock_data, f, indent=4)
    print(f"Saved mock data to {fp.DATA / 'mock_data.json'}")


if __name__ == "__main__":
    construct_mock_regdata()
