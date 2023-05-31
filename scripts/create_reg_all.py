import pandas as pd

import dstnx
from dstnx import data_utils
from dstnx.utils import siblings

START = 1995
END = 2000


def create_school_by_family():
    return siblings.construct_school_by_family(start=START, end=END, suffix="_whole")


def main():
    reg_all = (
        pd.concat(
            pd.read_parquet(dstnx.fp.REG_DATA / f"reg_{year}.parquet.gzip").assign(
                year=year
            )
            for year in range(START, END + 1)
        )
        .reset_index(drop=True)
        .astype(
            {
                # Avoid pyarrow issues
                "mor_BESKST": "str",
                "far_BESKST": "str",
            }
        )
    )
    # reg_all.to_parquet(dstnx.fp.REG_DATA / "reg_all.parquet.gzip")

    school_by_family = create_school_by_family()
    reg_school_by_family = (
        reg_all.astype({"PERSON_ID": "int64[pyarrow]"})
        .merge(school_by_family, how="left", on=["PERSON_ID", "INSTNR"])
        .dropna(subset="school_by_family")
        .reset_index(drop=True)
    )
    print(unqs := reg_school_by_family.school_by_family.unique().shape[0])
    print(obs := reg_school_by_family.shape[0])
    print(unqs / obs)
    reg_school_by_family.to_parquet(
        dstnx.fp.REG_DATA / f"reg_school_by_family_{START}-{END}.parquet.gzip"
    )
    name = "radius"
    nx_measures = (
        data_utils.load_nx_period(
            start=1985, end=1995, period=1, name=name, method="nw"
        )
        .dropna()
        .reset_index(names="PERSON_ID")
        .astype({"PERSON_ID": "int64[pyarrow]"})
    )
    nx_df = reg_school_by_family.merge(nx_measures, how="left", on="PERSON_ID").dropna()
    print("Shape without nans:", nx_df.shape[0])
    filename = (
        dstnx.fp.REG_DATA / f"reg_school_by_family_nx_{name}_{START}-{END}.parquet.gzip"
    )
    nx_df.to_parquet(filename)
    print(f"Saved file to {filename}")


if __name__ == "__main__":
    main()
