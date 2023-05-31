import numpy as np
import pandas as pd

from dstnx import funcs


def test_leave_one_out_mean():
    df = pd.DataFrame(
        {"group_id": [1, 1, 1, 2, 2, 2], "value": [1, 2, 3, 4, 5, 6]}
    ).assign(group_avg=lambda df: funcs.group_avg(df, rel_col="value"))
    assert (df.group_avg == [2] * 3 + [5] * 3).all()
    df = df.assign(
        group_count=[3] * 6,
        classmates_avg=lambda df: funcs.leave_one_out_mean(df, rel_col="value"),
    )
    assert (
        df.classmates_avg
        == np.array(
            [
                np.array([2, 3]).mean(),
                np.array([1, 3]).mean(),
                np.array([1, 2]).mean(),
                np.array([5, 6]).mean(),
                np.array([4, 6]).mean(),
                np.array([4, 5]).mean(),
            ]
        )
    ).all()


if __name__ == "__main__":
    test_leave_one_out_mean()
