import numpy as np
from dstnx.models import choice


def var_decomp(full, val_col, group_col):
    T = 6
    N = full[group_col].unique().shape[0]
    sw = np.sum(
        (full[val_col] - full.groupby([group_col])[val_col].transform("mean")).pow(2)
        / (N * T - N)
    )
    sb = np.sum(
        (full.groupby([group_col])[val_col].mean() - full[val_col].mean()).pow(2)
        / (N - 1)
    )
    print(f"Variance decomp for ({val_col:<12}, {group_col:<10}")
    print(f"s^2_w={sw}, s^2_b={sb}")
    print("- " * 60)


if __name__ == "__main__":
    full = choice.load_data("numpy_nullable", with_id=True)
    cols = [
        "female",
        "imm",
        "all_ses_small",
        "gpa",
        "par_inc_avg",
        "par_edu_avg",
        "eu_grad",
        "gym_grad",
    ]
    for group_col in ["INSTNR", "KOM"]:
        for col in cols:
            var_decomp(full, col, group_col)
