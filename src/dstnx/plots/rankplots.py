from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dstnx import data_utils

COL_NAMES = {
    "gym_apply": "Applied for high school (HSApply)",
    "gym_grad": "Graduated high school (HSGrad)",
    "eu_apply": "Applied for vocational training (VocApply)",
    "eu_grad": "Graduated vocational training (VocGrad)",
    "real_neet": "No job or upper secondary education at 25 (NoUSJob)",
}

COL_NAMES_COND = {
    "gym": "Graduated high school cond. on applying",
    "eu": "Graduated vocational training cond. on applying",
}

XLABEL_MAP = {
    "own_ses": "SES",
    "par_inc_avg": "ParInc",
    "par_edu_avg": "ParEduLen",
}


def bin_labels(bins: int = 100):
    match bins:
        case 100:
            return [rf"$q_{{{i}}}$" for i in range(0, 100)]
        case 50:
            return [rf"$q_{{{i}}}$" for i in range(0, 100, 2)]
        case 10:
            return [rf"$q_{{{i}}}$" for i in range(0, 100, 10)]
        case _:
            raise ValueError


def bin_delta_ticks(bins: int = 100):
    match bins:
        case 100:
            return 10
        case 50:
            return
        case _:
            raise ValueError


def qcut_col(full, col, bins=100):
    eps = np.random.normal(size=(full.shape[0]), loc=0, scale=0.005)
    return full.assign(
        qpc=lambda df: pd.qcut(df[col] + eps, q=bins, labels=bin_labels(bins))
    )


def rel_plot(
    gp,
    xlabel: str = "SES percentile",
    ylabel: str = r"$P(y_{it} = 1)$",
    title: str = "{col_name}",
    col="outcome",
    sharey: bool = True,
    **kwargs,
):
    if "row" in kwargs:
        extra_opts = kwargs
    else:
        extra_opts = {"col_wrap": 2}

    g: sns.FacetGrid = sns.relplot(
        data=gp,
        x="qpc",
        y="value",
        kind="scatter",
        facet_kws={
            "subplot_kws": {
                "xticks": range(0, 100 + 10, 10),
                "xticklabels": [rf"$q_{{{i}}}$" for i in range(0, 100 + 10, 10)],
            },
            "sharey": sharey,
        },
        col=col,
        color="black",
        **extra_opts,
    )
    g = g.set_axis_labels(xlabel, ylabel).set_titles(title).tight_layout(w_pad=0)
    return g


def single_bin_plot(
    gp,
    xlabel: str = "SES percentile",
    ylabel: str = r"$P(y_{it} = 1)$",
    title: str = "{col_name}",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.scatterplot(
        data=gp,
        x="qpc",
        y="value",
        color="black",
    )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        **{
            "xticks": range(0, 100 + 10, 10),
            "xticklabels": [rf"$q_{{{i}}}$" for i in range(0, 100 + 10, 10)],
        },
    )
    return fig


def cond_prob(full, target):
    gp = full.query(f"{target}_apply == 1", engine="python").groupby("qpc")[
        f"{target}_grad"
    ]
    assert (gp.count() > 5).all()
    return gp.mean().to_frame(target)


def rank_plot(
    full,
    col,
    include_targets: list[str] = ["all"],
    suffix: str = "",
    other_vars: dict[str, str] = None,
    single_plot: bool = False,
    **kwargs,
):
    if not other_vars:
        if include_targets == ["all"]:
            col_names = COL_NAMES
        else:
            col_names = {k: v for k, v in COL_NAMES.items() if k in include_targets}
    else:
        col_names = other_vars
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    gp = qcut_col(full, col).rename(columns=col_names).groupby("qpc")
    # DST compliance
    counts = gp.agg({v: "count" for v in col_names.values()})
    assert (counts > 5).all().all()
    gp = gp.agg({v: "mean" for v in col_names.values()})
    gp = gp.reset_index().melt(id_vars=["qpc"], var_name="outcome")

    if single_plot:
        title = list(other_vars.values())[0]
        fig = single_bin_plot(
            gp, xlabel=f"{XLABEL_MAP[col]} percentile", title=title, **kwargs
        )
    else:
        g = rel_plot(gp, xlabel=f"{XLABEL_MAP[col]} percentile", sharey=True, **kwargs)
        fig = g.fig

    data_utils.log_save_fig(fig=fig, filename=f"ses_outcomes_{col}{suffix}")
    return counts


def cond_plot(full, col):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    gp = qcut_col(full, col)
    gp = pd.concat(
        (
            cond_prob(gp, "gym"),
            cond_prob(gp, "eu"),
        ),
        axis=1,
    ).rename(columns=COL_NAMES_COND)
    gp = gp.reset_index().melt(id_vars=["qpc"], var_name="outcome")
    g = rel_plot(
        gp,
        xlabel=f"{XLABEL_MAP[col]} percentile",
        ylabel=r"$P(y_{it} = 1 | apply = 1)$",
        title="{col_name}",
        sharey=False,
    )
    data_utils.log_save_fig(fig=g.fig, filename=f"ses_outcomes_{col}_condapply")


def drop_low(gp):
    mask = gp["count"] <= 5
    if mask.any():
        print(f"{gp.loc[mask].shape[0]} bin had below 5 obs:\n{gp.loc[mask]}")
        gp = gp.loc[~mask]
    return gp["mean"]


def ses_q_facet(full, target="grad", bins=100, quintile_col: str = "SES_q"):
    gp = (
        full.pipe(qcut_col, "all_ses_small", bins=bins)
        .groupby([quintile_col, "qpc"])
        .agg({f"eu_{target}": ("mean", "count"), f"gym_{target}": ("mean", "count")})
    )
    gp = (
        pd.concat(
            (
                drop_low(gp[f"eu_{target}"]).to_frame("eu"),
                drop_low(gp[f"gym_{target}"]).to_frame("gym"),
            ),
            axis=1,
        )
        .reset_index()
        .melt(id_vars=[quintile_col, "qpc"], var_name="outcome")
    )
    g = rel_plot(
        gp,
        title="{row_name} (QuintileGroup={col_name})",
        xlabel="Avg. peer SES grundskole",
        row="outcome",
        col=quintile_col,
    )
    data_utils.log_save_fig(
        fig=g.fig, filename=f"ses_q_interact_outcomes_{target}_{quintile_col}"
    )


def load_plot_data(dtype_backend="pyarrow") -> pd.DataFrame:
    """
    NOTE: Loads only for merged data of k=30 nearest neighbors!
    """
    return (
        data_utils.load_reg("full_new_k30", as_pl=False, dtype_backend=dtype_backend)
        .assign(
            par_inc_avg=lambda df: (df.own_far_inc_14_17 + df.own_mor_inc_14_17) / 2,
            par_edu_avg=lambda df: (
                df.own_far_highest_edu_pria_14_17 + df.own_mor_highest_edu_pria_14_17
            )
            / 2,
        )
        .assign(
            INC_q=lambda df: pd.qcut(
                df.par_inc_avg, q=5, labels=[f"INC_Q{i}" for i in range(1, 6)]
            )
        )
    )


if __name__ == "__main__":
    full = load_plot_data(dtype_backend="numpy_nullable")

    # Condtional plots
    cond_plot(full, "own_ses")
    cond_plot(full, "par_inc_avg")
    cond_plot(full, "par_edu_avg")

    # SES / INC plots
    _ = rank_plot(
        full, "own_ses", include_targets=["gym_apply", "gym_grad"], suffix="_gym"
    )
    _ = rank_plot(
        full, "own_ses", include_targets=["eu_apply", "eu_grad"], suffix="_eu"
    )
    _ = rank_plot(full, "own_ses", include_targets=["real_neet"], suffix="_realneet")
    _ = rank_plot(
        full, "par_inc_avg", include_targets=["gym_apply", "gym_grad"], suffix="_gym"
    )
    _ = rank_plot(
        full, "par_inc_avg", include_targets=["eu_apply", "eu_grad"], suffix="_eu"
    )
    _ = rank_plot(
        full, "par_inc_avg", include_targets=["real_neet"], suffix="_realneet"
    )

    # GPA plots
    _ = rank_plot(
        full,
        "own_ses",
        suffix="_gpa",
        other_vars={"gpa": "9thGradeGPA"},
        ylabel="9thGradeGPA",
        single_plot=True,
    )
    _ = rank_plot(
        full,
        "par_inc_avg",
        suffix="_gpa",
        other_vars={"gpa": "9thGradeGPA"},
        ylabel="9thGradeGPA",
        single_plot=True,
    )
