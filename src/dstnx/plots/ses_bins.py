import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import dstnx
from dstnx import data_utils
from dstnx.data import neighbors
from dstnx.plots import pca

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def std_col(ser):
    return (ser - ser.mean()) / ser.std()


def minmax_scale_col(ser):
    return (ser - ser.min()) / (ser.max() - ser.min())


def make_plot():
    """
    Notes for paper:
    Figure shows the average value of each variable used in constructing the SES
    score in each percentile bin of the SES distribution.
    The `inc`, `inc_kont` and `highest_edu_pria` variables have been standardized
    to the unit interval [0, 1]; `arblos` is a dummy variable and `crimes`
    is the average number of crimes committed  (which do not exceed 1 in any of the bins).
    """
    geo_data = neighbors.GeoData(year=2000)
    gp = (
        geo_data.neighbors.assign(
            bins=lambda df: pd.cut(
                df.ses,
                bins=list(range(100 + 1)),
                labels=[rf"$SES_{{{i}}}$" for i in range(100)],
                right=False,
            )
        )[["bins"] + neighbors.SES_PCA_COLS]
        .set_index("bins")
        .pipe(
            lambda df: df.assign(
                **{
                    col: minmax_scale_col(df[col])
                    for col in ["inc", "inc_kont", "highest_edu_pria"]
                }
            )
        )
        .reset_index()
        .groupby("bins")
        .mean()
        .rename(columns=pca.PCA_MAP)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = gp.plot(
        ylabel="Average of PCA variable",
        title="Average of PCA features in each SES bin",
        marker="o",
        markersize=3,
        ax=ax,
    )
    fig = ax.get_figure()
    ax.set(
        **{
            "xticks": range(0, 100 + 10, 10),
            "xticklabels": [
                rf"$SES_{{{i}}}$" if i > 0 else "" for i in range(0, 100 + 10, 10)
            ],
        }
    )
    ax.legend(bbox_to_anchor=(1, 0.8))
    fig.tight_layout()
    fig.savefig(str(dstnx.fp.REG_PLOTS / "ses_bin_plot.png"))
    print(f"Saved plot to {dstnx.fp.REG_PLOTS / 'ses_bin_plot.png'}")


if __name__ == "__main__":
    make_plot()
