import matplotlib.pyplot as plt
import seaborn as sns

import dstnx
from dstnx.features import trans

PCA_MAP = {
    "highest_edu_pria": "EduLen",
    "arblos": "Arblos",
    "inc": "Inc",
    "inc_kont": "IncKont",
    "crimes": "Crimes",
}


def main():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    df = (
        trans.load_pca_df()
        .astype({"year": int})
        .query("year >= 1992")
        .set_index("year")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = df.rename(columns=PCA_MAP).plot(
        title="Principal component weights and explained variance",
        marker="o",
        xlabel="Year",
        ylabel="Value",
        ax=ax,
    )
    ax.legend(bbox_to_anchor=(1, 0.8))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(str(dstnx.fp.REG_PLOTS / "pca_plot.png"))
    print(f"Saved plot to {dstnx.fp.REG_PLOTS / 'pca_plot.png'}")


if __name__ == "__main__":
    main()
