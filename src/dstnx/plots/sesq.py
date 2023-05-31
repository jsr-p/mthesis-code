import seaborn as sns
import dstnx
import matplotlib.pyplot as plt
from dstnx.models import model_prep
from dstnx import data_utils

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

SESQ_MAP = {f"SES_Q{i}": rf"$Q_{{{i}}}^{{SES}}$" for i in range(1, 6)} | {
    f"SES_Q{i}0": rf"$Q_{{{i}0}}^{{SES}}$" for i in range(1, 11)
}
CLASS_MAP = {
    "all_ses_small": r"$\overline{{SES}}_{{-is}}$",
    "all_ses_large": r"$\overline{{SES}}_{{-is}}$",
}


if __name__ == "__main__":
    feats = ["all_ses_small", "SES_q"]
    data = (
        data_utils.load_reg("full_new_k50_all_all", dtype_backend="numpy_nullable")[
            feats
        ]
        .pipe(model_prep.convert_non_nullable)
        .dropna()
        .assign(
            SES_q=lambda df: df.SES_q.map(SESQ_MAP),
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.violinplot(data=data, y="all_ses_small", x="SES_q", ax=ax)
    ax.set(ylabel=CLASS_MAP["all_ses_small"], xlabel="Quantiles of SES distribution")
    fig.tight_layout()
    data_utils.log_save_fig(fig=fig, filename="sesqdist")
