from dstnx.utils import results
from dstnx.tables import name_mappings
from dstnx import data_utils
import pandas as pd
import dstnx
from functools import partial
from dstnx.tables import descriptive, name_mappings, interact
from dstnx.utils import tex
from dstnx.tables import regressions
import matplotlib.pyplot as plt
import seaborn as sns

featimps = (
    results.read_featimps()
    .query("fs != 'all_quintile' and fs != 'reduced' and trans != 'impute'")
    .drop(["conduct_optuna_study", "trans", "group_col"], axis=1)
)


subset = featimps.query(
    "name == 'lgbm' and data_suffix == 'k50_oneyeark' and fs == 'all'"
).reset_index(drop=True)
grouped_subset = subset.groupby(
    subset.columns.difference(["model.name", "score"]).tolist()
)
top_10_rows = grouped_subset.apply(lambda x: x.nlargest(15, "score")).reset_index(
    drop=True
)

# mapping
names = subset["model.name"].tolist()
mapping = (
    name_mappings.parse_cols(names, to_ttt=False, to_df=True)
    .assign(orig=names)
    .dropna()
    .query("period != ''")
)
final_mapping = {
    orig: rf"${var}_{{{period}}}$"
    for orig, var, period in zip(mapping.orig, mapping.colname, mapping.period)
} | {"all_ses_large": r"$\overline{SES}$"}

# Assuming your data is in a DataFrame called 'df'
# Sort the DataFrame by target and score
df_sorted = top_10_rows.sort_values(["target", "score"], ascending=[True, False])

# Create a figure with 7 subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(2 * 6, 4 * 4))
axes = axes.flatten()

# Iterate over the unique targets
targets = [
    "gym_grad",
    "eu_grad",
    "us_grad",
    "gym_apply",
    "eu_apply",
    "us_apply",
    "real_neet",
]
# targets = df_sorted['target'].unique()
for i, target in enumerate(targets):
    data = df_sorted[df_sorted["target"] == target].assign(
        score=lambda df: (df.score - df.score.min())
        / (df.score.max() - df.score.min()),
    )
    data["model.name"] = (
        data["model.name"].replace(name_mappings._CONTROL_MAP).replace(final_mapping)
    )
    sns.barplot(x="score", y="model.name", data=data, ax=axes[i])
    target = descriptive.TARGET_NONTT_MAP[target]
    axes[i].set_title(f"{target}")
    if i in [0, 5, 6]:
        axes[i].set_xlabel("Feature Importance Score")
    else:
        axes[i].set_xlabel("")
    if i in [0, 2, 4, 6]:
        axes[i].set(ylabel="Feature")
    else:
        axes[i].set(ylabel="")

axes[-1].set_axis_off()
plt.tight_layout()
data_utils.log_save_fig(fig=fig, filename=f"feature-importance")
