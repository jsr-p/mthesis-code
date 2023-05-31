import pandas as pd
from dstnx.utils import results
from dstnx.tables import name_mappings
import dstnx
from dstnx import data_utils
from functools import partial
from dstnx.tables import descriptive, name_mappings, interact
from dstnx.utils import tex

mlresults: pd.DataFrame = results.read_ml_results().query(
    "fs != 'all_quintile' and fs != 'reduced' and trans != 'impute'"
)
MODEL_MAP = {
    model: name_mappings.to_texttt(model_cap)
    for model, model_cap in zip(["logreg", "lgbm"], ["LogReg", "LGBM"])
}
FSETSMAP = {
    "all": "All",
    "all_exccontrols": r"\%Controls",
    "all_excfamily": r"\%Family",
    "all_excneighborhood": r"\%Neigh",
    "controls_only": r"ControlsOnly", 
    "fam_only": r"FamilyOnly", 
    "neighborhood_only": r"NeighOnly", 
    "gpa_only": r"GPAOnly", 
}
DMAP = {
    "radius200_oneyearradius": r"$r=200$",
    "k50_oneyeark": r"$k=50$",
}
ml = (
    mlresults.assign(
        fs=lambda df: pd.Categorical(df.fs, categories=list(FSETSMAP.keys()), ordered=True)
    )
    .melt(
        id_vars=["name", "target", "fs", "data_suffix", "trans"],
        value_vars=["ROC AUC Score"],
        var_name="measure",
        value_name="scores",
    )
    # .pivot(index=["fs"], columns=["target", "measure", "name"], values=["scores"])
    .pivot(index=["fs", "data_suffix"], columns=["target", "name"], values=["scores"])
    .round(2)
    .pipe(interact.strip_multiindex)
    .droplevel(0, axis=1)
)


def create_tab(ml, cols, name):
    tab = (
        ml[cols]
        .rename(columns=descriptive.TARGET_MAP_SPEC | MODEL_MAP, index=FSETSMAP | DMAP)
        .style.format(lambda val: round(val, 2))
        .to_latex(
            multirow_align="l",
            multicol_align="c",
            column_format=2 * "l" + len(cols) * 2 * "c",
        )
    )
    data_utils.log_save_txt(
        filename=name,
        suffix=".tex",
        text=tex.compose(
            tab,
            tex_fns=[
                tex.add_rules,
                partial(
                    tex.add_column_midrules,
                    add_row_midrule=False,
                    num_idx=2,
                    num_subcols=2,
                    num_cols=len(cols),
                ),
            ],
        ),
        fp=dstnx.fp.REG_OUTPUT / "tex_tables",
    )


create_tab(ml, ["gym_grad", "eu_grad", "us_grad", "real_neet"], "ml-results-grad")
create_tab(ml, ["gym_apply", "eu_apply", "us_apply"], "ml-results-apply")
