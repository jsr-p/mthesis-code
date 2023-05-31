"""
Helpers for getting data ready for R e.g. fixest
"""

import itertools as it
from pathlib import Path

import click

import dstnx
from dstnx import data_utils, log_utils
from dstnx.features import merge
from dstnx.features import utils as feat_utils
from dstnx.models import feat_sets

LOGGER = log_utils.get_logger(name=__name__)


def fixest_effects(group_col: str) -> str:
    return f" | {group_col} + cohort + {group_col}[year]"


def interact_term(prod: str, cats: list[str]) -> str:
    return " + ".join(
        [
            f"{prod} * {cat} \n" if i % 2 == 0 else f"{prod} * {cat}"
            for i, cat in enumerate(cats, 1)
        ]
    )


R_BASIC = """library("fixest")
library(arrow)
library("magrittr")
library(tidyverse)

fe_res_table <- function(res) {{
  cints <- confint(res)
  results <- data.frame(res$coeftable) 
  results$lower <- cints[, 1]
  results$upper <- cints[, 2]
  results$N <- res$nobs
  return (results)
}}

df <- read_parquet("{DATA_PATH}") %>% 
  drop_na() %>% 
  mutate(
      year = cohort,
      cohort = as.factor(cohort),
      KOM = as.factor(KOM),
      INSTNR = as.factor(INSTNR)
  )
"""

FIXEST_FORM = """reg <- feols(
  fml = as.formula(paste(readLines("{FORM_FILE}"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~{GROUP_COL})
print(res)
write.csv(fe_res_table(res), file = "{OUT_FILE}")
"""


def fixest_formula(
    form_file: Path, group_col: str, fp: Path = dstnx.fp.REG_TABLES
) -> str:
    out_file = (fp / form_file.stem).with_suffix(".csv")
    return FIXEST_FORM.format(
        FORM_FILE=form_file.as_posix(),
        GROUP_COL=group_col,
        OUT_FILE=out_file.as_posix(),
    )


@click.command()
@click.option(
    "--suffix", default="", help="File suffix for the resulting feature files"
)
@click.option("--radius", default=None, type=int, help="Radius to get features for")
@click.option("--k", default=None, type=int, help="k values to get features for")
@click.option(
    "--col-suffix",
    default=None,
    type=str,
    help="File suffix for neighbor measures",
)
@click.option(
    "--feature-suffix",
    default="",
    type=str,
    help="File suffix for features",
)
@click.option(
    "--deciles",
    default=False,
    type=bool,
    is_flag=True,
    help="Use deciles instead of quintiles",
)
def prepare_metadata(
    suffix: str,
    radius: int,
    k: int,
    col_suffix: str,
    feature_suffix: str,
    deciles: bool,
):
    extra_suffix = f"{col_suffix}{feature_suffix}"
    dist, dist_val = merge.validate_args(radius, k)
    if not (file := merge.full_file(suffix, dist, dist_val, extra_suffix)).exists():
        raise ValueError("File does not exist!")

    if deciles:
        interact_suffix = "-deciles"
        peer_group_cols = feat_sets.SES_Q_DEC_FEATS
    else:
        peer_group_cols = feat_sets.SES_Q_FEATS
        interact_suffix = "-quintiles"

    files = []
    filesuffix = f"{dist}{dist_val}{extra_suffix}"
    for name, cols in zip(
        ["w", "nw"],
        [
            data_utils.load_json(
                dstnx.fp.DATA / "feature-columns" / f"columns_w-{filesuffix}.json"
            ),
            data_utils.load_json(
                dstnx.fp.DATA / "feature-columns" / f"columns-{filesuffix}.json"
            ),
        ],
    ):
        for feat_case in ["reduced", "all"]:
            targets = [
                "eu_grad",
                "gym_grad",
                "us_grad",
                "eu_apply",
                "gym_apply",
                "us_apply",
                "real_neet",
            ]
            for target_col, group_col, peer_col in it.product(
                targets, feat_sets.GROUP_FEATS, feat_sets.PEER_FEATS
            ):
                feat_cols = feat_utils.cols_picker(cols, feat_case)
                features = (
                    feat_cols
                    + feat_sets.EXTRA_FEATS
                    + peer_group_cols
                    + feat_sets.PSYK_ANY
                    + [peer_col]
                )

                filename = (
                    dstnx.fp.DATA
                    / "formulas"
                    / (
                        f"{target_col}-{dist}{col_suffix}-{name}-{group_col}"
                        f"-{peer_col}-{extra_suffix}-{feat_case}{interact_suffix}.txt"
                    )
                )
                with open(filename, "w") as script:
                    script.write(f"{target_col} ~ ")
                    for i, feat in enumerate(features, start=1):
                        if i == 1:
                            script.write(f"{feat}")
                        elif i % 3 == 0:
                            script.write(f"\n + {feat}")
                        else:
                            script.write(f" + {feat}")
                    script.write("\n + " + interact_term(peer_col, peer_group_cols))
                    script.write(fixest_effects(group_col) + "\n")
                files.append((filename, group_col))
                LOGGER.info(f"Saved R formula to {filename}")

    # Construct R-script
    filename = (
        dstnx.fp.SCRIPTS / "R" / f"fe_{dist}{dist_val}{extra_suffix}{interact_suffix}.R"
    )
    with open(filename, "w") as script:
        reg_folder = dstnx.fp.REG_TABLES / filename.stem
        Path.mkdir(reg_folder, exist_ok=True)
        script.write(R_BASIC.format(DATA_PATH=file.as_posix()) + "\n")
        for file, group_col in files:
            print(file, group_col)
            script.write(fixest_formula(file, group_col, fp=reg_folder) + "\n\n")
    LOGGER.info(f"Script saved to {filename}")


if __name__ == "__main__":
    prepare_metadata()
