import re

import click
import pandas as pd
import polars as pl
import seaborn as sns

from dstnx import data_utils
from dstnx.features import utils, education, agg_features


def leave_one_out_mean_expr(col: str, group_col: str) -> pl.Expr:
    r"""Computes a polars expression for the leave on out mean.

    Note:
        The equation can be written as :
            $\bar{x}_{-i} := (\bar{x} - x_i / N) * (N / (N - 1))$
    """
    return (pl.col(f"{group_col}_avg") - pl.col(col) / pl.col(f"{group_col}_count")) * (
        pl.col(f"{group_col}_count") / (pl.col(f"{group_col}_count") - 1)
    )


def leave_one_out_mean(
    df: pl.DataFrame, col: str, alias: str, group_id: str
) -> pl.DataFrame:
    r"""Computes the LOO mean"""
    group_avgs = df.groupby(group_id).agg(
        # Compute the count & averages for each group
        [
            pl.col(col).mean().alias(f"{alias}_avg"),
            pl.col(col).count().alias(f"{alias}_count"),
        ]
    )
    return (
        # Merge the counts back and compute the leave one out
        # mean for the all_ses score
        df.join(group_avgs, on=group_id).with_columns(
            leave_one_out_mean_expr(col, alias).alias(alias)
        )
    )


def group_ses(nodes, all_ses, alias="all_ses"):
    group_ses = (
        nodes.join(all_ses, on="PERSON_ID", how="left")
        .pipe(leave_one_out_mean, col="all_ses", group_id="group_id", alias=alias)
        .sort(by="group_id")
    )
    cols = ["PERSON_ID", alias]
    return (
        nodes.join(group_ses, on="group_id", how="left")[cols].groupby("PERSON_ID")
        # Computes the average of the leave one out means of all the
        # school that each individual has attended
        .agg(pl.col(alias).mean())
    )


def transform_own_ses(all_ses: pl.DataFrame) -> pd.DataFrame:
    """Constructs quantile bins for the SES score for each individual."""
    return (
        all_ses.to_pandas()
        .assign(
            SES_q=lambda df: pd.qcut(
                df.all_ses, q=5, labels=[f"SES_Q{i}" for i in range(1, 5 + 1)]
            ),
            SES_q10=lambda df: pd.qcut(
                df.all_ses, q=10, labels=[f"SES_Q{i}0" for i in range(1, 10 + 1)]
            ),
        )
        .pipe(lambda df: df.join(pd.get_dummies(df.SES_q).astype(int)))
        .pipe(lambda df: df.join(pd.get_dummies(df.SES_q10).astype(int)))
        .rename(columns={"all_ses": "own_ses"})
    )


def get_class_qs(cat: str = "not_top"):
    match cat:
        case "q99_below":
            return pl.col("qs") != "top"
        case "q95_below":
            return (pl.col("qs") == "leq90") | (pl.col("qs") == "leq95")
        case "q90_below":
            return pl.col("qs") == "leq90"
        case "no_filter":
            return ""
        case _:
            raise ValueError


def filter_classes(node_classes, class_filter):
    filter_expr = get_class_qs(class_filter)
    if isinstance(filter_expr, pl.expr.expr.Expr):
        return node_classes.filter(filter_expr)
    return node_classes


def _select_cols(node_classes):
    return node_classes.select(
        pl.col(["PERSON_ID", "group_id", "group_count"])
    ).drop_nulls()


@click.command()
@click.option("--suffix", default="", help="Suffix for data files")
def construct_peer_ses(suffix: str):
    re_match = re.compile("(far|mor)_ses")
    parents = data_utils.load_reg(f"parents_features{suffix}", as_pl=True)
    ses_parents = utils.subset_cols(
        parents,
        re_match=re_match,
        re_exclude=re.compile(r"nan_fm"),
        include_cols=["PERSON_ID"],
    )
    all_ses = ses_parents.with_columns(
        ses_parents.select(pl.all().exclude("PERSON_ID")).mean(axis=1).alias("all_ses")
    ).select(pl.col("PERSON_ID", "all_ses"))

    # Load data on nodes and class sizes
    nodes_classes = education.school_types("_class_new", with_counts=True, as_pl=True)

    group_ses_large = group_ses(
        nodes_classes.pipe(_select_cols),
        all_ses,
        alias="all_ses_large",
    )
    group_ses_q99 = group_ses(
        nodes_classes.pipe(filter_classes, class_filter="q99_below").pipe(_select_cols),
        all_ses,
        alias="all_ses_q99",
    )
    group_ses_q95 = group_ses(
        nodes_classes.pipe(filter_classes, class_filter="q95_below").pipe(_select_cols),
        all_ses,
        alias="all_ses_q95",
    )
    group_ses_q90 = group_ses(
        nodes_classes.pipe(filter_classes, class_filter="q90_below").pipe(_select_cols),
        all_ses,
        alias="all_ses_q90",
    )
    group_measures = agg_features.merge_dataframes(
        [group_ses_large, group_ses_q99, group_ses_q95, group_ses_q90],
        on="PERSON_ID",
        how="outer",
    )

    data_utils.log_save_pq(
        filename=f"own_ses{suffix}", df=transform_own_ses(all_ses), verbose=True
    )
    data_utils.log_save_pq(
        filename=f"peer_ses{suffix}", df=group_measures, verbose=True
    )
    # data_utils.log_save_pq(
    #     filename=f"group_ses_class{suffix}", df=group_ses_small, verbose=True
    # )
    # data_utils.log_save_pq(
    #     filename=f"group_ses{suffix}", df=group_ses_large, verbose=True
    # )

    data_utils.log_save_fig(
        filename="peer-parents-ses-dist",
        fig=sns.pairplot(
            group_measures.to_pandas()[["all_ses_large", "all_ses_q90"]]
        ).fig,
    )
    data_utils.log_save_fig(
        filename="own-ses-dist", fig=sns.pairplot(all_ses.to_pandas()[["all_ses"]]).fig
    )


if __name__ == "__main__":
    construct_peer_ses()
