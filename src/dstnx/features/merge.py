import re
from typing import Optional

import click
import pandas as pd
import polars as pl

import dstnx
from dstnx import data_utils, log_utils
from dstnx.data import grades, psyk
from dstnx.features import address, agg_features, bef, education
from dstnx.features import parents as parents_mod
from dstnx.features import utils
from dstnx.outcomes import job

LOGGER = log_utils.get_logger(name=__name__)

RE_COL_YEARS = re.compile(r"(?P<year_from>\d+)_(?P<year_to>\d+)")
string = "crimes_wavg_14_17"


def _filter_teenage_crime(cols):
    new_cols = []
    for col in cols:
        if not "crime" in col:
            new_cols.append(col)
        else:
            match = RE_COL_YEARS.search(col)
            if match:
                if int(match["year_from"]) >= 13:
                    new_cols.append(col)
    LOGGER.debug(f"Filtered teenage crime; new cols: {new_cols}")
    return new_cols


def save_columns(parents, youth, adult, extra_suffix: str, weighted: bool = False):
    """Saves feature columns to be used later for estimation."""
    if weighted:
        re_match = re.compile("wavg")
        weight_suffix = "_w"
    else:
        re_match = re.compile("[^w]avg")
        weight_suffix = ""
    parent_cols = utils.subset_cols(
        parents,
        re_exclude=re.compile(r"nan_fm|imm"),
        re_match=re.compile("par_.+"),
        exclude_cols=["PERSON_ID"],
    ).columns
    familystatus_cols = utils.subset_cols(
        parents,
        re_exclude=re.compile("^with_parents"),  # Reference category; with parents!
        re_match=re.compile("^(with|not)_.+"),
        exclude_cols=["PERSON_ID"],
    ).columns
    youth_cols = utils.subset_cols(
        youth,
        re_match=re_match,
        exclude_cols=["PERSON_ID"],
    ).columns
    adult_cols = utils.subset_cols(
        adult, re_match=re_match, exclude_cols=["PERSON_ID"]
    ).columns
    columns = {
        "adult": adult_cols,
        "parent": parent_cols,
        "youth": youth_cols,
        "familystatus": familystatus_cols,
    }
    data_utils.log_save_json(
        filename=f"columns{weight_suffix}-{extra_suffix}",
        obj=columns,
        fp=dstnx.fp.DATA / "feature-columns",
    )


def full_file(suffix: str, dist: str, dist_val: int, extra_suffix: str):
    return dstnx.fp.REG_DATA / f"full{suffix}_{dist}{dist_val}{extra_suffix}.parquet"


def full_exists(suffix: str, dist: str, dist_val: int, extra_suffix: str):
    return full_file(suffix, dist, dist_val, extra_suffix).exists()


def overview_table(table: pd.DataFrame, name: str):
    overview = table.groupby("cohort").agg(
        {  # type: ignore
            "imm": ("count", "mean", "sum"),
            "female": ("mean", "sum"),
            "efterskole": ("mean", "sum"),
            "klasse_10": ("mean", "sum"),
        }
    )
    data_utils.log_save_tabulate(
        filename=name, df=overview, fp=dstnx.fp.REG_TABLES / "merge-descriptive"
    )


def table_summary(full_table: pl.DataFrame):
    desc = (
        full_table.groupby("cohort")
        .agg(
            [
                (pl.col("us_grad") == 0).sum().alias("neet_count"),
                (pl.col("us_grad") == 0).mean().alias("neet_avg"),
                pl.col("real_neet").sum().alias("real_neet_count"),
                (pl.col("not_neet") == 0).mean().alias("neet_avg2"),
                pl.col("real_neet").mean(),
                pl.col(["eu_grad", "gym_grad"]).mean(),
                ((pl.col("klasse_11") == 1) & (pl.col("fortidspension") == 0)).mean(),
            ]
        )
        .to_pandas()
        .round(3)
    )
    data_utils.log_save_tabulate(
        filename="desc_fulltable", df=desc, fp=dstnx.fp.REG_TABLES / "merge-descriptive"
    )

    desc_gpa = (
        full_table.groupby(["cohort", "real_neet"])
        .agg([(pl.col("gpa")).mean()])
        .sort(by=["cohort", "real_neet"])
        .to_pandas()
        .round(3)
    )
    data_utils.log_save_tabulate(
        filename="desc_gpa", df=desc_gpa, fp=dstnx.fp.REG_TABLES / "merge-descriptive"
    )

    table = full_table.to_pandas()

    overview_table(table=table, name="overview_table")
    overview_table(table=table.dropna(), name="overview_table_nona")

    overview2 = (
        table.groupby(["cohort", "imm", "female"])
        .agg(
            {  # type: ignore
                "efterskole": ("mean", "sum", "count", "size"),
                "klasse_10": ("mean", "sum", "count", "size"),
                "klasse_10_ne": ("mean", "sum", "count", "size"),
                "klasse_11": ("mean", "sum", "count", "size"),
            }
        )
        .round(3)
    )
    data_utils.log_save_tabulate(
        filename="overview_klasser",
        df=overview2,
        fp=dstnx.fp.REG_TABLES / "merge-descriptive",
    )


def validate_args(radius: Optional[int], k: Optional[int]):
    if (is_radius := isinstance(radius, int)) and isinstance(k, int):
        raise ValueError("Please choose either k or radius")
    if is_radius:
        return "radius", radius
    return "k", k


def load_neighbor_features(suffix: str, dist: str, dist_val: int, col_suffix: str):
    youth = (
        data_utils.load_reg(
            f"youthneigh_features_{dist}{suffix}{col_suffix}", as_pl=True
        )
        .filter(pl.col(dist) == dist_val)
        .drop(dist)
    )
    adult = (
        data_utils.load_reg(
            f"adultneigh_features_{dist}{suffix}{col_suffix}", as_pl=True
        )
        .filter(pl.col(dist) == dist_val)
        .drop(dist)
        .select([pl.col("PERSON_ID"), pl.all().exclude("PERSON_ID").prefix("adults_")])
    )
    return youth, adult


def neet_trans(df: pl.DataFrame, extra_suffix: str = "") -> pl.DataFrame:
    return df.with_columns(
        pl.any(
            pl.col(
                [
                    f"socio_edu{extra_suffix}",
                    f"socio_work{extra_suffix}",
                    f"beskst_work{extra_suffix}",
                ]
            )
        )
        .cast(pl.Int8)
        .alias(f"not_neet{extra_suffix}")
    ).with_columns(
        (
            (pl.col(f"us_grad{extra_suffix}") == 0)
            & (pl.col(f"not_neet{extra_suffix}") == 0)
        )
        .cast(pl.Int8)
        .alias(f"real_neet{extra_suffix}")
    )


def _reduce_parents(parents: pl.DataFrame):
    parents = parents.select(pl.all().exclude("^.*(_new_couple|fm_mark|_imm).*$"))
    par_cols = parents.select(pl.col("^(mor|far).*$")).columns
    new_cols = {col: col.replace("mor_", "").replace("far_", "") for col in par_cols}
    parent_variables = (
        parents.select([pl.col("PERSON_ID"), pl.col("^(mor|far).*$")])
        .melt(id_vars="PERSON_ID")
        .with_columns(pl.col("variable").map_dict(new_cols))
        .groupby(["PERSON_ID", "variable"])
        .mean()
        .pivot(
            index="PERSON_ID",
            columns="variable",
            values="value",
            aggregate_function="mean",
        )
        .select(
            pl.col("PERSON_ID"),
            pl.all().exclude("PERSON_ID").prefix("par_"),  # Add par prefix
        )
    )
    family_status = parents.select(pl.col("PERSON_ID"), pl.col("^(with|not)_.*$"))
    reduced_parents = parent_variables.join(family_status, how="outer", on="PERSON_ID")
    return reduced_parents


@click.command()
@click.option("--suffix", default="", help="Suffix for data files")
@click.option("--radius", default=None, type=int, help="Radius to get features for")
@click.option("--k", default=None, type=int, help="k values to get features for")
@click.option(
    "--force", default=False, is_flag=True, help="Force construction even it exists"
)
@click.option("--save", default=False, is_flag=True, help="Save full")
@click.option(
    "--col-suffix",
    default="",  # NOTE: Equals empty string and not None here; fix later ;)
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
    "--all-feats",
    default=False,
    is_flag=True,
    type=bool,
    help="File suffix for features",
)
def construct_full(
    suffix: str,
    radius: int,
    k: int,
    force: bool = False,
    save: bool = True,
    col_suffix: str = "",
    feature_suffix: str = "",
    all_feats: bool = False,
):
    dist, dist_val = validate_args(radius, k)

    extra_suffix = f"{col_suffix}{feature_suffix}"

    # Return dataframe if it already has been created (and force-mode not enabled)
    if full_exists(suffix, dist, dist_val, extra_suffix) and not force:
        return data_utils.load_reg(f"full{suffix}_{dist}{dist_val}{extra_suffix}")

    LOGGER.info(f"Constructing full merge data for ({suffix=}, {dist=}, {dist_val=})")

    # Load neighborhood features
    youth, adult = load_neighbor_features(suffix, dist, dist_val, col_suffix)
    koms = address.get_kom(suffix, as_pl=True)

    # parent measures
    parents = data_utils.load_reg(f"parents_features{suffix}{col_suffix}", as_pl=True)

    if not all_feats:
        youth = youth.select(pl.col(["PERSON_ID"]), pl.col("^(ses|crime).*$"))
        youth = youth[_filter_teenage_crime(youth.columns)]  # Filter teenage crime
        adult = adult.select(pl.all().exclude("^.*imm.*$"))
        parents = _reduce_parents(parents)

    # Outcomes
    rel_cols = ["PERSON_ID", "eu", "gym", "gs", "eg", "us"]
    outcomes_grad = data_utils.load_reg(f"outcomes_grad{suffix}", as_pl=True)[
        rel_cols
    ].select([pl.col("PERSON_ID"), pl.all().exclude("PERSON_ID").suffix("_grad")])
    outcomes_apply = data_utils.load_reg(f"outcomes_apply{suffix}", as_pl=True)[
        rel_cols
    ].select([pl.col("PERSON_ID"), pl.all().exclude("PERSON_ID").suffix("_apply")])

    outcomes_grad20 = data_utils.load_reg(f"outcomes_grad{suffix}_y20", as_pl=True)[
        rel_cols
    ].select([pl.col("PERSON_ID"), pl.all().exclude("PERSON_ID").suffix("_grad_y20")])
    outcomes_apply20 = data_utils.load_reg(f"outcomes_apply{suffix}_y20", as_pl=True)[
        rel_cols
    ].select([pl.col("PERSON_ID"), pl.all().exclude("PERSON_ID").suffix("_apply_y20")])
    jobs = job.get_jobs_df(suffix, as_pl=True)
    jobs20 = job.get_jobs_df(suffix, extra_suffix="_y20", as_pl=True)

    # Cohort BEF variables
    rel_cols = ["PERSON_ID", "cohort", "IE_TYPE", "KOEN"]
    cohorts = (
        data_utils.load_reg(f"cohorts{suffix}", as_pl=True)[rel_cols]
        .unique(subset=rel_cols)
        .pipe(bef.assign_koen)
        .pipe(bef.assign_imm_pl)
    )

    # School dummies
    school_type_dummies = education.school_type_dummies(suffix).pipe(
        data_utils.pd_to_pl
    )
    inst9thgrade = data_utils.load_reg(f"inst9thgrade{suffix}", as_pl=True).drop("YEAR")

    # SES scores for self & peers
    peer_ses = data_utils.load_reg(f"peer_ses{suffix}", as_pl=True)
    own_ses = data_utils.load_reg(f"own_ses{suffix}", as_pl=True)

    # psyk & crime
    own_kraf = data_utils.load_reg(f"own_kraf{suffix}", as_pl=True)
    psykdata = data_utils.pd_to_pl(
        # Highest agg.
        psyk.psyk_dummies(suffix, spec=False)
    )

    df = agg_features.merge_dataframes(
        [
            cohorts,
            peer_ses,
            own_ses,
            youth,
            adult,
            parents,
            school_type_dummies,
            koms,
            inst9thgrade,
            psykdata,
            own_kraf,
        ],
        on="PERSON_ID",
        how="left",
    ).with_columns(
        # Fill null in columns where missing mean 0
        [
            pl.col([*psyk.NON_SPEC_ABV, "own_crimes", "any_psyk"]).fill_null(0),
        ]
    )
    gpa = grades.load_unique(suffix, as_pl=True, pivot=True, one_avg=True)
    full_table = (
        agg_features.merge_dataframes(
            [
                df,
                jobs,
                jobs20,
                outcomes_grad,
                outcomes_apply,
                outcomes_grad20,
                outcomes_apply20,
                gpa,
            ],
            on="PERSON_ID",
            how="left",
        )
        .pipe(neet_trans)
        .pipe(neet_trans, extra_suffix="_y20")
        .pipe(job.filter_fortidspension)
    )

    if save:
        filesuffix = f"{dist}{dist_val}{extra_suffix}"
        data_utils.log_save_pq(
            filename=f"full{suffix}_{filesuffix}",
            df=full_table,
            describe=True,
            verbose=True,
        )
        save_columns(parents, youth, adult, filesuffix, weighted=True)
        save_columns(parents, youth, adult, filesuffix, weighted=False)
        table_summary(full_table)
    return full_table


def load_full(
    suffix: str, radius: int, k: int, extra_suffix: str, force: bool = False, **kwargs
):
    dist, dist_val = validate_args(radius, k)
    if full_exists(suffix, dist, dist_val, extra_suffix) and not force:
        return data_utils.load_reg(
            f"full{suffix}_{dist}{dist_val}{extra_suffix}", **kwargs
        )
    else:
        # construct_full(suffix, radius, force, save)
        raise ValueError("Not implemented yet")


if __name__ == "__main__":
    construct_full()
