from dataclasses import dataclass
from functools import reduce
from typing import Optional
import os

import click
from tqdm import tqdm
import polars as pl

from dstnx import data_utils, log_utils

LOGGER = log_utils.get_logger(name=__name__)

YEAR_FIRST = 1985
YEAR_LAST = 2019


os.environ["POLARS_MAX_THREADS"] = "12"  # Limit usage


@dataclass
class ColNameConfig:
    """Util config for renaming columns suitably when aggreating."""

    col_suffix: Optional[str] = None
    append_col_suffix: bool = False

    def suffix_from_ages(self, age_s, age_e) -> str:
        ages_specified = self.is_ages_specified(age_s, age_e)
        if ages_specified and not self.append_col_suffix:
            return f"_{age_s}_{age_e}"
        elif not ages_specified and self.col_suffix:
            return self.col_suffix
        elif self.col_suffix and self.append_col_suffix:
            return f"_{age_s}_{age_e}{self.col_suffix}"
        else:
            return f"_{age_s}_{age_e}"

    def expr_alias_from_ages(
        self, value: str, age_s: Optional[int], age_e: Optional[int]
    ) -> str:
        """Returns an alias for a polars expression based on conditions.

        Has become a bit involved; see comments in each case.
        """
        ages_specified = self.is_ages_specified(age_s, age_e)
        if ages_specified and self.col_suffix and not self.append_col_suffix:
            return f"{value}_{age_s}_{age_e}"
        elif ages_specified and not self.col_suffix and not self.append_col_suffix:
            # Append only ages
            return f"{value}_{age_s}_{age_e}"
        elif ages_specified and self.col_suffix and self.append_col_suffix:
            # Append ages and custom col suffix
            return f"{value}_{age_s}_{age_e}{self.col_suffix}"
        elif not ages_specified and self.col_suffix and self.append_col_suffix:
            # Append only col suffix
            return f"{value}{self.col_suffix}"
        else:  # No alias to the expression
            return ""

    def is_ages_specified(self, age_s, age_e) -> bool:
        return isinstance(age_e, int) and isinstance(age_s, int)

    def file_suffix(self) -> str:
        if self.col_suffix:
            return self.col_suffix
        return ""


def merge_dataframes(dataframes, on, how="inner"):
    # Perform inner join using functools.reduce
    merged_df = reduce(lambda left, right: left.join(right, on=on, how=how), dataframes)
    return merged_df


def avg_combined(
    name: str,
    col_config: ColNameConfig,
    wavg: bool = False,
    age_s: Optional[int] = None,
    age_e: Optional[int] = None,
):
    """Combines multiple averages into a single average.

    It does this by multiplying each average by its count, summing these
    products, and dividing by the total count.
    """
    if wavg:
        value_col = f"{name}_wavg"
        count_col = f"{name}_weight_tot"
    else:
        value_col = f"{name}_avg"
        count_col = f"{name}_count"

    expr = (pl.col(value_col) * pl.col(count_col)).sum() / pl.col(count_col).sum()
    if alias := col_config.expr_alias_from_ages(value_col, age_s, age_e):
        LOGGER.debug(f"Alias {alias=} specified for columns:")
        return expr.alias(alias)
    LOGGER.debug(f"No alias specified for columns")
    return expr


def avg_combined_cols(cols: list[str], age, age_end, col_config: ColNameConfig):
    return [
        expr
        for col in cols
        for expr in [
            avg_combined(name=col, age_s=age, age_e=age_end, col_config=col_config),
            avg_combined(
                name=col, age_s=age, age_e=age_end, wavg=True, col_config=col_config
            ),
        ]
    ]


def assign_age_year(data, year: int, year_born: pl.DataFrame):
    return data.join(year_born, on="PERSON_ID").with_columns(
        [(pl.lit(year) - pl.col("year_born")).alias("age")]
    )


def agg_dist_feats(
    df: pl.DataFrame,
    age_s: int,
    age_e: int,
    cols: list[str],
    dist: str,
    col_config: ColNameConfig,
):
    """Aggregates the features for the given age period and distance measure.

    Args:
        df: dataframe to aggregate
        age_s: start age
        age_e: end age
        cols: columns to aggregate
        dist: distance measure (k or radius)

    Returns: aggregated dataframe

    """
    return (
        df.filter((pl.col("age") >= age_s) & (pl.col("age") <= age_e))
        .groupby(["PERSON_ID", dist])
        .agg(avg_combined_cols(cols, age_s, age_e, col_config=col_config))
        .sort(by=["PERSON_ID", dist])
    )


def split_cols_avg(cols):
    """Helper to get _avg cols"""
    return [col.split("_avg")[0] for col in cols if col.endswith("_avg")]


def agg_parents_period(df, age_e, age_s, col_config: ColNameConfig):
    col_suffix = col_config.suffix_from_ages(age_s=age_s, age_e=age_e)
    return (
        df.filter((pl.col("age") >= age_s) & (pl.col("age") <= age_e))
        .groupby(["PERSON_ID"])
        .agg(pl.all().exclude(["age", "year"]).mean().suffix(col_suffix))
    )


def age_periods(period: str) -> list[tuple]:
    """Returns age periods for the given period.

    Notes:
        Default is for the periods:
        0. EC: 0-2 years old
        1. PS: 3-5 years old
        2. ES: 6-9 years old
        3. MS: 10-13 years old
        4. HS: 14-17 years old
    """
    match period:
        case "default":
            return [(0, 2), (3, 5), (6, 9), (10, 13), (14, 17)]
        case "twoyear":
            return [(age, age + 2) for age in range(0, 18, 2)]
        case "oneyear":
            return [(age, age + 1) for age in range(18)]
        case "full":
            return [(0, 17)]
        case _:
            raise ValueError("Wrong age period")


def agg_neighborhood(
    dist: str,
    ages: list[tuple[int, int]],
    year_born: pl.DataFrame,
    suffix: str,
    col_config: ColNameConfig,
):
    # Youth
    df_youth = pl.concat(
        data_utils.load_features(  # type: ignore
            year, year + 1, suffix="_youth", neighbors="youth", dist=dist, polars=True
        )
        for year in tqdm(range(YEAR_FIRST, YEAR_LAST), "loading youth features...")
    )
    cols = split_cols_avg(df_youth.columns)
    LOGGER.debug(f"Columns:\n{cols}\nAges:\n{ages}")
    nodes_youth = merge_dataframes(
        [
            agg_dist_feats(
                df_youth, age_s, age_e, cols, dist=dist, col_config=col_config
            )
            for age_s, age_e in ages
        ],
        on=["PERSON_ID", dist],
    )

    # Adults
    df_adults = pl.concat(
        data_utils.load_features(
            year,
            year + 1,
            suffix="_adults",
            neighbors="neighbors",
            dist=dist,
            polars=True,
        ).pipe(assign_age_year, year, year_born)
        for year in tqdm(range(YEAR_FIRST, YEAR_LAST), "loading adults features...")
    )
    adults_filtered = df_adults.filter(pl.col("age") < 18)
    cols = split_cols_avg(adults_filtered.columns)
    nodes_neighbors = merge_dataframes(
        [
            agg_dist_feats(
                adults_filtered,
                age_s,
                age_e,
                cols=cols,
                dist=dist,
                col_config=col_config,
            )
            for age_s, age_e in ages
        ],
        on=["PERSON_ID", dist],
    )

    # Save
    for df, name in zip(
        [nodes_neighbors, nodes_youth],
        ["adultneigh", "youthneigh"],
    ):
        data_utils.log_save_pq(
            filename=f"{name}_features_{dist}{suffix}{col_config.file_suffix()}",
            df=df,
            verbose=True,
            describe=True,
        )
        # Log ids
        num_ids = df.select(pl.col("PERSON_ID").unique().len()).item()
        LOGGER.debug(f"#Ids: {num_ids} for {name}")


def agg_parents(
    ages: list[tuple[int, int]],
    suffix: str,
    col_config: ColNameConfig,
):
    # Parents
    parents_all = data_utils.load_reg_period(
        start=YEAR_FIRST, end=YEAR_LAST, name="parent_measures_adults", as_pl=True
    )

    parents_nodes = merge_dataframes(
        [
            agg_parents_period(
                parents_all, age_s=age_s, age_e=age_e, col_config=col_config
            )
            for age_s, age_e in ages
        ],
        on=["PERSON_ID"],
    )

    # Log ids
    LOGGER.debug(parents_nodes.select(pl.col("PERSON_ID").unique().len()).item())
    data_utils.log_save_pq(
        filename=f"parents_features{suffix}{col_config.file_suffix()}",
        df=parents_nodes,
        verbose=True,
        describe=True,
    )


@click.command()
@click.argument("agg_type", type=click.Choice(["neighbors", "parents", "all"]))
@click.option(
    "--age-period",
    default="default",
    help="Age period to compute features for",
    type=click.Choice(["default", "twoyear", "full", "oneyear"]),
)
@click.option(
    "--suffix", default="", help="File suffix for the resulting feature files"
)
@click.option(
    "--k-nearest",
    default=False,
    is_flag=True,
    help="To compute k-nearest instead of radius",
    type=bool,
)
@click.option(
    "--col-suffix",
    default=None,
    type=str,
    help="File suffix for the resulting feature files",
)
@click.option(
    "--append-col-suffix",
    is_flag=True,
    default=False,
    help="Append col suffix to the year",
)
def aggregate_features(
    agg_type: str,
    age_period: str,
    suffix: str,
    k_nearest: bool,
    col_suffix: str,
    append_col_suffix: bool,
):
    year_born = data_utils.load_reg("year_born_new", as_pl=True)
    ages = age_periods(age_period)

    if k_nearest:
        dist = "k"
    else:
        dist = "radius"

    col_config = ColNameConfig(col_suffix, append_col_suffix)
    LOGGER.info(
        f"Aggregating features for ({age_period=}, {agg_type=}, "
        f"{k_nearest=}, {col_config=})..."
    )

    match agg_type:
        case "neighbors":
            agg_neighborhood(dist, ages, year_born, suffix, col_config)
        case "parents":
            agg_parents(ages, suffix, col_config)
        case "all":
            agg_neighborhood(dist, ages, year_born, suffix, col_config)
            agg_parents(ages, suffix, col_config)


if __name__ == "__main__":
    aggregate_features()
