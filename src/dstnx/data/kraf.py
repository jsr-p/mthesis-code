import click
import pandas as pd

from dstnx import data_utils, db, fp, log_utils

LOGGER = log_utils.get_logger(name=__name__)


def crime_codes(kind: str):
    straffelov = ["10", "11", "12", "13", "14"]
    match kind:
        case "all":
            faerdselslov = ["21", "22", "24", "26"]
            ovrige = ["32", "34", "36", "38"]
            codes = straffelov + faerdselslov + ovrige
        case "sel":
            faerdselslov = ["24"]
            ovrige = ["32", "34"]
            codes = straffelov + faerdselslov + ovrige
        case _:
            raise ValueError
    return codes


def count_crimes(df: pd.DataFrame, kind: str = "sel"):
    """
    Notes:
    See https://www.dst.dk/da/TilSalg/Forskningsservice/Dokumentation/hoejkvalitetsvariable/kriminalitet---afgoerelser/afg-ger7

    Categories:
        To første cifre:

        Straffelov:
        10 Uoplyst straffelov
        11 Seksualforbrydelser'
        12 Voldsforbrydelser
        13 Ejendomsforbrydelser
        14 Andre forbrydelser

        Færdselslov:
        21 Færdselsuheld uspecificeret'
        22 Færdselslov spiritus
        24 Mangler ved køretøj
        26 Færdselslov i øvrigt

        Øvrige særlove:
        32 Lov om euforiserende stoffer
        34 Våbenloven
        36 Skatte- og afgiftslove
        38 Særlove i øvrigt
    """
    codes = crime_codes(kind)
    pattern = "^(?:" + "|".join(codes) + ")"
    return (
        df.astype({"AFG_GER7": str})
        .assign(any_crime=lambda df: df["AFG_GER7"].str.contains(pattern, regex=True))
        .groupby("PERSON_ID")
        .any_crime.sum()
        .to_frame("crimes")
        .pipe(lambda df: df[df.crimes > 0])
        .reset_index()
    )


def load_kraf(year: int, **kwargs):
    return data_utils.load_reg(f"KRAF{year}", fp=fp.REG_DATA / "kraf", **kwargs)


def own_kraf(st, et, **kwargs) -> pd.DataFrame:
    return pd.concat(load_kraf(year, **kwargs) for year in range(st, et))


@click.group()
def cli():
    pass


@cli.command()
def collect_all():
    dst_db = db.DSTDB()
    years = range(1985, 2020 + 1)
    dfs = dict()
    for year in years:
        LOGGER.info(f"Collecting KRAF{year}")
        df = dst_db.extract_data(f"select * from KRAF{year}").pipe(count_crimes)
        dfs[year] = df.assign(year=year)
        data_utils.log_save_pq(
            filename=f"KRAF{year}", df=df, verbose=True, fp=fp.REG_DATA / "kraf"
        )


@cli.command()
@click.option("--suffix", type=str, required=True)
def construct_kraf_cohort(suffix: str):
    st, et = 1992, 1996  # first and last cohort
    base_year = 14
    cohorts = data_utils.load_reg(f"cohorts{suffix}", dtype_backend="numpy_nullable")[
        ["PERSON_ID", "cohort"]
    ]
    all_kraf = []
    for cohort in range(st, et + 1):
        subset_cohorts = (
            cohorts.query(f"cohort == {cohort}")
            .drop(["cohort"], axis=1)
            .copy()
            .merge(
                (
                    own_kraf(
                        cohort + base_year,
                        cohort + base_year + 3 + 1,
                        dtype_backend="numpy_nullable",
                    )
                    .groupby("PERSON_ID")
                    .crimes.sum()
                    .reset_index()
                    .rename(columns={"crimes": "own_crimes"})
                ),
                how="left",
                on="PERSON_ID",
            )
            .fillna(0)
        )
        all_kraf.append(subset_cohorts)
    data_utils.log_save_pq(
        filename=f"own_kraf{suffix}",
        df=pd.concat(all_kraf, axis=0).reset_index(drop=True),
    )


if __name__ == "__main__":
    cli()
