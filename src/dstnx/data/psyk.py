import re

import pandas as pd

import dstnx
from dstnx import data_utils, db


def proc_cats(df):
    pattern = r"^(.*?)\s+\[(.*?)\]$"
    cats = df.query("Kode.isna()").Tekst.tolist()
    data = []
    for text in cats:
        match = re.match(pattern, text)
        if match:
            first_text = match.group(1)
            square_brackets_text = match.group(2)
            data.append((first_text, square_brackets_text))
    data = pd.DataFrame(data, columns=["tekst", "cat"]).iloc[1:]
    return data


def construct_cat_map(data):
    """Maps each DF00, DF01, ..., DF99 to its category"""
    texts = data.tekst.tolist()[:-1]
    return {
        cat: text
        for j, text in enumerate(texts)
        for cat in [f"DF{i:02}" for i in range(j * 10, (j + 1) * 10)]
    }


def construct_codes_df(df, cat_map):
    codes = df.query("Kode.notna()").assign(
        spec=lambda df: df.Kode.apply(len),
        kode4=lambda df: df.Kode.str[:4],
        cat=lambda df: df.kode4.map(cat_map),
    )
    return codes


def proc_psyk_pcodes():
    df = pd.read_table(dstnx.fp.DATA / "other-dst" / "psykkoder.txt")
    data = proc_cats(df)
    cat_map = construct_cat_map(data)
    codes = construct_codes_df(df, cat_map)
    data_utils.log_save_json(filename="psyk_cat_map", obj=cat_map)
    data_utils.log_save_pq(filename="psyk_codes", df=codes)


QUERY_PSYK = """
select diag.C_DIAG, diag.C_DIAGTYPE, adm.C_ADIAG, 
adm.PERSON_ID, adm.D_UDDTO, adm.D_INDDTO from d222008.PSYKDIAG diag
inner join d222008.PSYKADM adm
on diag.RECNUM = adm.RECNUM
"""


def query_psyk(suffix: str):
    """Queries psykdata for cohorts.

    While I do not have write access to LPR I query all obs and
    then subset by the cohort ids because I cannot create a temp table
    in the database.
    """
    cohorts = data_utils.load_reg(f"cohorts{suffix}")
    dst_db = db.DSTDB(dsn="STATPROD.world", proxy=False, database=None)
    psyk_all = dst_db.extract_data(QUERY_PSYK)
    psyk_cohort = psyk_all.query(
        "PERSON_ID in @cohorts.PERSON_ID.unique()"
    ).reset_index(drop=True)

    # Transform
    cat_map = data_utils.load_json(dstnx.fp.REG_OUTPUT / "psyk_cat_map.json")
    codes = data_utils.load_reg("psyk_codes")
    all_map = dict(zip(codes.Kode, codes.Tekst))
    psyk_cohort = (
        psyk_cohort.assign(
            spec=lambda df: df.C_DIAG.apply(len),
            kode4=lambda df: df.C_DIAG.str[:4],
            cat=lambda df: df.kode4.map(cat_map),
            spec_cat=lambda df: df.kode4.map(all_map),
        )
        .pipe(lambda df: df[df.kode4.str.startswith("DF")])
        .reset_index(drop=True)
    )
    data_utils.log_save_pq(filename=f"psyk_cohort{suffix}", df=psyk_cohort)


def join_col(col):
    """Helper for joining col below."""
    # To avoid encoding error in R
    col = (
        col.replace("-", "")
        .replace(",", "")
        .replace("æ", "ae")
        .replace("ø", "oe")
        .replace("å", "aa")
    )
    return "_".join([s.lower() for s in col.split(" ")])


NON_SPEC = [
    "Adfærds- og følelsesmæssige forstyrrelser sædvanligvis opstået i barndom eller adolescens",
    "Adfærdsændringer forbundet med fysiologiske forstyrrelser og fysiske faktorer",
    "Affektive sindslidelser",
    "Forstyrrelser i personlighedsstruktur og adfærd i voksenalderen",
    "Mental retardering",
    "Nervøse og stress-relaterede tilstande samt tilstande med psykisk betingede legemlige symptomer",
    "Organiske inklusive symptomatiske psykiske lidelser",
    "Psykiske lidelser og adfærdsmæssige forstyrrelser forårsaget af brug af psykoaktive stoffer",
    "Psykiske udviklingsforstyrrelser",
    "Skizofreni, skizotypisk sindslidelse, paranoide psykoser, akutte og forbigående psykoser samt skizoaffektive psykoser",
]

NON_SPEC_ABV = [
    f"{join_col(cat[:20])}" + "_" * (len(cat) > 20) for cat in NON_SPEC
]  # Cut-off at 20 char
NON_SPEC_MAP = dict(zip(NON_SPEC, NON_SPEC_ABV))


def _load_psyk(suffix):
    """Avoid bad control"""
    suffix = "_new"
    cohorts = data_utils.load_reg(f"cohorts{suffix}")
    psyk = data_utils.load_reg(f"psyk_cohort{suffix}")
    return (
        psyk.merge(cohorts[["PERSON_ID", "cohort"]], how="left", on="PERSON_ID")
        .assign(age_diag=lambda df: (df.D_INDDTO.dt.year - df.cohort).astype(int))
        .query("age_diag <= 15")
        .reset_index(drop=True)
    )


def psyk_dummies(suffix, spec: bool = False):
    """Loads psyk dummies.

    Args:
        spec: If True returns more specific category dummies.
    """
    psyk = _load_psyk(suffix)
    if spec:
        col = "spec_cat"
    else:
        col = "cat"
        psyk[col] = psyk[col].map(NON_SPEC_MAP)
    return (
        psyk.assign(val=1)
        .pivot_table(index="PERSON_ID", values="val", columns=col, fill_value=0)
        .assign(any_psyk=lambda df: df.any(axis=1).astype(int))
        .reset_index()
    )


def psyk_dummy(suffix) -> pd.DataFrame:
    return (
        psyk_dummies(suffix, spec=False).any(axis=1).to_frame("psyk_diag").reset_index()
    )


if __name__ == "__main__":
    proc_psyk_pcodes()
    query_psyk("_new")
