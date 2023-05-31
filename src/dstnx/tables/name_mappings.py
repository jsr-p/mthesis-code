import re

import pandas as pd

from dstnx.models import feat_sets

RE_COL_YEARS = re.compile(r"(?P<colname>\w+)_(?P<year_from>\d+)_(?P<year_to>\d+)")

PEER_COLS = feat_sets.PEER_FEATS
GROUP_COLS = feat_sets.SES_Q_DEC_FEATS + feat_sets.SES_Q_FEATS
CONTROLS = feat_sets.PSYK_ANY + feat_sets.EXTRA_FEATS


def to_texttt(name: str):
    return rf"\texttt{{{name}}}"


PERIOD_MAP = {"0-2": "EC", "3-5": "PS", "6-9": "ES", "10-13": "MS", "14-17": "HS"}

_CONTROL_MAP = {
    "any_psyk": "Psych",
    "klasse_10": "10thGrade",
    "klasse_11": "11thGrade",
    "efterskole": "BoardingSchool",
    "imm": "Imm.",
    "gpa": "GPA",
    "female": "Female",
    "own_crimes": "OwnTeenCrimes",
}
CONTROL_MAP = {k: to_texttt(v) for k, v in _CONTROL_MAP.items()}


VARIABLES = [
    "highest_edu_pria",
    "arblos",
    "inc",
    "inc_kont",
    "crimes",
    "highest_gs",
    "highest_eu",
    "ses",
    "klasse_10",
    "klasse_11",
    "efterskole",
    "imm",
    "gpa",
    "female",
    "own_crimes",
    "with_parents",
    "with_mom_alone",
    "with_dad_alone",
    "not_lives_with_parents",
]


def parse_colname(colname: str):
    return "".join([s.capitalize() for s in colname.split("_")])


def extract_match(match, col, prefix, to_ttt: bool = True):
    if match:
        name = f"{prefix}{parse_colname(match['colname'])}"
        data = {
            "colname": to_texttt(name) if to_ttt else name,
            "period": f'{match["year_from"]}-{match["year_to"]}',
        }
    else:
        raise AssertionError(f"{col} did not have a match!")
    return data


def parse_cols(cols, to_df: bool = False, to_ttt: bool = True):
    parsed_cols = []
    for col in cols:
        if col.startswith("par"):
            col = col.replace("par_", "")
            match = RE_COL_YEARS.search(col)
            data = extract_match(match, col, prefix="Par", to_ttt=to_ttt)
        elif col.startswith("adults"):
            col = col.replace("adults_", "").replace("_avg", "").replace("_wavg", "")
            match = RE_COL_YEARS.search(col)
            data = extract_match(match, col, prefix="", to_ttt=to_ttt)
        elif col.startswith("with") or col.startswith("not"):
            match = RE_COL_YEARS.search(col)
            data = extract_match(match, col, prefix="", to_ttt=to_ttt)
        elif col.startswith("ses") or col.startswith("crimes"):
            col = "youth_" + col.replace("_avg", "").replace("_wavg", "")
            match = RE_COL_YEARS.search(col)
            data = extract_match(match, col, prefix="", to_ttt=to_ttt)
        else:
            parsed_cols.append({"colname": "", "period": ""})
            continue
        parsed_cols.append(data)
    if to_df:
        parsed_cols = pd.DataFrame(parsed_cols)
    return parsed_cols


VARS = [
    "par_highest_edu_pria_0_2",
    "par_arblos_0_2",
    "par_inc_kont_0_2",
    "par_crimes_0_2",
    "par_highest_edu_pria_3_5",
    "par_arblos_3_5",
    "par_inc_kont_3_5",
    "par_crimes_3_5",
    "par_highest_edu_pria_6_9",
    "par_arblos_6_9",
    "par_inc_kont_6_9",
    "par_crimes_6_9",
    "par_highest_edu_pria_10_13",
    "par_arblos_10_13",
    "par_inc_kont_10_13",
    "par_crimes_10_13",
    "par_highest_edu_pria_14_17",
    "par_arblos_14_17",
    "par_inc_kont_14_17",
    "par_crimes_14_17",
    "adults_highest_edu_pria_avg_0_2",
    "adults_arblos_avg_0_2",
    "adults_inc_avg_0_2",
    "adults_inc_kont_avg_0_2",
    "adults_crimes_avg_0_2",
    "adults_highest_edu_pria_avg_3_5",
    "adults_arblos_avg_3_5",
    "adults_inc_avg_3_5",
    "adults_inc_kont_avg_3_5",
    "adults_crimes_avg_3_5",
    "adults_highest_edu_pria_avg_6_9",
    "adults_arblos_avg_6_9",
    "adults_inc_avg_6_9",
    "adults_inc_kont_avg_6_9",
    "adults_crimes_avg_6_9",
    "adults_highest_edu_pria_avg_10_13",
    "adults_arblos_avg_10_13",
    "adults_inc_avg_10_13",
    "adults_inc_kont_avg_10_13",
    "adults_crimes_avg_10_13",
    "adults_highest_edu_pria_avg_14_17",
    "adults_arblos_avg_14_17",
    "adults_inc_avg_14_17",
    "adults_inc_kont_avg_14_17",
    "adults_crimes_avg_14_17",
    "adults_highest_gs_avg_0_2",
    "adults_highest_eu_avg_0_2",
    "adults_highest_gs_avg_3_5",
    "adults_highest_eu_avg_3_5",
    "adults_highest_gs_avg_6_9",
    "adults_highest_eu_avg_6_9",
    "adults_highest_gs_avg_10_13",
    "adults_highest_eu_avg_10_13",
    "adults_highest_gs_avg_14_17",
    "adults_highest_eu_avg_14_17",
    "with_parents_0_2",
    "with_mom_alone_0_2",
    "with_dad_alone_0_2",
    "not_lives_with_parents_0_2",
    "with_parents_3_5",
    "with_mom_alone_3_5",
    "with_dad_alone_3_5",
    "not_lives_with_parents_3_5",
    "with_parents_6_9",
    "with_mom_alone_6_9",
    "with_dad_alone_6_9",
    "not_lives_with_parents_6_9",
    "with_parents_10_13",
    "with_mom_alone_10_13",
    "with_dad_alone_10_13",
    "not_lives_with_parents_10_13",
    "with_parents_14_17",
    "with_mom_alone_14_17",
    "with_dad_alone_14_17",
    "not_lives_with_parents_14_17",
    "ses_avg_0_2",
    "ses_avg_3_5",
    "ses_avg_6_9",
    "ses_avg_10_13",
    "ses_avg_14_17",
    "crimes_avg_14_17",
    "klasse_10",
    "klasse_11",
    "efterskole",
    "imm",
    "gpa",
    "female",
    "own_crimes",
    "SES_Q1",
    "SES_Q2",
    "SES_Q3",
    "SES_Q4",
    "any_psyk",
    "all_ses_large",
    "SES_Q1:all_ses_large",
    "SES_Q2:all_ses_large",
    "SES_Q3:all_ses_large",
    "SES_Q4:all_ses_large",
]


parsed_cols = parse_cols(VARS)
df = pd.DataFrame(parsed_cols)
