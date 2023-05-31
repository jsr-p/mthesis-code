import pandas as pd

from dstnx import data_utils, dst_tools


class ColMapper:
    def __init__(self):
        self.instnr = dst_tools.get_mapping("instnr", dst_id="HQ").mapping
        self.audd_obj = data_utils.DISCEDMapping("audd")
        self.audd = self.audd_obj.get_mapping(niveau=5)
        self.udd_obj = data_utils.DISCEDMapping("udd")
        self.udd = self.udd_obj.get_mapping(niveau=5)

    def map_all(self, df):
        return df.assign(
            audd_name=lambda df: df.AUDD.map(self.audd),
            udd_name=lambda df: df.UDD.map(self.udd),
            instname=lambda df: df.INSTNR.astype(int).map(self.instnr),
        )

    def map_group_ids(self, df):
        return df.assign(
            audd_name=lambda df: df.AUDD.map(self.audd),
            instname=lambda df: df.INSTNR.astype(int).map(self.instnr),
        )


EFTERSKOLE = [
    "8. klasse, efterskole",
    "9. klasse, efterskole",
    "10. klasse, efterskole",
    "11. klasse, efterskole",
]
KLASSE_10 = ["10. klasse, efterskole", "10. klasse"]
KLASSE_10_NOT_EFTERSKOLE = ["10. klasse"]
KLASSE_11 = ["11. klasse", "11. klasse, efterskole"]
SCHOOL_DUMMIES = ["klasse_10", "klasse_10_ne", "klasse_11", "efterskole"]
KLASSE_89 = [
    "8. klasse",
    "8. klasse, efterskole",
    "9. klasse",
    "9. klasse, efterskole",
]


def map_audd_cols(df: pd.DataFrame) -> pd.DataFrame:
    cats = [
        "10. klasse",
        "10. klasse, efterskole",
        "11. klasse",
        "11. klasse, efterskole",
        "8. klasse",
        "8. klasse, efterskole",
        "9. klasse",
        "9. klasse, efterskole",
    ]
    assert sorted(df.audd_name.unique()) == cats
    return df.assign(
        klasse_10=lambda df: df.audd_name.isin(KLASSE_10).astype(int),
        klasse_10_ne=lambda df: df.audd_name.isin(KLASSE_10_NOT_EFTERSKOLE).astype(int),
        klasse_11=lambda df: df.audd_name.isin(KLASSE_11).astype(int),
        efterskole=lambda df: df.audd_name.isin(EFTERSKOLE).astype(int),
    )


# ----------------------- SOCIO -------------------------- #

SOCIO_WORK = [
    "Selvstændig",
    "Selvstændig, 10 eller flere ansatte",
    "Selvstændig, 5 - 9 ansatte",
    "Selvstændig, 1 - 4 ansatte",
    "Selvstændig, ingen ansatte",
    "Medarbejdende ægtefælle",
    "Lønmodtager med ledelsesarbejde",
    "Lønmodtager i arbejde der forudsætter færdigheder på højeste niveau",
    "Lønmodtager i arbejde der forudsætter færdigheder på mellemniveau",
    "Lønmodtager i arbejde der forudsætter færdigheder på grundniveau",
    "Andre lønmodtagere",
    "Lønmodtager, stillingsangivelse ikke oplyst",
]

SOCIO_NOT = [
    "Arbejdsløs mindst halvdelen af året",
    "Modtager af sygedagpenge, uddannelsesgodtgørelse, orlovsydelser mm.",
    "Førtidspensionister",
    "Efterlønsmodtager mv.",
    "Kontanthjælpsmodtager",
]

SOCIO_EDU = [
    "Under uddannelse, inkl.skoleelever min. 15 år",
]

# ----------------------- BESKST -------------------------- #

# Taken from: list(dst_tools.get_mapping("BESKST13").mapping.values())
BESKST_WORK = [
    "Selvstændig",
    "Medarbejdende ægtefælle",
    "Lønmodtager og ejer af virksomhed",
    "Lønmodtager",
    "Lønmodtager med understøttelse",
]
BESKST_NOT = [
    "Efterlønsmodtager",
    "Arbejdsløs mindst halvdelen af året (nettoledighed)",
    "Modtager af dagpenge (aktivering og lign.,sygdom, barsel og orlov)",
    "Kontanthjælpsmodtager",
]


# ----------------------- MAPPING -------------------------- #


JOB_COLS = [
    "socio_edu",
    "socio_work",
    "socio_not",
    "beskst_work",
    "beskst_not",
    "fortidspension",
]


def map_job_cols(df: pd.DataFrame) -> pd.DataFrame:
    disco = data_utils.DSTNestedMapping(table_name="disco", convert_cols=False)
    branche = data_utils.DSTNestedMapping(table_name="branche", convert_cols=False)
    return df.assign(
        socio=lambda df: df.SOCIO13.map(dst_tools.get_mapping("SOCIO13").mapping),
        beskst=lambda df: df.BESKST13.map(dst_tools.get_mapping("BESKST13").mapping),
        branche=lambda df: df.ARB_HOVED_BRA_DB07.map(
            branche.get_mapping(keys_to_int=True)
        ),
        disco=lambda df: df.DISCO_KODE.map(disco.get_mapping(keys_to_str=True)),
        socio_edu=lambda df: df.socio.isin(SOCIO_EDU).astype(int),
        socio_work=lambda df: df.socio.isin(SOCIO_WORK).astype(int),
        socio_not=lambda df: df.socio.isin(SOCIO_NOT).astype(int),
        beskst_work=lambda df: df.beskst.isin(BESKST_WORK).astype(int),
        beskst_not=lambda df: df.beskst.isin(BESKST_NOT).astype(int),
        fortidspension=lambda df: df.socio.isin(["Førtidspensionister"]).astype(int),
    )


if __name__ == "__main__":
    col_mapper = ColMapper()
