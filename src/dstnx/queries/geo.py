"""
Module to query geodata from DST.
The GEO string does the following:
- Queries unique addreses that person lived at in a given 
year with coordinates and selected features
- The features change name over time; so a wrapper is wrapped around
the feature names. 
- In the part where the geodata is queried; it select the observation
with the highest data quality as indicated by a repsonsible at DST 
"""
from dstnx import db, features

DST_USER = "XL2"

GEO = """
SELECT DISTINCT BEF.PERSON_ID,
       BEF.ADRESSE_ID,
       BEF.BOP_VFRA,
       BEF.IE_TYPE,
       BEF.ALDER,
       BEF.KOEN,
       BEF.FAMILIE_ID,
       BEF.MOR_PID,
       BEF.FAR_PID,
       BEF.FAMILIE_TYPE,
       BEF.FM_MARK,
       BEF.HUSTYPE,
       BEFBOP.BOP_VTIL,
       GEOADR.DDKN_M100,
       GEOADR.DDKN_KM1,
       GEOADR.DDKN_KM10,
       GEOADR.ETRS89_EAST,
       GEOADR.ETRS89_NORTH,
       GEOADR.POSTNR,
       GEOADR.REGIONSKODE,
       GEOADR.SOGNEKODE,
       GEOADR.STATUS,
       GEOADR.KOM,
       GEOADR.OPGIKOM,
       UDDF.HFAUDD,
       UDDF.INSTNR,
       IND.PERINDKIALT_13,
       IND.ERHVERVSINDK_13,
       IND.LOENMV_13,
       IND.{KONT},
       IND.OFF_OVERFORSEL_13,
       AKM.{BESKST},
       AKM.{SOCIO},
       RAS.{GRAD}
FROM D222202.BEF{year}12 BEF

inner join D222202.BEFBOP202012 BEFBOP
on BEF.ADRESSE_ID = BEFBOP.ADRESSE_ID
AND BEF.PERSON_ID = BEFBOP.PERSON_ID
AND BEF.BOP_VFRA = BEFBOP.BOP_VFRA

inner join
{DST_USER}.GEOADR GEOADR
    on BEF.ADRESSE_ID = GEOADR.ADRESSE_ID
    and BEF.KOM = GEOADR.KOM

LEFT JOIN
  (SELECT T1.*
   FROM D222202.UDDF202009 T1
   INNER JOIN
     (SELECT PERSON_ID,
             MAX(HF_VTIL) AS MAX_VTIL
      FROM D222202.UDDF202009
      WHERE (HF_VFRA < TO_DATE('{year_plus}-01-01', 'YYYY-MM-DD'))
      GROUP BY PERSON_ID) T2 ON T1.PERSON_ID = T2.PERSON_ID
   AND T1.HF_VTIL = T2.MAX_VTIL) UDDF ON BEF.PERSON_ID = UDDF.PERSON_ID
LEFT JOIN
  (SELECT PERINDKIALT_13, ERHVERVSINDK_13, LOENMV_13, PERSON_ID,
   OFF_OVERFORSEL_13, {KONT}
   FROM D222202.IND{year}) IND ON IND.PERSON_ID = BEF.PERSON_ID
LEFT JOIN
  (SELECT {SOCIO},
          {BESKST},
          PERSON_ID
   FROM D222202.AKM{year}) AKM ON AKM.PERSON_ID = BEF.PERSON_ID
LEFT JOIN
  (SELECT {GRAD},
          PERSON_ID
   FROM D222202.RAS{year}) RAS ON RAS.PERSON_ID = BEF.PERSON_ID
"""


kont_var = features.DSTVariable(
    # From AKM table
    name="KONTANTHJ",
    variable_ranges=[
        features.DSTVariableRange(1985, 1993, "DAGPENGE_KONTANT_13"),
        features.DSTVariableRange(1994, 2020, "KONTANTHJ_13"),
    ],
)

besk_var = features.DSTVariable(
    name="BESKST",
    variable_ranges=[
        features.DSTVariableRange(1985, 1990, "BESKST"),
        features.DSTVariableRange(1991, 2020, "BESKST13"),
    ],
)


grad_var = features.DSTVariable(
    # Only used from 1985-1991 to infer arblos
    # In 1991 we can consider SOCIO_13 from AKM; see below
    name="GRADAT1",
    variable_ranges=[
        features.DSTVariableRange(1980, 2005, "GRADAT1"),
        features.DSTVariableRange(2006, 2007, "DISCO_RAS_KODE"),
        features.DSTVariableRange(2008, 2020, "DISCO_KODE"),
    ],
)

socio_var = features.DSTVariable(
    # From AKM table
    name="SOCIO",
    variable_ranges=[
        features.DSTVariableRange(1985, 1990, "SOCIO_GL"),
        features.DSTVariableRange(1991, 2020, "SOCIO13"),
    ],
)


def get_query(year: int):
    return GEO.format(
        year=year,
        year_plus=year + 1,
        BESKST=besk_var[year],
        SOCIO=socio_var[year],
        GRAD=grad_var[year],
        KONT=kont_var[year],
        DST_USER=DST_USER,
    )


def extract(dst_db: db.DSTDB, year: int):
    """Helper function to rename columns after extracting"""
    return dst_db.extract_data(get_query(year)).rename(
        columns={
            besk_var[year]: besk_var.name,
            socio_var[year]: socio_var.name,
            grad_var[year]: grad_var.name,
            kont_var[year]: kont_var.name,
        }
    )


if __name__ == "__main__":
    print(get_query(1985))
