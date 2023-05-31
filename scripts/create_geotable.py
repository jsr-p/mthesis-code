"""
This script creates the table with addresses
and their geo metadata.
"""
from dstnx import db 

QUERY = """
CREATE TABLE GEOADR
AS
SELECT DISTINCT BEFADR.ADRESSE_ID,
                BEFADR.BOPIKOM,
                DAR.OPGIKOM,
                DAR.KOM,
                DAR.REGIONSKODE,
                DAR.POSTNR,
                DAR.SOGNEKODE,
                DAR.DDKN_M100,
                DAR.DDKN_KM1,
                DAR.DDKN_KM10,
                DAR.ETRS89_EAST,
                DAR.ETRS89_NORTH,
                DAR.STATUS,
                BEFADR.ADR_RFRA,
                BEFADR.ADR_VFRA,
                BEFADR.ADR_VTIL,
                SUBSTR(BEFADR.BOPIKOM, 1, 4) AS VEJNR,
                SUBSTR(BEFADR.BOPIKOM, 5, 3) AS HUSNR,
                SUBSTR(BEFADR.BOPIKOM, 8, 1) AS HUSBOGSTAV,
                SUBSTR(BEFADR.BOPIKOM, 9, 2) AS ETAGE,
                SUBSTR(BEFADR.BOPIKOM, 11, 4) AS SIDE_DOOR
FROM   d222202.BEFADR202109 BEFADR
       INNER JOIN (SELECT TEMP_DAR.*
                   FROM   (SELECT DDKN_M100,
                                  DDKN_KM1,
                                  DDKN_KM10,
                                  ETRS89_EAST,
                                  ETRS89_NORTH,
                                  POSTNR,
                                  REGIONSKODE,
                                  SOGNEKODE,
                                  STATUS,
                                  CONCAT(VEJKODE, HUSNR) AS OPGIKOM,
                                  KOM
                           FROM   d221916.DAR_ADGADR16012023) TEMP_DAR
                          INNER JOIN (SELECT Min(STATUS) AS MIN_STATUS,
                                             KOM,
                                             OPGIKOM
                                      FROM   (SELECT STATUS,
                                                     KOM,
                                                     POSTNR,
                                                     CONCAT(VEJKODE, HUSNR) AS
                                                     OPGIKOM
                                              FROM   d221916.DAR_ADGADR16012023)
                                      GROUP  BY KOM,
                                                OPGIKOM) TEMP_DAR_GP
                                  ON TEMP_DAR.KOM = TEMP_DAR_GP.KOM
                                     AND TEMP_DAR.OPGIKOM = TEMP_DAR_GP.OPGIKOM
                                     AND TEMP_DAR.STATUS =
                                         TEMP_DAR_GP.MIN_STATUS) DAR
               ON BEFADR.OPGIKOM = DAR.OPGIKOM
                  AND BEFADR.KOM = DAR.KOM
"""


def main():
    dst_db = db.DSTDB(proxy=False)
    dst_db.execute(QUERY)
    print("Finished constructing geotable")

if __name__ == "__main__":
    main()