"""
This script creates the table with addresses
and their geo metadata.
"""
from dstnx import db 

QUERY = """
CREATE TABLE GRUNDSKOLE_KLASSER
as
SELECT G.*, K.KLASSEID
FROM D400600.PSD_GRUNDSKOLE G
INNER JOIN (
  SELECT PERSON_ID, REFERENCETID, MAX(VERSION) AS MAX_VERSION
  FROM D400600.PSD_GRUNDSKOLE
  GROUP BY PERSON_ID, REFERENCETID
) T
ON G.PERSON_ID = T.PERSON_ID
  AND G.REFERENCETID = T.REFERENCETID
  AND G.VERSION = T.MAX_VERSION
inner join (
    SELECT K.*
FROM D400600.PSD_GRUNDSKOLE_KLASSEID K
INNER JOIN (
  SELECT PERSON_ID, REFERENCETID, MAX(VERSION) AS MAX_VERSION
  FROM D400600.PSD_GRUNDSKOLE_KLASSEID
  GROUP BY PERSON_ID, REFERENCETID
) T
ON K.PERSON_ID = T.PERSON_ID
  AND K.REFERENCETID = T.REFERENCETID
  AND K.VERSION = T.MAX_VERSION
    ) K
ON K.PERSON_ID = G.PERSON_ID
AND K.REFERENCETID = G.REFERENCETID
"""



def main():
    dst_db = db.DSTDB(database=None, dsn="DB_PSD.world", proxy=False)
    dst_db.execute(QUERY)
    print("Finished constructing GRUNDSKOLE_KLASSER")

if __name__ == "__main__":
    main()