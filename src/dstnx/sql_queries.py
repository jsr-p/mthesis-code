KOTRE_DB = """
select *
from KOTRE2020
left join (select ALDER, PERSON_ID, FAMILIE_ID, MOR_PID, FAR_PID from BEF{year_start}12) B using (PERSON_ID)
where (
    KOTRE2020.ELEV3_VTIL >= TO_DATE('{year_start}-01-01', 'YYYY-MM-DD')
    and KOTRE2020.ELEV3_VTIL < TO_DATE('{year_end}-01-01', 'YYYY-MM-DD')
)
"""

KOTRE_QUERY = """
select *
from KOTRE2020
left join (select ALDER, FOED_DAG, PERSON_ID, FAMILIE_ID, MOR_PID, FAR_PID from BEF{year_start}12) B using (PERSON_ID)
where (
    KOTRE2020.ELEV3_VTIL >= TO_DATE('{year_start}-01-01', 'YYYY-MM-DD')
    and KOTRE2020.ELEV3_VTIL < TO_DATE('{year_end}-01-01', 'YYYY-MM-DD')
    and {edu_col} in (select * from table(:uddtmp))
)
"""
SIBLING_QUERY1 = """
select FAMILIE_ID, PERSON_ID, ALDER
from BEF{year_start}12
inner join (select * from TMPFAMID) B using (FAMILIE_ID)
where (
    BEF{year_start}12.PERSON_ID not in (select * from table(:pidkotre))
)
"""
SIBLING_QUERY2 = """
select B.PERSON_ID, B.{PARENT}_PID, B.ALDER, B.FOED_DAG,
T.PERSON_ID, T.{PARENT}_PID
from BEF{year_start}12 B
inner join TMPIDS T
on B.{PARENT}_PID = T.{PARENT}_PID
where (
    B.PERSON_ID not in (select * from table(:pidkotre))
)
"""

SIBLING_QUERY_BOTH = """
select B.PERSON_ID, B.MOR_PID, B.FAR_PID, B.ALDER, B.FOED_DAG,
T.PERSON_ID, T.MOR_PID, T.FAR_PID
from BEF{year_start}12 B
inner join TMPIDS T
on B.FAR_PID = T.FAR_PID
AND B.MOR_PID = T.MOR_PID
where (
    B.PERSON_ID not in (select * from table(:pidkotre))
)
"""

OUTCOME_QUERY = """
select *
from KOTRE2020
where (
    PERSON_ID in (select * from table(:id))
    and KOTRE2020.ELEV3_VFRA < TO_DATE('{end_year}-01-01', 'YYYY-MM-DD')
    and AUDD not in (select * from table(:audd))
)
"""
