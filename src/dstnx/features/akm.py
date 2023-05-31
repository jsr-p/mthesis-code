from dstnx import log_utils

LOGGER = log_utils.get_logger(name=__name__)


ID_COL = "PERSON_ID"
QUERY_AKM = """
select * from TMPFEATIDS
left join (select {cols_str} from AKM{year}) A using (PERSON_ID)
"""
