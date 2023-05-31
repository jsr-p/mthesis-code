from pathlib import Path

PROJ = Path(__file__).parents[2]
DATA = PROJ / "nsdata"
TEST_DATA = PROJ / "nsdata" / "testdata"
DST = PROJ.parent
LOG_DST = DST / "log"
SCRIPTS = PROJ / "scripts"
REG_DATA = DST / "regdata"
REG_PLOTS = DST / "plots"
REG_OUTPUT = DST / "output"
REG_TABLES = DST / "tables"
FIGS = DATA / "figs"
FILTER_MAPS = DATA / "filter-mappings"
LOG_MODELS = LOG_DST / "models"
PL = DATA / "lightning"
PL_MODELS = PL / "models"
PL_LOGS = PL / "logs"

for fp in [
    TEST_DATA,
    LOG_DST,
    REG_DATA,
    REG_PLOTS,
    FIGS,
    REG_TABLES,
    PL,
    PL_MODELS,
    PL_LOGS,
]:
    Path.mkdir(fp, exist_ok=True)
