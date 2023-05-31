import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import oracledb
import pandas as pd

import dstnx
from dstnx import log_utils

LOGGER = log_utils.get_logger(name=__name__)


@dataclass
class DBConfig:
    db1: str
    dsn1: str


def load_config(file: Path = dstnx.fp.PROJ.parent / "config.json") -> DBConfig:
    with open(file, "r") as f:
        config: dict[str, str] = json.load(f)
        return DBConfig(**config)


try:
    CONFIG = load_config()
except FileNotFoundError:
    file = dstnx.fp.DATA / "config_placeholder.json"
    CONFIG = load_config(file=file)
    LOGGER.debug("Loaded file from placeholder config")


def extract_col_names(cur: oracledb.Cursor) -> list[str]:
    return [row[0] for row in cur.description]


TMP_EXC_QUERY = """
CREATE GLOBAL TEMPORARY TABLE {name}(
        {dtypes}
) ON COMMIT PRESERVE ROWS
"""
TMP_EXISTS = "ORA-00955: name is already used by an existing object"
TMP_NEXISTS = "ORA-00942: table or view does not exist"


class DSTDB:
    def __init__(
        self,
        database: Optional[str] = CONFIG.db1,
        dsn: str = CONFIG.dsn1,
        proxy: bool = True,
    ) -> None:
        self.con = self.con_db(database=database, dsn=dsn, proxy=proxy)
        self.collections: dict[str, oracledb.DbObject] = dict()

    def con_db(
        self,
        database: Optional[str] = CONFIG.db1,
        dsn: str = CONFIG.dsn1,
        proxy: bool = True,
    ) -> oracledb.Connection:
        oracledb.init_oracle_client()
        if proxy:
            con = oracledb.connect(user=f"[{database}]", dsn=dsn)
        else:
            con = oracledb.connect(dsn=dsn)
        return con

    def execute(self, query: str, parameters: Optional[dict] = None) -> pd.DataFrame:
        with self.con.cursor() as cur:
            cur.execute(query, parameters=parameters)

    def extract_data(
        self, query: str, parameters: Optional[dict] = None
    ) -> pd.DataFrame:
        with self.con.cursor() as cur:
            cur.execute(query, parameters=parameters)
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=extract_col_names(cur))

    def create_collection_table(
        self, name: str, dtype: str = "integer", values: Optional[list] = None
    ):
        with self.con.cursor() as cur:
            statement = f"CREATE OR REPLACE TYPE {name} AS TABLE OF {dtype}"
            cur.execute(statement)
        self.collections[name.lower()] = self.con.gettype(name.upper()).newobject()
        if values:
            self.extend_table_values(name=name, values=values)

    def get_collection_table(self, name: str):
        return self.collections[name]

    def extend_table_values(self, name: str, values: list):
        self.collections[name.lower()].extend(values)

    def create_global_tmp_table(self, name: str, dtypes: list[str]):
        dtypes = "\n\t,".join(dtypes)
        exc_query = TMP_EXC_QUERY.format(name=name, dtypes=dtypes)
        with self.con.cursor() as cur:
            try:
                cur.execute(exc_query)
            except oracledb.DatabaseError as exc:
                if TMP_EXISTS in exc.args[0].message:
                    LOGGER.debug(f"Temporary table `{name}` already in use!")
                else:
                    LOGGER.debug(f"Error in query:\n{exc_query}")
                    raise exc

    def insert_global_tmp_table(self, statement: str, rows: list):
        with self.con.cursor() as cur:
            cur.executemany(statement, rows)

    def reset_global_tmp_table(self, name: str):
        try:
            self.execute(f"truncate table {name}")
        except oracledb.DatabaseError as exc:
            if TMP_NEXISTS in exc.args[0].message:
                print(f"Temporary table `{name}` does not exist!")
            else:
                raise exc

    def drop_global_tmp_table(self, name: str):
        try:
            self.execute(f"drop table {name}")
        except oracledb.DatabaseError as exc:
            if TMP_NEXISTS in exc.args[0].message:
                LOGGER.debug(f"Temporary table `{name}` does not exist!")
            else:
                raise exc

    def check_cols_exist(
        self, year_start: int, year_end: int, cols: list[str], query: str
    ):
        len_cols = len(cols)
        set_cols = set(cols)
        for year in range(year_start, year_end):
            table_cols = self.extract_data(query.format(year=year)).columns
            if not len(set(table_cols) & set_cols) == len_cols:
                LOGGER.debug(f"Error in year {year}:")
                LOGGER.debug(
                    f"Diff1 (exist in specified cols but not in the table):\n{set_cols - set(table_cols)}"
                )
                LOGGER.debug(
                    f"Diff2 (exist in table but not in the specified cols):\n{set(table_cols) - set_cols}"
                )
                return False
        return True

    def inspect_cols(self, table: str):
        return self.extract_data(f"select * from {table} where 1=2").columns


def prefix_cols(cols: list[str], prefix: str):
    return [f"{prefix}_{col}" for col in cols]


def prefix_right(df: pd.DataFrame, prefix: str, prefix_start: int):
    """Helper function to prefix column names."""
    cols = df.columns.tolist()
    right_cols = prefix_cols(cols=cols[prefix_start:], prefix=prefix)
    new_cols = cols[:prefix_start] + right_cols
    df.columns = new_cols
    return df


def prefix_left(df: pd.DataFrame, prefix: str, prefix_end: int):
    """Helper function to prefix column names."""
    cols = df.columns.tolist()
    left_cols = prefix_cols(cols=cols[:prefix_end], prefix=prefix)
    new_cols = left_cols + cols[prefix_end:]
    df.columns = new_cols
    return df


def join_cols_sql(cols: list[str]):
    return ", ".join(cols)
