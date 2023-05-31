import json
import re
from pathlib import Path
from typing import Any, Optional

import bs4
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

from dstnx import fp, log_utils

LOGGER = log_utils.get_logger(__name__)

FP_TIMES_MAPPINGS = fp.DATA / "times-mappings"
FP_HQ_MAPPINGS = fp.DATA / "hq-mappings"

for fp_map in [FP_HQ_MAPPINGS, FP_TIMES_MAPPINGS]:
    Path.mkdir(fp_map, exist_ok=True)

TABLE_URLS = fp.DATA / "table_urls.json"

URL_DST = "https://www.dst.dk/extranet/forskningvariabellister/Oversigt%20over%20registre.html"
URL_HQ = "https://www.dst.dk/da/TilSalg/Forskningsservice/Dokumentation/hoejkvalitetsvariable"
PREFIX_DST_URL = "http://dst.dk"


def get_link_dict(links: list[bs4.element.Tag], prefix: Optional[str] = None):
    links_dict = dict(
        zip([link.text for link in links], [link["href"] for link in links])
    )
    if prefix:
        links_dict = {k: prefix + url for k, url in links_dict.items()}
    return links_dict


def fetch_links():
    response = requests.get(URL_DST)
    soup = bs(response.text, features="lxml")
    tables = soup.select("table[summary*='Data Set']")
    assert len(tables) == 1
    (table,) = tables
    tbody = table.tbody
    if tbody:
        rows = tbody.find_all("tr")
        tables_link = [row.td.a for row in rows]
        df = pd.DataFrame(
            {
                "table": [link.text for link in tables_link],
                "url": [link["href"] for link in tables_link],
            }
        )
        LOGGER.debug(f"Fetched tables with {df.shape} with overview:\n{df.head()}")
        return df
    else:
        raise ValueError("Table not found!")


def load_json(file: Path):
    with open(file, "r") as f:
        return json.load(f)


def save_json(obj: Any, file: Path, indent: int = 1):
    with open(file, "w") as f:
        json.dump(obj, f, indent=indent)


def get_registers() -> dict[str, str]:
    if TABLE_URLS.exists():
        table_urls = load_json(TABLE_URLS)
    else:
        registers = fetch_links()
        table_urls = registers.set_index("table").url.to_dict()
        save_json(table_urls, TABLE_URLS)
    return table_urls  # type: ignore


def collect_table_info(url: str):
    r = requests.get(url)
    tr_tags = bs(r.text, features="lxml").tbody.find_all("tr")
    th_tags = [tr_tag.find_all("th") for tr_tag in tr_tags]
    th_header = th_tags[0]
    th_rows = th_tags[1:]
    col_names = [th.text.strip() for th in th_header]
    th_data = [
        [th.a["href"] if th.a else th.text.strip() for th in th_row]
        for th_row in th_rows
    ]
    df = pd.DataFrame(th_data, columns=[col.lower() for col in col_names]).rename(
        columns={"times": "times_url", "højkvalitetsdokumentation": "highqual_url"}
    )
    LOGGER.debug(f"Collected table info for {url=} of size {df.shape}")
    return df


def remove_nullbytes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.times_url != "\xa0"]


class BaseVariableMap:
    def __init__(
        self,
        variable: str,
        dst_id: str = "TIMES",
        custom_name: Optional[str] = None,
        verbose: bool = False,
    ):
        self.variable = variable
        self.custom_name = custom_name
        self.verbose = verbose
        match dst_id:
            case "TIMES":
                self.fp = FP_TIMES_MAPPINGS
            case "HQ":
                self.fp = FP_HQ_MAPPINGS

    def _get_rev_mapping(self) -> dict:
        return {v: k for k, v in self.mapping.items()}

    def save_mapping(self):
        filename = self.get_file_name()
        if filename.exists():
            if self.verbose:
                LOGGER.debug("File already exists; skipping save")  # Log
                return
        save_json(self.mapping, filename)
        if self.verbose:
            LOGGER.debug(f"Saved mapping to: {filename}")

    def get_file_name(self) -> Path:
        if self.custom_name:
            name = self.custom_name
        else:
            name = self.variable
        return (self.fp / name).with_suffix(".json")

    def convert_keys_to_int(self):
        self.mapping = {int(k): v for k, v in self.mapping.items()}
        self.rev_mapping = self._get_rev_mapping()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "key": [k for k in self.mapping],
                "value": [val for val in self.mapping.values()],
            }
        )


class DSTVariableMap(BaseVariableMap):
    def __init__(
        self,
        variable: str,
        variable_table: Optional[pd.DataFrame] = None,
        custom_name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            variable=variable, dst_id="TIMES", custom_name=custom_name, verbose=verbose
        )
        self.mapping = self._check_table(variable_table)
        self.rev_mapping = self._get_rev_mapping()
        self.save_mapping()

    def __repr__(self):
        return f"DSTVariableMap(variable='{self.variable}')"

    def _check_table(self, variable_table: Optional[pd.DataFrame]) -> dict[str, str]:
        if isinstance(variable_table, pd.DataFrame):
            mapping = (
                variable_table.astype({"Kode": int}).set_index("Kode").Tekst.to_dict()
            )
        elif (file := self.get_file_name()).exists():
            mapping = load_json(file)
            if self.verbose:
                print(f"Mapping exists; loading mapping from {file}")
        else:
            raise ValueError("No mapping found!")
        return mapping  # type: ignore


def get_html_table(url: str):
    try:
        tables = pd.read_html(url)
    except ValueError as _:
        LOGGER.debug(f"No table found for {url=}!")
        return
    else:
        assert len(tables) == 1
        (table,) = tables
        return table


class DSTTable:
    def __init__(self, table: str, table_info: pd.DataFrame):
        self.table = table
        self.table_info = table_info
        self.variable_times_urls = self.construct_url_map()
        self.variable_maps = dict()

    def construct_url_map(self) -> dict[str, str]:
        return (  # type: ignore
            self.table_info.pipe(remove_nullbytes)
            .set_index("variabel")
            .times_url.to_dict()
        )

    def get_variable_map(
        self, variable: str, custom_name: Optional[str] = None
    ) -> None | DSTVariableMap:
        if variable not in self.variable_times_urls:
            print("Variable not in table")
            return None
        variable_table = self.get_variable_table(variable)
        if isinstance(variable_table, pd.DataFrame):
            dst_variable_map = DSTVariableMap(
                variable=variable,
                variable_table=variable_table,
                custom_name=custom_name,
            )
            self.variable_maps[variable] = dst_variable_map
            return dst_variable_map
        return None

    def get_variable_table(self, variable: str) -> pd.DataFrame | None:
        url = self.variable_times_urls[variable]
        table = get_html_table(url)
        if isinstance(table, pd.DataFrame):
            LOGGER.debug(f"Found table for ({url=})\n:{table.head()}")
        return table


class DSTTables:
    def __init__(self):
        self.table_urls = get_registers()
        self.tables = dict()

    def get_table(self, table: str):
        url = self.lookup_register(table)
        dst_table = DSTTable(table=table, table_info=collect_table_info(url))
        self.tables[table] = dst_table
        return dst_table

    def lookup_register(self, register: str):
        if (register := register.upper()) not in self.table_urls:
            raise ValueError(f"Table not found for {register=}!")
        return self.table_urls[register]


# --------------------- HQ --------------------- #


def get_hq_soup():
    r = requests.get(URL_HQ)
    soup = bs(r.text, features="lxml")
    return soup


def extract_cats(soup):
    return [a.text for a in soup.find_all("a", {"class": "accordion__header"})]


def get_hq_cats():
    soup = get_hq_soup()
    return extract_cats(soup)


def parse_sibling(sibling: str | bs4.element.Tag):
    if not isinstance(sibling, str):
        sibling = sibling.text
    if match := RE_SIBLING.search(sibling):
        return match["key"].strip(), match["value"].strip()
    return None


def parse_siblings(siblings) -> list[tuple[str, str]]:
    map_values = []
    for sibling in siblings:
        parsed_sibling = parse_sibling(sibling)
        if parsed_sibling:
            map_values.append(parsed_sibling)
    return map_values


# Make regex non-greedy; some keys have `:` inside them
RE_SIBLING = re.compile("(?P<key>.*?):(?P<value>.+)")


class HQVariableMap(BaseVariableMap):
    def __init__(self, variable: str, mapping: Optional[dict] = None):
        super().__init__(variable=variable, dst_id="HQ")
        if not mapping:
            if not (file := self.get_file_name()).exists():
                raise ValueError(f"Mapping {variable} does not exist!")
            mapping = load_json(file)
        self.mapping = mapping
        self.rev_mapping = self._get_rev_mapping()

    def __repr__(self):
        return f"HQVariableMap(variable='{self.variable}')"


class DSTHQRegister:
    def __init__(self, register: str, url: str):
        self.register = register
        self.url = url
        self.links: dict[str, str] = dict()
        self.variable_maps = dict()

    def find_variables(self):
        LOGGER.debug(f"Looking for variables in {self.url=}")
        sub_soup = bs(requests.get(self.url).text, features="lxml")
        rows = sub_soup.find("table").find_all("tr")
        variable_links = [row.td.a for row in rows]
        self.links = get_link_dict(variable_links, prefix=PREFIX_DST_URL)
        LOGGER.debug(f"Found {self.links=}")

    def get_variable_maps(self, variables: Optional[list[str]] = None):
        if variables:
            maps_to_collect = variables
        else:
            maps_to_collect = [variable for variable in self.links]
        LOGGER.debug(f"Collecting HQ variables for {maps_to_collect=}")
        for variable in maps_to_collect:
            hq_map = self.get_variable_map(variable)
            if hq_map:
                hq_map.save_mapping()
                self.variable_maps[variable] = hq_map

    def get_variable_map(self, variable: str):
        url = self.links[variable]
        return self._html_table_method(variable, url)
        # return self._html_table_method(variable, url)  # Deprecated

    def _html_table_method(self, variable: str, url: str):
        table = get_html_table(url)
        if isinstance(table, pd.DataFrame):
            mapping = table.set_index("Kode").tekst.to_dict()
            LOGGER.debug(f"Found mapping ({url=}):\n{mapping=}")
            return HQVariableMap(variable=variable, mapping=mapping)

    def _sibling_method(self, variable, url):
        """Note: DST changed their website so this doesn't work atm; might do later"""
        soup = bs(requests.get(url).text, features="lxml")
        siblings = list(soup.find("h3", string="Værdisæt").next_siblings)
        parsed_siblings = parse_siblings(siblings)
        LOGGER.debug(f"Parsed siblings for ({variable=}, {url=}):\n{parsed_siblings}")
        if parsed_siblings:
            return HQVariableMap(variable=variable, mapping=dict(parsed_siblings))
        print(f"Did not find any mapping for {variable}")
        return None


class DSTHQ:
    """
    Class for handling DST HQ
    """

    def __init__(self):
        self.soup = get_hq_soup()
        self.cats = extract_cats(self.soup)
        self.registers: dict[str, dict[str, DSTHQRegister]] = dict()

    def find_registers(self, cat: str):
        if cat not in self.cats:
            raise ValueError("Invalid category!")
        register_urls = self._get_sub_cat_urls(cat)
        self.registers[cat] = {
            k: DSTHQRegister(register=k, url=v) for k, v in register_urls.items()
        }

    def find_register_variables(
        self,
        cat: str,
        register: str,
    ):
        hq_register = self.registers[cat][register]
        hq_register.find_variables()
        return hq_register

    def _get_sub_cat_urls(self, cat: str):
        tag = self.soup.find("a", string=cat, attrs={"class": "accordion__header"})
        sub_cats = tag.parent.find("div", {"class": "accordion__body"})
        sub_cat_urls = sub_cats.find_all("a")
        return get_link_dict(sub_cat_urls, prefix=PREFIX_DST_URL)


def get_mapping(
    variable: str, keys_to_int: bool = True, dst_id: str = "TIMES"
) -> DSTVariableMap | HQVariableMap:
    match dst_id:
        case "TIMES":
            dst_variable_map = DSTVariableMap(variable)
        case "HQ":
            dst_variable_map = HQVariableMap(variable)
        case _:
            raise ValueError("Incorrect DST id!")
    if keys_to_int:
        dst_variable_map.convert_keys_to_int()
    return dst_variable_map


def get_mappings(
    variables: list[str],
    keys_to_lower: bool = False,
    keys_to_int: bool = True,
    dst_id: str = "TIMES",
) -> dict[str, DSTVariableMap | HQVariableMap]:
    mappings = dict()
    for variable in variables:
        dst_variable_map = get_mapping(variable, keys_to_int=keys_to_int, dst_id=dst_id)
        mappings[variable] = dst_variable_map
    if keys_to_lower:
        mappings = {k.lower(): mapping for k, mapping in mappings.items()}
    return mappings
