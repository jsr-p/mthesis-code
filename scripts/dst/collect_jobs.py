"""
Gather information on table to pick variables to choose.
"""
from dstnx import dst_tools, log_utils

LOGGER = log_utils.get_logger(__name__)


def collect_akm():
    table = "AKM"
    variables = ["BESKST13", "SOCIO13"]
    dst_tables = dst_tools.DSTTables()
    dst_table = dst_tables.get_table(table)
    for variable in variables:
        dst_variable_map = dst_table.get_variable_map(variable)
        LOGGER.debug(f"{table=}, {dst_variable_map.mapping=}")

    # Test rev mapping
    for var in variables:
        dst_variable_map = dst_tools.DSTVariableMap(var)
        if dst_variable_map:
            dst_variable_map.convert_keys_to_int()
            LOGGER.debug(f"{table=}, {dst_variable_map.mapping=}")


if __name__ == "__main__":
    collect_akm()
