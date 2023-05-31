from dstnx import dst_tools, log_utils

LOGGER = log_utils.get_logger(__name__)


def collect_kotre():
    table = "KOTRE"
    variables = ["UDEL", "AFG_ART", "TILG_ART", "AUDD", "UDD"]
    # INSTNR suddenly doesn't have names at: 
    # https://www.dst.dk/da/Statistik/dokumentation/Times/moduldata-for-uddannelse-og-kultur/instnr
    dst_tables = dst_tools.DSTTables()
    dst_table = dst_tables.get_table(table)
    for variable in variables:
        dst_variable_map = dst_table.get_variable_map(variable)
        LOGGER.debug(
            f"Mapping for ({table=}, {variable=})\n:{dst_variable_map.mapping=}"
        )

    # Test rev mapping
    dst_variable_map = dst_tools.DSTVariableMap("UDEL")
    if dst_variable_map:
        dst_variable_map.convert_keys_to_int()


def collect_udd():
    # Times
    dst_times = dst_tools.DSTHQ()
    dst_times.find_registers("Uddannelse")
    times_register = dst_times.find_register_variables(
        cat="Uddannelse", register="Elevregister 3"
    )
    times_register.find_variables()
    times_register.get_variable_maps(["INSTNR"])


if __name__ == "__main__":
    collect_udd()
    collect_kotre()
