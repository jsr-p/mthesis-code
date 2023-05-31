from dstnx import dst_tools


def test_load():
    variables = ["AFG_ART", "INSTNR", "TILG_ART", "UDEL"]
    mappings = dst_tools.get_mappings(variables, dst_id="HQ", keys_to_int=False)
    for _map in mappings.values():
        print(_map.mapping)


if __name__ == "__main__":
    test_load()
