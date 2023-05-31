import pandas as pd

from dstnx import data_utils
from dstnx.utils import address, address_comp


def load_edge_finding_data(suffix: str = ""):
    time_cols = ["VFRA", "VTIL"]
    feat_cols = ["pria"]

    _date_types = address_comp.astype_dict(time_cols, ["date32[pyarrow]"] * 2)
    neighbors = (
        data_utils.load_test(f"neighbors{suffix}")
        .astype(_date_types)
        .assign(inc=lambda df: df.pria)
    )
    nodes = (
        data_utils.load_test(f"nodes{suffix}")
        .rename(columns={"node": "neighbor"})
        .astype(_date_types)
    )
    loc_neighbors = data_utils.load_test(f"loc_neighbors{suffix}")

    addr_map = address.construct_address_map(
        pd.concat((nodes.ADRESSE_ID, loc_neighbors.ADRESSE_ID)).reset_index()
    )
    nodes, neighbors, loc_neighbors = address.transform_data_neigh(
        nodes,
        neighbors,
        loc_neighbors,
        feat_cols=feat_cols,
        addr_map=addr_map,
        time_cols=time_cols,
        coord_cols=["e", "n"],
        id_col="neighbor",
    )
    return nodes, neighbors, loc_neighbors
