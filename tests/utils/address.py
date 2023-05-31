import pandas as pd

import dstnx
from dstnx.features import geo
from dstnx.utils import address, test


def test_date_overlap():
    # create a sample DataFrame
    df = pd.DataFrame(
        {
            "BOP_VFRA": pd.date_range(start="2022-01-01", periods=10, freq="D"),
            "BOP_VTIL": pd.date_range(start="2022-01-05", periods=10, freq="D"),
        }
    )

    # specify the datetime interval
    start_date = pd.Timestamp("2022-01-02")
    end_date = pd.Timestamp("2022-01-07")
    print(address.subset_years_mask(start_date, end_date, df))


def time_construct_network():
    time_cols = ["VFRA", "VTIL"]
    features = ["pria"]
    nodes, neighbors, loc_neighbors = test.load_edge_finding_data()
    nh = geo.NeighborNanMeasures(
        loc_neighbors,
        nodes,
        neighbors,
        features=features,
        time_cols=time_cols,
        id_col="neighbor",
        coord_cols=["e", "n"],
    )
    times = []
    for radius in [0.2, 0.4, 0.6, 0.8, 1]:
        ids = nh.measure.kd_tree.query_radius(nh.node_coords, radius)
        counts = geo._count_edges(ids)
        print(f"{radius=}: ", counts.describe())
        edges = nh.compute_network(
            radius=radius,
            convert_to_array=True,
            return_addresses=True,
            func_type="default",
        )
        address.save_edges(edges, name=f"edges_{radius}", fp=dstnx.fp.TEST_DATA)
        averages = address.compute_neighbor_edge_nanaverages(
            edges=edges,
            features=features,
            feature_maps=nh.measure.neighborhood_maps.feature,
        )

    pd.DataFrame(times, columns=["radius", "time"]).to_csv(
        dstnx.fp.TEST_DATA / "times.csv"
    )


def load_edges():
    edges = address.load_edges("edges_0", fp=dstnx.fp.TEST_DATA)
    print(edges)
    return edges


if __name__ == "__main__":
    # test_date_overlap()
    # time_construct_network()
    edges = load_edges()
