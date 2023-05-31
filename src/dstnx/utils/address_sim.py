from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import dstnx
from dstnx import data_utils
from dstnx.utils import address_comp

np.random.seed(99)


def generate_locations(size: int, id_start: int = 0):
    X = np.random.rand(size, 2)
    data = {
        "e": X[:, 0],
        "n": X[:, 1],
        "ADRESSE_ID": np.arange(id_start, id_start + size),
    }
    return pd.DataFrame(data)


def generate_address_intervals(
    start, end, num: int = 10, extra_rows: bool = True
) -> np.ndarray:
    """Returns (`num` + 1, 2) array of intervals"""
    weeks = pd.date_range(start, end, freq="w")
    movements = np.sort(np.random.choice(weeks, num, replace=False))
    idc = np.searchsorted(weeks, movements)
    idc_weeks = np.array(list(zip(idc[:-1], idc[1:])))
    intervals = weeks.values[idc_weeks]
    if extra_rows:
        # First row
        rows = [np.array([start, weeks[0].asm8])]  # type: ignore
        if idc[0] != 0:  # Append first time interval if it wasn't chosen by randomness
            rows.append(np.array([weeks[0].asm8, weeks[idc[0]].asm8]))  # type: ignore
        rows.append(intervals)
        # Append last time interval if it wasn't chosen by randomness
        if idc[-1] != len(weeks) - 1:
            rows.append(np.array([weeks[idc[-1]].asm8, weeks[-1].asm8]))  # type: ignore
        rows.append(np.array([weeks[-1].asm8, end]))  # type: ignore
        intervals = np.row_stack(rows)
    # Make move out day one day before
    intervals[:, 1] = intervals[:, 1] - np.timedelta64(1, "D")  # type: ignore
    return intervals


def simulate_data(
    size: int,
    num_intervals: int = 10,
    feature: Optional[str] = None,
    node_id: Optional[str] = None,
    id_start: int = 0,
    large: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = np.datetime64("2020-01-01")
    end = np.datetime64("2021-01-01")
    locations = generate_locations(size, id_start=id_start)
    blocks = []
    lens = []
    ids = []
    if large:
        blocks_to_pick = []
        lens_to_pick = []
        for _ in range(1000):
            intervals = generate_address_intervals(start, end, num_intervals)
            blocks_to_pick.append(intervals)
            lens_to_pick.append(intervals.shape[0])

        idc = np.random.choice(np.arange(1000), size=size)
        for idx in tqdm(range(size)):
            blocks.append(blocks_to_pick[idc[idx]])
            lens.append(lens_to_pick[idc[idx]])
    else:
        for _ in range(size):
            intervals = generate_address_intervals(start, end, num_intervals)
            blocks.append(intervals)
            lens.append(intervals.shape[0])
    data = pd.DataFrame(np.row_stack(blocks), columns=["VFRA", "VTIL"])
    data["ADRESSE_ID"] = np.concatenate(
        [np.repeat(id_start + i, lens[i]) for i in range(size)]
    )
    data = data.merge(locations, on="ADRESSE_ID", how="left")
    if feature:
        data[feature] = np.random.normal(0, 1, size=data.shape[0])
    if node_id:
        ids = np.concatenate([np.repeat(id_start + i, lens[i]) for i in range(size)])
        data[node_id] = ids
    return locations, data.sort_values(by="VFRA").reset_index(drop=True)


def switch_nodes_dict(nodes, mask, node_id: str = "node"):
    num_true = mask.sum()
    idx_pairs = 2 * round(num_true // 2)
    idc = nodes.index[mask.values]
    idc = np.random.choice(idc, idx_pairs, replace=False)
    node_ids = nodes.loc[idc, node_id].values.reshape(-1, 2)
    map1 = dict(zip(node_ids[:, 0], node_ids[:, 1]))
    map2 = dict(zip(node_ids[:, 1], node_ids[:, 0]))
    return map1 | map2  # Cat dicts


def flip_node_addresses(nodes: pd.DataFrame, node_id: str = "node"):
    move_out_dates = nodes.VTIL.unique()
    for move_out_date in tqdm(move_out_dates, f"Flipping nodes for {node_id=}"):
        mask = nodes.VTIL == move_out_date
        node_map = switch_nodes_dict(nodes, mask, node_id)
        mask = nodes.VTIL > move_out_date
        nodes.loc[mask, node_id] = nodes.loc[mask, node_id].replace(node_map)  # type: ignore
    return nodes


# --------------------- Main simulation of data --------------------- #


def main_simulate(
    size: int,
    suffix: str,
    large: bool,
    neighbor_intervals: int = 10,
    node_intervals: int = 3,
):
    print(f"Generating test data of size {size} with suffix {suffix}")
    loc_neighbors, neighbors = simulate_data(
        size, neighbor_intervals, feature="pria", node_id="neighbor", large=large
    )
    loc_nodes, nodes = simulate_data(
        size, node_intervals, node_id="node", id_start=size, large=large
    )
    nodes = flip_node_addresses(nodes, node_id="node")
    neighbors = flip_node_addresses(neighbors, node_id="neighbor")

    print("Saving data ..")
    names = ["loc_neighbors", "neighbors", "loc_nodes", "nodes"]
    dfs = [loc_neighbors, neighbors, loc_nodes, nodes]
    for name, df in zip(names, dfs):
        filename = dstnx.fp.TEST_DATA / f"{name}{suffix}.gzip.parquet"
        df.to_parquet(filename)
        print(f"Saved {name} with dimension {df.shape} to {filename}")


# --------------------- Profiling stuff --------------------- #


def prof_data(suffix: str = "_large"):
    suffix = "_large"
    cols_types = address_comp.astype_dict(
        ["e", "n", "ADRESSE_ID"], ["float64", "float64", "int64"]
    )
    nodes = (
        data_utils.load_test(f"nodes{suffix}")
        .pipe(address_comp.mock_dst_dates)
        .astype(cols_types)
    )
    neighbors = (
        data_utils.load_test(f"neighbors{suffix}")
        .pipe(address_comp.mock_dst_dates)
        .astype(cols_types)
        .pipe(address_comp.cols_to_numpy, ["pria"])
    )
    loc_neighbors = data_utils.load_test(f"loc_neighbors{suffix}").astype(cols_types)
    time_cols = ["VFRA", "VTIL"]
    neighbors = neighbors.pipe(address_comp.date_cols_to_int, time_cols)
    nodes = nodes.pipe(address_comp.date_cols_to_int, time_cols)
    return nodes, neighbors, loc_neighbors
