from dataclasses import dataclass
from typing import Optional, overload

import numpy as np
import pandas as pd
from numba.typed import List as nb_List
from sklearn.neighbors import KDTree

from dstnx import log_utils
from dstnx.utils import address, address_comp

LOGGER = log_utils.get_logger(name=__name__)

# --------------------- cols --------------------- #

cols = ["ADRESSE_ID", "KOM", "BOP_VFRA", "BOP_VTIL"]

addr_col = ["ADRESSE_ID"]
time_cols = ["BOP_VFRA", "BOP_VTIL"]
coord_cols = ["ETRS89_EAST", "ETRS89_NORTH"]
id_col = ["PERSON_ID"]
famid_col = ["FAMILIE_ID"]

id_coord_cols = ["ADRESSE_ID", "ETRS89_EAST", "ETRS89_NORTH"]
addr_time_cols = ["ADRESSE_ID", "BOP_VFRA", "BOP_VTIL"]

# --------------------- funcs --------------------- #


class GeoKDTree:
    """Class to construct (geographical) KDTree.

    It builds the features from the neighbors using the mean of the neighbors.
    Furthermore, it iteratively removes nan values from the columns of interest
    to avoid dropping non-nan values in the other columns of interest
    that could be used to construct the neighborhood measure of that
    given column.

    Attributes:
        neighbors: data of the neighbors
        coord_cols: coordinate columns
        X: neighbor data matrix
        kd_tree: estimated KD tree
    """

    def __init__(self, neighbors: pd.DataFrame, coord_cols: list[str]):
        self.neighbors = neighbors
        self.coord_cols = coord_cols

        self.X = self.neighbors[self.coord_cols].values
        self.kd_tree: EstimatedGeoKDTree

    @overload
    def find_all(
        self,
        df: pd.DataFrame,
        cols: list[str],
        k: int = 7,
        as_df: bool = True,
    ) -> pd.DataFrame:
        ...

    @overload
    def find_all(
        self,
        df: pd.DataFrame,
        cols: list[str],
        k: int = 7,
        as_df: bool = False,
    ) -> np.ndarray:
        ...

    def find_all(
        self, df: pd.DataFrame, cols: list[str], k: int = 7, as_df: bool = False
    ) -> pd.DataFrame | np.ndarray:
        neighbors_avg = np.zeros(shape=(df.shape[0], len(cols)))
        for i, col in enumerate(cols):
            self.build(col)
            neighbors_avg[:, i] = self.query(df[self.coord_cols].values, k=k)
        if as_df:
            return pd.DataFrame(
                neighbors_avg, columns=[f"{col}_neighbor" for col in cols]
            )
        return neighbors_avg

    def build(self, col: Optional[str] = None):
        if col:
            mask = self.neighbors[self.coord_cols + [col]].isna().any(axis=1)
            neighbors = self.neighbors.loc[~mask, :]
            X = self.X[~mask, :]
            print(f"# of NaNs in {col}: {mask.sum()}")
        else:
            X = self.X
            neighbors = self.neighbors
        self.kd_tree = EstimatedGeoKDTree(KDTree(X), col, neighbors)

    def query(self, arr: np.ndarray, k: int, compute_avg: bool = True):
        return self.kd_tree.query(arr, k, compute_avg=compute_avg)

    def query_radius(self, arr: np.ndarray, radius: float) -> np.ndarray:
        return self.kd_tree.kd_tree.query_radius(arr, radius)


class EstimatedGeoKDTree:
    def __init__(self, kd_tree: KDTree, col: str, neighbors: pd.DataFrame):
        self.kd_tree = kd_tree
        self.col = col
        self.neighbors = neighbors

    def query(
        self, arr: np.ndarray, k: int = 7, compute_avg: bool = True
    ) -> np.ndarray:
        """Returns the mean of the neighbors `col` value.

        Args:
            arr: coordinates to find neighbors of.

        Returns:
            (np.ndarray): (N, ) mean of neighbors `col` value

        """
        self.dist, self.ind = self.kd_tree.query(arr, k=k)
        if not compute_avg:
            return self.ind
        return (
            np.take(self.neighbors[[self.col]].values, self.ind, axis=0)
            .mean(axis=1)
            .ravel()
        )


def compute_geo_measures(neighbors: pd.DataFrame):
    geo_feats = dict()
    geo_cols = ["DDKN_M100", "DDKN_KM1"]
    for col in geo_cols:
        geo_feats[col] = (
            neighbors.groupby(col)
            .agg(
                {
                    "arblos": "mean",
                    "PERINDKIALT_13": lambda df: df.div(10_000).mean(),
                    "highest_edu_pria": "mean",
                    "imm": "mean",
                }
            )
            .rename(mapper=lambda val: f"{val}_{col}", axis=1)
        )
        LOGGER.info(f"Constructed geofeatures for {col}")
    return geo_feats


# --------------------- Neighbor specific --------------------- #


def subset_valid_addr(neighbors: pd.DataFrame, loc_neighbors: pd.DataFrame):
    # TODO: log those that are being dropped
    valid_addresses = neighbors.loc[:, "ADRESSE_ID"].unique()
    loc_neighbors_valid = loc_neighbors.loc[
        loc_neighbors["ADRESSE_ID"].isin(valid_addresses), :
    ]
    return loc_neighbors_valid


def subset_loc(neighbors, loc_neighbors: pd.DataFrame, feat: str):
    LOGGER.info(f"#Obs before subsetting {(neighbors.shape, loc_neighbors.shape)}")
    mask = ~neighbors[feat].isna()
    neighbors_valid = neighbors.loc[mask, :]
    loc_neighbors_valid = subset_valid_addr(neighbors_valid, loc_neighbors)
    LOGGER.info(
        f"#Obs after subsetting {(neighbors_valid.shape, loc_neighbors_valid.shape)}"
    )
    return loc_neighbors_valid, neighbors_valid


# --------------------- Class --------------------- #


def _count_edges(ids: np.ndarray):
    """Count the number of edges for each node."""
    return pd.Series([ids[i].shape[0] for i in range(ids.shape[0])])


@dataclass
class NeighborhoodMaps:
    feature: dict
    addr_to_neigh: dict
    neigh_time: dict


# --------------------- Network with nan features --------------------- #


@dataclass
class NeighborhoodNanMaps:
    feature: list[dict]
    addr_to_neigh: dict
    neigh_time: dict


class NeighborhoodNanMeasure:
    def __init__(
        self,
        loc_neighbors: pd.DataFrame,
        neighbors: pd.DataFrame,
        features: list[str],
        time_cols: list[str],
        coord_cols: list[str] = ["ETRS89_EAST", "ETRS89_NORTH"],
        id_col: str = "PERSON_ID",
        adr_col: str = "ADRESSE_ID",
    ):
        self.loc_neighbors_valid = loc_neighbors
        self.neighbors_valid = neighbors
        self.features = features
        self.time_cols = time_cols
        self.coord_cols = coord_cols
        self.id_col = id_col
        self.adr_col = adr_col

        self.neighborhood_maps = self.construct_maps()
        self.idx_to_adr_id: np.ndarray = self.loc_neighbors_valid["ADRESSE_ID"].values
        self.build_tree()

    def compute_network(
        self,
        node_ids: np.ndarray,
        node_dates: np.ndarray,
        node_coords: np.ndarray,
        radius: float,
        return_addresses: bool = True,
        convert_to_array: bool = True,
        **kwargs,
    ) -> dict:
        """Computes the neighborhood measure by querying the nearest neighbors
        and computing the average of those nodes.

        `k` and `radius` are by default equal to None.
        """
        nearest_addr_ids = self.query_tree(node_coords, radius, return_addresses)
        return address.compute_neighbor_nan_network(
            node_dates,
            node_ids,
            self.neighborhood_maps.addr_to_neigh,
            self.neighborhood_maps.neigh_time,
            self.neighborhood_maps.feature,
            nearest_addr_ids,
            convert_to_array=convert_to_array,
            **kwargs,
        )

    def build_tree(self):
        """We assume the loc_values are unique.

        We have to make a mapping from idx -> adresse_id
        """
        self.kd_tree = GeoKDTree(self.loc_neighbors_valid, coord_cols=self.coord_cols)
        self.kd_tree.build()

    def query_tree(
        self,
        node_coords: np.ndarray,
        radius: float,
        return_addresses: bool = True,
    ):
        """Returns the location ids that are closest to the input coords."""
        return self._query_radius(
            node_coords, radius=radius, return_addresses=return_addresses
        )

    def _query_radius(
        self,
        node_coords: np.ndarray,
        radius: float,
        return_addresses: bool = True,
    ):
        """
        Query radius returns an ndarray of objects; arrays with nonequal dimensions
        """
        ids = self.kd_tree.query_radius(node_coords, radius=radius)
        LOGGER.debug(
            f"Distribution of #neighbors for each node:\n"
            # When querying kdtree with a radius, the returned array is an array of
            # arrays; thus they don't have equal lengths
            f"{_count_edges(ids).describe()}"
        )
        if return_addresses:
            return {
                # Handle the case where given node does not have any neighbors
                # in the specified radius
                idx: self.idx_to_adr_id[ids[idx]]
                if ids[idx].size > 0
                else np.array([], dtype=np.int64)
                for idx in range(ids.shape[0])
            }
        return {idx: ids[idx] for idx in range(ids.shape[0])}

    def construct_maps(self):
        LOGGER.info("Constructing maps with possible nans")
        feature_maps = nb_List()
        LOGGER.info("Constructing feature maps ...")
        for feature in self.features:
            _map = address_comp.feature_map(
                self.neighbors_valid, feature, id_col=self.id_col, as_nb_dict=True
            )
            feature_maps.append(_map)

        LOGGER.info("Constructing address to neighbors map ...")
        addr_to_neigh = address_comp.addr_to_neigh_map(
            self.neighbors_valid,
            id_col=self.id_col,
            adr_col=self.adr_col,
            as_nb_dict=True,
        )
        LOGGER.info("Constructing neighbor to address time map ...")
        neigh_time_map = address_comp.neigh_addr_time_map(
            self.neighbors_valid,
            id_col=self.id_col,
            adr_col=self.adr_col,
            as_nb_dict=True,
            time_cols=self.time_cols,
        )
        return NeighborhoodNanMaps(feature_maps, addr_to_neigh, neigh_time_map)


class NeighborNanMeasures:
    def __init__(
        self,
        loc_neighbors: pd.DataFrame,
        nodes: pd.DataFrame,
        neighbors: pd.DataFrame,
        features: list[str],
        time_cols: list[str],
        radius: int = 800,
        coord_cols: list[str] = ["ETRS89_EAST", "ETRS89_NORTH"],
        id_col: str = "PERSON_ID",
        loc_id_col: str = "ADRESSE_ID",
        batch_size: Optional[int] = None,
    ):
        self.loc_neighbors = loc_neighbors
        self.nodes = nodes
        self.neighbors = neighbors.reset_index(drop=True)
        self.features = features
        self.time_cols = time_cols
        self.radius = radius
        self.coord_cols = coord_cols
        self.id_col = id_col
        self.loc_id_col = loc_id_col
        self.batch_size = batch_size

        self._pre_proc()
        self._filter_by_tree()
        self._transform_data()

        self._init_arrays()
        self._init_meas()
        self.N = nodes.shape[0]

        if self.batch_size:
            self._prepare_batcher()

    def _pre_proc(self):
        """Only keep those neighbors with addresses in address data"""
        neighbors_valid = self.neighbors
        self.loc_neighbors = subset_valid_addr(neighbors_valid, self.loc_neighbors)

    def _filter_by_tree(self):
        """Filter observations by the coordinates being queried"""
        LOGGER.debug("Filtering observations by radius..")
        kd_tree = GeoKDTree(self.neighbors[self.coord_cols], coord_cols=self.coord_cols)
        kd_tree.build()
        nested_addr_obj = kd_tree.query_radius(
            self.nodes[self.coord_cols], radius=self.radius
        )
        adr = self._unique_addresses(nested_addr_obj)

        def _filter_adr(df):
            return df.loc[df[self.loc_id_col].isin(adr)].reset_index(drop=True)

        LOGGER.debug(f"#Neighbors before filter: {self.neighbors.shape[0]}")
        self.loc_neighbors = _filter_adr(self.loc_neighbors)
        self.neighbors = _filter_adr(self.neighbors)
        LOGGER.debug(f"#Neighbors after filter: {self.neighbors.shape[0]}")

    def _unique_addresses(self, nested_addr_obj: np.ndarray) -> pd.Series:
        adr = set()
        for arr in nested_addr_obj:
            adr.update(set(arr))
        return self.neighbors["ADRESSE_ID"].iloc[np.array(list(adr))]

    def _batch_by_ids(self):
        """Batch by node id; avoids memory error"""
        ...

    def _transform_data(self):
        LOGGER.debug("Transforming filtered data...")
        self.addr_map = address.construct_address_map(
            pd.concat(
                (self.loc_neighbors[[self.loc_id_col]], self.nodes[[self.loc_id_col]])
            ),
            adr_col=self.loc_id_col,
        )
        self.nodes, self.neighbors, self.loc_neighbors = address.transform_data_neigh(
            self.nodes,
            self.neighbors,
            self.loc_neighbors,
            feat_cols=self.features,
            addr_map=self.addr_map,
            coord_cols=self.coord_cols,
            adr_col=self.loc_id_col,
            id_col=self.id_col,
            time_cols=self.time_cols,
        )
        # TODO: Do we query ourselves when querying youth?
        assert (self.neighbors.ADRESSE_ID.isin(self.loc_neighbors.ADRESSE_ID)).all()

    def _init_arrays(self):
        self.node_dates = self.nodes[self.time_cols].values
        self.node_coords = self.nodes[self.coord_cols].values
        self.node_ids = self.nodes[[self.id_col] + [self.loc_id_col]].values
        self.sorted_unq_node_ids = np.unique(self.node_ids[:, 0])

    def _init_meas(self):
        self.measure = NeighborhoodNanMeasure(
            self.loc_neighbors,
            self.neighbors,
            self.features,
            self.time_cols,
            coord_cols=self.coord_cols,
            id_col=self.id_col,
        )

    def _prepare_batcher(self):
        self.batcher = Batcher(self.nodes, batch_size=self.batch_size)
        self.batches = self.batcher.yield_batches()

    def batch_compute_network(self, batch_idc, **kwargs):
        return self.measure.compute_network(
            self.node_ids[batch_idc, :],
            self.node_dates[batch_idc, :],
            self.node_coords[batch_idc, :],
            self.radius,
            **kwargs,
        )

    def compute_network(
        self,
        radius: float,
        **kwargs,
    ):
        return self.measure.compute_network(
            self.node_ids, self.node_dates, self.node_coords, radius, **kwargs
        )


# --------------------- Other utils --------------------- #


class Batcher:
    def __init__(self, nodes: pd.DataFrame, batch_size):
        self.nodes = nodes.sort_values(by="PERSON_ID").reset_index(drop=True)
        self.batch_size = batch_size
        self._init()
        self._batch()

    def _init(self):
        self.gp = self.nodes.groupby("PERSON_ID").BOP_VFRA.count()
        self.bins = int(np.ceil(self.gp.shape[0] / self.batch_size))
        self.groups = pd.cut(self.gp.cumsum(), bins=self.bins)

    def _batch(self):
        sums = 0
        self.batch_pnrs = dict()
        batch_overview = []
        for batch_num, cat in enumerate(self.groups.cat.categories):
            mask = self.groups == cat
            pnrs = mask.index[mask.values]
            _sum = self.gp[pnrs].sum()
            sums += _sum
            self.batch_pnrs[batch_num] = pnrs
            batch_overview.append((batch_num, _sum))
        assert sums == self.gp.sum()
        self.num_batches = len(self.batch_pnrs)
        LOGGER.debug(
            f"Distribution of batches (batch_num, counts): {dict(batch_overview)}"
        )

    def yield_batches(self):
        for batch_num in self.batch_pnrs:
            pnrs = self.batch_pnrs[batch_num]
            idc = self.nodes.PERSON_ID.isin(pnrs)
            yield self.nodes.loc[idc], batch_num
