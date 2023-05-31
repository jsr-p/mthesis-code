"""
This module contains functions for computing nearest neighbors.
The main function is `main_multiple_periods`, which computes
the nearest neighbors for a given cohort.
See `README.md` at the top of the project directory.
"""

import datetime
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import click
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import dstnx
from dstnx import data_utils, log_utils
from dstnx.data import neighbors as neighbors_mod
from dstnx.features import geo, parents
from dstnx.utils import address, address_comp

LOGGER = log_utils.get_logger(name=__name__)


# --------------------- Utils --------------------- #


FEAT_COLS = [
    "highest_edu_pria",
    "arblos",
    "imm",
    "ses",
    "inc",
    "inc_kont",
    "inc_off",
    "highest_gs",
    "highest_eu",
    "crimes",
]

YOUTH_FEAT_COLS = ["ses", "mor_ses", "far_ses", *neighbors_mod.FM_MARK_COLS_1, "crimes"]


def get_feat_cols(neighbor_type: str = "neighbors"):
    if neighbor_type == "neighbors":  # Use the same cols for parents
        return FEAT_COLS
    elif neighbor_type == "youth":
        return YOUTH_FEAT_COLS
    else:
        raise ValueError


def amount_overlap_single(
    start: np.ndarray,
    end: np.ndarray,
    neigh_start: np.ndarray,
    neigh_end: np.ndarray,
) -> np.ndarray:
    """
    See:
        http://baodad.blogspot.com/2014/06/date-range-overlap.html
    """
    return np.min(
        np.column_stack(
            (neigh_end - neigh_start, neigh_end - start, end - start, end - neigh_start)
        ),
        axis=1,
    )


@dataclass
class Config:
    start: int
    end: int
    neighbor_type: str
    file_suffix: str
    batch_size: int
    youth_age: str = ""
    max_radius: int = 1000
    k_nearest: bool = False
    save_edges: bool = False


@dataclass
class NeighborIds:
    pid: np.ndarray
    famid: np.ndarray


def _construct_map_indices(
    nodes: pd.DataFrame,
) -> tuple[np.ndarray, Callable[[np.ndarray], list[int]]]:
    """Constructs a mapping from index back to PID.

    Map PID -> {0, 1, ..., N-1}; we use this for
    the numpy ufuncs.
    """
    _id_col = "PERSON_ID"
    node_ids = np.arange(nodes.shape[0])
    idx_to_id = {k: v for k, v in zip(node_ids, nodes[_id_col].values)}

    def _map_indices(idcs: np.ndarray) -> list[int]:
        return [idx_to_id[val] for val in idcs]

    return node_ids, _map_indices


def radii_interval(conf: Config):
    if conf.neighbor_type == "youth":
        return [100, 200, 400, 600, 800, 1000, 1500]
    return [100, 200, 400, 600, 800]


def k_values(conf: Config):
    if conf.neighbor_type == "youth":
        return [15, 30, 50]
    return [15, 30, 50, 80]


def select_interval(conf: Config):
    if conf.k_nearest:
        return k_values(conf)
    return radii_interval(conf)


def time_overlap_masks(
    nodes: pd.DataFrame,
    node_idcs: np.ndarray,
    neighbors_times: np.ndarray,
    cat_ind: np.ndarray,
):
    """Computes the time overlaps between neighbors and self.

    Args:
        nodes: node data with time columns
        node_idcs: indices for the nodes corresponding to each obs
        neighbors_times: neighbor time columns
        cat_ind: concatenated neighbor indices
    """
    node_times = nodes[geo.time_cols].values  # Node times
    times = node_times[node_idcs]
    times_n = neighbors_times[cat_ind]

    mask_overlap = address.years_overlap(
        times[:, 0], times[:, 1], times_n[:, 0], times_n[:, 1]
    )
    # Compute overlaps also for non-valid; we subset by the mask corresponding to full araray
    overlaps = amount_overlap_single(
        times[:, 0], times[:, 1], times_n[:, 0], times_n[:, 1]
    )
    return mask_overlap, overlaps


def parents_mask(
    nodes: pd.DataFrame,
    neighbor_ids: NeighborIds,
    node_idcs: np.ndarray,
    cat_ind: np.ndarray,
    conf: Config,
    mp_logger,
):
    neighbor_idcs = neighbor_ids.pid[cat_ind]
    if conf.neighbor_type == "neighbors":
        # Exclude parents from neighborhood adults
        mask_exclude = (nodes["MOR_PID"].values[node_idcs] != neighbor_idcs) & (
            nodes["FAR_PID"].values[node_idcs] != neighbor_idcs
        )
    elif conf.neighbor_type == "youth":
        # Exclude siblings from youth  (assuming siblings == same famid)
        fam_idcs = neighbor_ids.famid[cat_ind]
        mask_exclude = (nodes["PERSON_ID"].values[node_idcs] != neighbor_idcs) & (
            nodes["FAMILIE_ID"].values[node_idcs] != fam_idcs
        )
    else:
        raise ValueError("Need either youth or neighbors")

    mp_logger.debug(
        f"Exclude {np.sum(~mask_exclude)}/{neighbor_idcs.shape[0]} invalid dyads based on ids"
    )
    return mask_exclude


# --------------------- New Abstraction with radius & k --------------------- #


def query_tree(nodes: pd.DataFrame, kd_tree: KDTree, max_val: int, conf: Config):
    if conf.k_nearest:
        # Note: k-nearest returns (dist, ind) in this order :)
        dist, ind = kd_tree.query(
            X=nodes[geo.coord_cols].values, k=max_val, return_distance=True
        )
    else:
        ind, dist = kd_tree.query_radius(
            X=nodes[geo.coord_cols].values, r=max_val, return_distance=True
        )
    return ind, dist


def create_array(k: int, n: int):
    return np.concatenate((np.full(k, True), np.full(n - k, False)))


def get_dist_mask(all_dists, select_val, dist, conf: Config):
    """Returns a distance mask for computing neighbors.

    Args:
        all_dists: all distances of neighbors as 1-dim array
        select_val: radius or k
        dist: distances of neighbor as (reps, n_each) array
        conf: config

    If k_nearest it returns a repeated array with true
    values for the first kth entries repeated N times.
    If radius it returns all distances below the current
    `select_val`
    """
    if conf.k_nearest:
        n_each = dist.shape[1]
        reps = dist.shape[0]
        # Return the distance of the kth nearest
        return np.tile(create_array(select_val, n_each), reps=reps)
    return all_dists <= select_val


def get_col_name(conf: Config):
    if conf.k_nearest:
        # Return the distance of the kth nearest
        return "k"
    return "radius"


def condition_save_youth_edges(conf: Config, select_val: int):
    if (
        conf.youth_age in ["14", "15", "16", "17"]
        and conf.k_nearest
        and conf.save_edges
        and select_val in [30]  # Only for 30 (and below) closest
    ):
        return True
    return False


def save_youth_edges(
    conf: Config, edges: np.ndarray, batch_num: int, logger, select_val: int
):
    fname = (
        f"youth_edges{conf.file_suffix}_{conf.youth_age}"
        f"_{conf.start}-{conf.end}_k{select_val}_batch_{batch_num}"
    )
    data_utils.log_save_array(
        filename=fname, array=edges, logger=logger, fp=dstnx.fp.REG_DATA / "edges"
    )


def compute_batch(
    nodes: pd.DataFrame,
    batch_num: int,
    kd_tree: KDTree,
    neighbor_feats: pd.DataFrame,
    neighbors_times: np.ndarray,
    neighbor_ids: NeighborIds,
    cols: list[str],
    conf: Config,
):
    """Batch computes neighborhood measures."""
    # mp_logger = LOGGER
    mp_logger = log_utils.get_logger(
        name=f"mp_batch_{batch_num}_{conf.start}-{conf.end}"
    )

    select_vals = select_interval(conf)
    max_val = max(select_vals)
    ind, dist = query_tree(nodes, kd_tree, max_val, conf)

    cat_ind = np.concatenate(ind)
    all_dists = np.concatenate(dist)
    degrees = np.array([len(x) for x in ind])  # Have same length for k-nearest
    node_ids, idx_mapper = _construct_map_indices(nodes)
    node_idcs = np.repeat(node_ids, degrees)

    mask_overlap, overlaps = time_overlap_masks(
        nodes, node_idcs, neighbors_times, cat_ind
    )

    mask_exclude = parents_mask(
        nodes, neighbor_ids, node_idcs, cat_ind, conf=conf, mp_logger=mp_logger
    )

    # Intersect by overlapping and valid
    mask_valid = mask_overlap & mask_exclude

    arrays = []
    ids = []
    num_cols = 5 * len(cols) + 1  # 5 for measures; 1 for edge counts
    for select_val in select_vals:
        mp_logger.debug(
            f"Computing for ({select_val=}, {conf.k_nearest=}, {batch_num=})..."
        )
        dist_mask = get_dist_mask(all_dists, select_val, dist, conf)
        mask_dist = mask_valid & dist_mask
        features = neighbor_feats[cat_ind[mask_dist]]
        if condition_save_youth_edges(conf, select_val):
            mp_logger.debug(
                f"Saving edges for {select_val=}, {conf.k_nearest}, {conf.save_edges}"
            )
            edges = np.column_stack(
                # Map indices back to pids for the edges
                (
                    idx_mapper(node_idcs[mask_dist]),
                    cat_ind[mask_dist],
                    all_dists[mask_dist],
                )
            )
            save_youth_edges(
                conf, edges, batch_num, logger=mp_logger, select_val=select_val
            )
        # All those ids with entries in large arrays
        unqs = np.unique(node_idcs[mask_dist])
        bin_counts = np.bincount(node_idcs[mask_dist])[unqs]
        cum_bin_counts = bin_counts.cumsum()  # Cumsum the bins to get indices[i:i+1]
        indices_ufunc = cum_bin_counts - 1
        mask = np.isnan(features)
        nan_counts = np.add.reduceat(mask, indices_ufunc)
        counts = np.add.reduceat(~mask, indices_ufunc)

        features_masked = np.where(mask, 0, features)
        nan_sums = np.add.reduceat(
            features_masked, indices_ufunc
        )  # Sum masked featured
        nan_avgs = nan_sums / np.where(counts == 0, 1, counts)  # Handle all nan case
        nan_avgs = np.where(counts == 0, np.nan, nan_avgs)  # Replace 0 with nans

        # Weighted averages
        # Get sums for weighted average of non-nan
        overlaps_radius = overlaps[mask_dist]
        weight_counts = np.add.reduceat(
            np.where(
                mask, 0, np.repeat(overlaps_radius[:, None], repeats=len(cols), axis=1)
            ),
            indices_ufunc,
        )
        # Weights set to 0 for those with nan features
        weighted_sum = np.add.reduceat(
            features_masked * overlaps_radius[:, None], indices_ufunc
        )
        weighted_avg = weighted_sum / np.where(weight_counts == 0, 1, weight_counts)
        # Replace those without obs (with counts == 0) with nan
        weighted_avg = np.where(counts == 0, np.nan, weighted_avg)
        mp_logger.debug(
            f'{select_val=}: {datetime.datetime.now().strftime("%H:%M:%S")}'
        )

        computed_arrs = np.column_stack(
            (bin_counts, nan_avgs, counts, nan_counts, weight_counts, weighted_avg)
        )

        # Find idcs who were subsetted out
        if (set_diff := np.setdiff1d(node_ids, unqs)).size > 0:
            mp_logger.debug(
                f"#{set_diff.shape[0]} ids without obs in batch {batch_num}"
            )
            nan_arr = np.column_stack(
                (
                    # Those without had 0 edges
                    np.zeros(shape=(set_diff.shape[0], 1)),
                    np.full(shape=(set_diff.shape[0], num_cols - 1), fill_value=np.nan),
                )
            )
            unqs = np.concatenate((unqs, set_diff))
            computed_arrs = np.row_stack((computed_arrs, nan_arr))

        arrays.append(computed_arrs)
        ids.append(idx_mapper(unqs))

    cols_all = ["count"] + [
        f"{col}_{coltype}"
        for coltype in ["avg", "count", "nan_count", "weight_tot", "wavg"]
        for col in cols
    ]
    # Note: might be multiple observations for given id and radius if person lived
    # at multiple addresses in the period considered
    col_name = get_col_name(conf)
    df_all = pd.concat(
        pd.DataFrame(arr, columns=cols_all).assign(
            **{col_name: select_val, "PERSON_ID": _ids}
        )
        for arr, select_val, _ids in zip(arrays, select_vals, ids)
    )
    filename = (
        dstnx.fp.REG_DATA
        / "batches"
        / (
            f"features{conf.file_suffix}_{conf.neighbor_type}{conf.youth_age}"
            f"_{conf.start}-{conf.end}_{col_name}_batch_{batch_num}.parquet"
        )
    )
    df_all.to_parquet(filename)
    mp_logger.debug(
        f"Done {batch_num=}: saved file to {filename} of size {df_all.shape[0]}"
        f"and {np.unique(nodes['PERSON_ID'].values).shape[0]} unique ids"  # type: ignore
    )


# --------------------- Other  --------------------- #


def cap_time_years(df: pd.DataFrame, st: int, et: int) -> pd.DataFrame:
    return df.pipe(address_comp.cap_time_column, "BOP_VTIL", et + 1, upper=True).pipe(
        address_comp.cap_time_column, "BOP_VFRA", st, upper=False
    )


def prepare_nodes(nodes: pd.DataFrame, st: int, et: int):
    """Caps time columns and merges addresses onto nodes"""
    start, end = pd.Timestamp(f"{st}"), pd.Timestamp(f"{et}")
    return (
        address.subset_years(df=nodes, start=start, end=end)
        .copy()
        .pipe(cap_time_years, st=st, et=et)
        .pipe(address_comp.date_cols_to_int, geo.time_cols)
    )


def construct_parent_measures(nodes, neighbors, feat_cols, conf: Config):
    """Computes parent features.

    The parent features are the same as for adult neighbors + some
    family-specific variables.
    """
    id_cols = ["PERSON_ID", "age"]
    parent_upbringing = parents.ParentYouthUpbringing(
        nodes, neighbors, parent_feats=feat_cols
    )
    measures = parent_upbringing.add_parent_features(nodes)[
        id_cols + parent_upbringing.merged_feats + neighbors_mod.FM_MARK_COLS
    ].assign(year=conf.start)
    measures.to_parquet(
        dstnx.fp.REG_DATA
        / f"parent_measures{conf.file_suffix}_{conf.start}-{conf.end}.parquet"
    )
    LOGGER.debug(
        f"Saved parent measures of size {measures.shape[0]} for {conf.start}-{conf.end}"
    )


def compute_measures(nodes, neighbors, feat_cols, conf: Config):
    # Build KD-tree on all neighbors
    kd_tree = KDTree(neighbors[geo.coord_cols].values)

    # Batch nodes and construct neighborhood measures
    batcher = geo.Batcher(nodes, batch_size=conf.batch_size)
    neighbors_times = neighbors[geo.time_cols].values
    neighbor_feats = neighbors[feat_cols].astype(float).values
    neighbor_ids = NeighborIds(
        pid=neighbors["PERSON_ID"].values,
        famid=neighbors["FAMILIE_ID"].values,
    )
    batch_fn = partial(
        compute_batch,
        kd_tree=kd_tree,
        neighbor_feats=neighbor_feats,
        neighbors_times=neighbors_times,
        neighbor_ids=neighbor_ids,
        cols=feat_cols,
        conf=conf,
    )
    LOGGER.debug(f"Starting to compute ({conf.start=}, {conf.neighbor_type=}):")
    with Pool(4) as p:
        p.starmap(batch_fn, batcher.yield_batches())


class YouthBatcher:
    def __init__(self, nodes: pd.DataFrame, year_born: pd.DataFrame, year: int):
        self.nodes = nodes
        self.year_born = year_born
        self.year = year
        self.compute_ages()

    def compute_ages(self):
        self.year_diffs = self.year - self.year_born.year_born
        mask_born = self.year_diffs >= 0  # Only get those who are born
        mask_youth = self.year_diffs <= 18
        self.mask_valid = mask_born & mask_youth
        self.sorted_unqs_year = sorted(self.year_diffs.loc[self.mask_valid].unique())
        LOGGER.debug(
            f"Valid observations from 0-18 {self.mask_valid.sum()}\n"
            f"Non-born individuals:        {(self.year_diffs < 0).sum()}\n"
            f"Adults:                      {(~mask_youth).sum()}"
        )

    @property
    def youth_ages(self):
        return self.sorted_unqs_year

    @property
    def age_span(self):
        return min(self.youth_ages), max(self.youth_ages)

    def merge_year_born(self) -> pd.DataFrame:
        """Get the age of each individual in the given year.

        Only return for those in the valid 0-18 year range.
        """
        ages = (
            self.year_born.loc[self.mask_valid, ["PERSON_ID"]]
            .assign(age=self.year_diffs.loc[self.mask_valid])
            .drop_duplicates()
        )
        mask = ages.PERSON_ID.duplicated()
        if any_dups := mask.sum():
            LOGGER.debug(f"#{any_dups} duplicates in age df\nObs:{ages.loc[mask]}")
            ages = ages.loc[~mask].reset_index(drop=True)
        return self.nodes.merge(ages, how="left", on="PERSON_ID")

    def yield_batches(self):
        for age in self.sorted_unqs_year:
            mask = self.year_diffs == age
            pnrs = self.year_born.loc[mask, "PERSON_ID"].unique()
            nodes_age = self.nodes.loc[self.nodes.PERSON_ID.isin(pnrs)]
            LOGGER.debug(f"Yielding {pnrs.shape[0]} pnrs of {age=}")
            yield nodes_age, age


def construct_youth(
    nodes: pd.DataFrame,
    year_born: pd.DataFrame,
    geo_years: neighbors_mod.GeoYears,
    st: int,
    et: int,
    neighbor_type: str,
    file_suffix: str,
    batch_size: int,
    k_nearest: bool,
    save_edges: bool = False,
):
    feat_cols = get_feat_cols(neighbor_type)
    youth_batcher = YouthBatcher(nodes, year_born, year=st)
    min_age, max_age = youth_batcher.age_span
    neighbors = (
        geo_years.load_youth_interval(year=st, min_age=min_age, max_age=max_age)
        .loc[
            :,
            geo.id_col
            + geo.famid_col
            + geo.addr_col
            + geo.time_cols
            + geo.coord_cols
            + feat_cols
            + ["ALDER"],
        ]
        .pipe(cap_time_years, st=st, et=et)
        .dropna(subset=geo.addr_col + geo.coord_cols)
        .reset_index(drop=True)
        .pipe(address_comp.date_cols_to_int, geo.time_cols)
    )
    for nodes_age, age in youth_batcher.yield_batches():  # Loop through each age
        lb_age, ub_age = neighbors_mod._get_bounds(age)
        neighbors_age = neighbors_mod.query_age(neighbors, lb_age=lb_age, ub_age=ub_age)
        conf = Config(
            start=st,
            end=et,
            neighbor_type=neighbor_type,
            file_suffix=file_suffix,
            batch_size=batch_size,
            youth_age=str(int(age)),
            k_nearest=k_nearest,
            save_edges=save_edges,
        )
        LOGGER.info(f"Computing for youth {age=}")
        compute_measures(nodes_age, neighbors_age, feat_cols, conf)
    LOGGER.info(f"Finished computing measures for {st}-{et} ({file_suffix})")


def construct_adults(
    nodes: pd.DataFrame,
    year_born: pd.DataFrame,
    geo_years: neighbors_mod.GeoYears,
    st: int,
    et: int,
    neighbor_type: str,
    file_suffix: str,
    batch_size: int,
    k_nearest: bool,
    parents: bool,
):
    feat_cols = get_feat_cols(neighbor_type)
    neighbors = (
        geo_years.load_period(
            start_period=st, end_period=et, neighbor_type=neighbor_type
        )
        .loc[
            :,
            geo.id_col
            + geo.famid_col
            + geo.addr_col
            + geo.time_cols
            + geo.coord_cols
            + feat_cols,
        ]
        .pipe(cap_time_years, st=st, et=et)
        .dropna(subset=geo.addr_col + geo.coord_cols)
        .reset_index(drop=True)
        .pipe(address_comp.date_cols_to_int, geo.time_cols)
    )
    conf = Config(
        start=st,
        end=et,
        neighbor_type=neighbor_type,
        file_suffix=file_suffix,
        batch_size=batch_size,
        k_nearest=k_nearest,
    )
    # Compute parent measures
    if parents:
        construct_parent_measures(
            (
                YouthBatcher(nodes, year_born, year=st)
                .merge_year_born()
                # Merge FM-mark variable
                .pipe(
                    lambda df: df.merge(
                        geo_years.load_specific(
                            year=st,
                            # Load specific pnrs based on df
                            pnrs=df.PERSON_ID.unique(),
                            cols=["PERSON_ID", "FM_MARK"],
                        ),
                        how="left",
                        on="PERSON_ID",
                    )
                )
                .pipe(neighbors_mod.fam_mark)
            ),
            neighbors,
            feat_cols,
            conf,
        )
    # Compute network measures
    # TODO: For adults we actually end up computing for > 18 years!
    compute_measures(nodes, neighbors, feat_cols, conf)
    LOGGER.info(f"Finished computing measures for {st}-{et}  ({file_suffix})")


class ParentsMap:
    def __init__(self, suffix: str = "_new"):
        self.parents = (
            data_utils.load_reg(f"node_metadata{suffix}", as_pl=False)[
                ["PERSON_ID", "MOR_PID", "FAR_PID", "FAMILIE_ID"]
            ].drop_duplicates()
            # Fill nans with -99; when comparing the ids in the main function
            # the entries won't get dropped while no real ids equal -99.
            .assign(
                MOR_PID=lambda df: df.MOR_PID.fillna(-99),
                FAR_PID=lambda df: df.FAR_PID.fillna(-99),
                # Famids are str values
                FAMILIE_ID=lambda df: df.FAMILIE_ID.fillna("-99"),
            )
        )

    def assign_parents(self, nodes: pd.DataFrame):
        return nodes.merge(self.parents, on="PERSON_ID", how="left")


@click.command(name="neighbors")
@click.option("--suffix", default="", help="File suffix to get neighbors for")
@click.option("--neighbor-type", default="neighbor", help="Kind of neighbor")
@click.option("--spacing", default=1, help="End year")
@click.option("--file-suffix", default="", help="Suffix for saved file")
@click.option("--batch-size", default=10_000, help="Batch size")
@click.option("--start", default=None, help="Start year", type=int)
@click.option("--end", default=None, help="End year", type=int)
@click.option(
    "--k-nearest",
    default=False,
    is_flag=True,
    help="To compute k-nearest instead of radius",
    type=bool,
)
@click.option(
    "--parents",
    default=False,
    is_flag=True,
    help="Whether to compute parent measures",
    type=bool,
)
@click.option(
    "--save-edges",
    default=False,
    is_flag=True,
    help="Whether to save edges",
    type=bool,
)
def main_multiple_periods(
    suffix: str,
    neighbor_type: str,
    spacing: int,
    file_suffix: str,
    batch_size: int,
    start: int,
    end: int,
    k_nearest: bool,
    parents: bool,
    save_edges: bool,
):
    node_addr = data_utils.load_reg(f"address{suffix}", as_pl=False)
    year_born = data_utils.load_reg(f"year_born{suffix}", as_pl=False)
    years = address.get_years(year_born)
    if not (start and end):
        LOGGER.info("No start and end years provided, using min and max years")
        start, end = min(years), max(years)
    geoaddr = data_utils.load_reg("geoaddr", as_pl=False).drop_duplicates(
        subset=["ADRESSE_ID"]
    )
    # Nodes
    nodes = (
        node_addr.merge(
            geoaddr[geo.addr_col + geo.coord_cols], how="left", on="ADRESSE_ID"
        )
        .dropna(subset=geo.addr_col + geo.coord_cols)
        .reset_index(drop=True)
        # Merge parent ids for adult neighbors and famids for youth
        .pipe(ParentsMap().assign_parents)
    )
    year_ranges = [(st, st + spacing) for st in range(start, end, spacing)]
    LOGGER.info(
        f"Computing neighborhood measures for {neighbor_type=} and {k_nearest=} "
        f"in the intervals:\n{year_ranges}"
    )
    for st, et in year_ranges:
        geo_years = neighbors_mod.GeoYears(start=st, end=et)
        nodes_year = prepare_nodes(nodes, st=st, et=et)

        if neighbor_type == "neighbors":
            construct_adults(
                nodes_year,
                year_born,
                geo_years,
                st,
                et,
                neighbor_type,
                file_suffix,
                batch_size,
                k_nearest,
                parents,
            )
        else:
            construct_youth(
                nodes_year,
                year_born,
                geo_years,
                st,
                et,
                neighbor_type,
                file_suffix,
                batch_size,
                k_nearest,
                save_edges,
            )


if __name__ == "__main__":
    main_multiple_periods()
