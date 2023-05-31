import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from dstnx import data_utils, log_utils
from dstnx.models import choice
from dstnx.plots.rankplots import load_plot_data


def load_par():
    suffix = "_new"
    full = load_plot_data()
    assert full.PERSON_ID.duplicated().sum() == 0
    cohorts = data_utils.load_reg(f"cohorts{suffix}", as_pl=False)
    merged = full.merge(
        cohorts[["PERSON_ID", "MOR_PID", "FAR_PID"]].drop_duplicates(
            subset="PERSON_ID"
        ),
        how="left",
        on="PERSON_ID",
    )
    return merged


def sample_data():
    merged = load_par()
    parents = dict()
    cols = ["inc_14_17", "highest_edu_pria_14_17"]
    # merged = choice.clean_arrow_na(
    #     merged, features=[f"own_{par}_{col}" for par in ["mor", "far"] for col in cols]
    # )
    edges = []
    for par in ["mor", "far"]:
        par_col = f"{par.upper()}_PID"
        par_edges = (
            merged[["PERSON_ID", par_col]]
            .dropna()
            .drop_duplicates()
            .astype(np.int64)
            .to_numpy()
        )
        edges.append(par_edges)
        par_cols = [f"{par.upper()}_PID"] + [f"own_{par}_{col}" for col in cols]
        parent = merged[par_cols]
        parent.columns = ["pid", "inc", "highest_edu"]
        parents[par] = parent
    all_parents = (
        pd.concat(v for v in parents.values()).drop_duplicates(subset="pid").dropna()
    )
    all_edges = np.row_stack(edges)
    # all_edges = edges[
    #     np.isin(all_edges, all_parents.pid)
    # ]  # Drop edges for parents who got dropped
    # all_edges = edges[np.isin(all_edges, all_parents.pid.values)]  # Drop edges for parents who got dropped
    print(f"#ParentEdges: {all_edges.shape[0]}")
    print(f"#Parents: {all_parents.shape[0]}")
    return merged, all_parents, all_edges


if __name__ == "__main__":
    merged, parents, edges = sample_data()
    data = HeteroData()
    data["parent"].x = torch.tensor(
        parents[["inc", "highest_edu"]].astype(np.float64).values, dtype=torch.float
    )
    print(data)
