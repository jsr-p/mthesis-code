from dataclasses import dataclass
from typing import TypeAlias

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PYGDataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures

from dstnx import data_utils, network
from dstnx.models import choice, litbc, nn_utils, model_utils
from dstnx.models.model_utils import DSTCohortMask

BATCH_SIZE = 256
GROUP_COL = "KOM"


def id_map_from_df(df):
    person_ids = df.PERSON_ID.unique()
    return dict(zip(person_ids, np.arange(person_ids.shape[0])))


TorchData: TypeAlias = Data | Dataset
TorchLoader: TypeAlias = DataLoader | PYGDataLoader


def neighbor_loader(dataset, mode: str = "train", batch_size: int = 128):
    return NeighborLoader(
        dataset,
        # Sample X neighbors for each node for 2 iterations
        num_neighbors=[50] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=batch_size,
        input_nodes=getattr(dataset, f"{mode}_mask"),
        num_workers=8,
    )


@dataclass
class DSTTorchData:
    data: TorchData
    train: TorchLoader
    val: TorchLoader
    test: TorchLoader

    @classmethod
    def pyg_data(cls, target="eu_grad", interaction=True):
        full = choice.load_data("numpy_nullable", with_id=True)
        df = (
            data_utils.load_reg("node_metadata_class_new")
            .query("PERSON_ID in @full.PERSON_ID")
            .reset_index(drop=True)
        )
        id_map = id_map_from_df(df)
        df = (
            df.assign(person_id=df.PERSON_ID.map(id_map))
            .sort_values(by="person_id")
            .reset_index(drop=True)
        )
        assert df.person_id.notna().all()
        y, X = choice.prepare_sm(
            full, target, as_arrays=True, group_col=GROUP_COL, interaction=interaction
        )
        edges = network.edges_from_df(df, id_map=id_map)
        edges = (
            # (|E|, 2)
            torch.from_numpy(np.array(edges, dtype=np.int64))
            .long()
            .t()
            .contiguous()
        )
        X, y = nn_utils.tensors_from_numpy(X, y.ravel())
        masks = DSTCohortMask.from_df(full, as_torch=True)
        # X = model_utils.std_by_masks(X, masks)
        dataset = Data(
            x=X,
            y=y,
            edge_index=edges,
            train_mask=masks.train,
            val_mask=masks.val,
            test_mask=masks.test,
        )
        dataset = NormalizeFeatures()(dataset)
        assert dataset.validate(raise_on_error=True)
        return cls(
            dataset,
            train=neighbor_loader(dataset, mode="train", batch_size=BATCH_SIZE),
            val=neighbor_loader(dataset, mode="val", batch_size=BATCH_SIZE),
            test=neighbor_loader(dataset, mode="test", batch_size=BATCH_SIZE),
        )

    @classmethod
    def torch_data(cls, target="eu_grad", interaction=True):
        full = choice.load_data("numpy_nullable", with_id=True)
        y, X = choice.prepare_sm(
            full,
            target,
            as_arrays=True,
            group_col=GROUP_COL,
            interaction=interaction,
        )
        X, y = nn_utils.tensors_from_numpy(X, y.ravel())
        masks = DSTCohortMask.from_df(full, as_torch=True)
        dataset = nn_utils.BinaryClassificationData(X, y)
        return cls(
            dataset,
            train=nn_utils.create_data_loader(
                nn_utils.BinaryClassificationData(X[masks.train], y[masks.train]),
                batch_size=BATCH_SIZE,
            ),
            val=nn_utils.create_data_loader(
                nn_utils.BinaryClassificationData(X[masks.val], y[masks.val]),
                batch_size=BATCH_SIZE,
            ),
            test=nn_utils.create_data_loader(
                nn_utils.BinaryClassificationData(X[masks.test], y[masks.test]),
                batch_size=BATCH_SIZE,
            ),
        )


def fit_model(model_name: str, target: str, without_interaction: bool):
    match model_name:
        case "gcn":
            dataset = DSTTorchData.pyg_data(
                target=target, interaction=not without_interaction
            )
            model = litbc.BinaryClassifier(
                model=litbc.GCN(input_dim=dataset.data.x.shape[1], hidden_dim=64),
                gnn=True,
                batch_size=BATCH_SIZE,
            )
        case "gat":
            dataset = DSTTorchData.pyg_data(
                target=target, interaction=not without_interaction
            )
            model = litbc.BinaryClassifier(
                model=litbc.GAT(
                    input_dim=dataset.data.x.shape[1], hidden_dim=64, heads=4
                ),
                gnn=True,
                batch_size=BATCH_SIZE,
            )
        case "vanilla":
            dataset = DSTTorchData.torch_data(
                target=target, interaction=not without_interaction
            )
            model = litbc.BinaryClassifier(
                litbc.SimpleBinaryClassifier(
                    input_dim=dataset.data.X.shape[1], hidden_dim=64
                ),
                gnn=False,
                batch_size=BATCH_SIZE,
            )
        case _:
            raise ValueError(f"Model {model_name} does not exist!")

    # Train model
    litbc.train_model(
        model,
        train_loader=dataset.train,
        val_loader=dataset.val,
        test_loader=dataset.test,
        early_stopping=True,
        model_name=model_name,
        target=target,
    )


@click.group()
def cli():
    ...


@cli.command()
@click.argument("model_name", default="gcn")
@click.option("--target", default="eu_grad")
@click.option("--without-interaction", is_flag=True, default=False)
def estimate(model_name: str, target: str, without_interaction: bool):
    fit_model(model_name, target, without_interaction)


@cli.command()
def estimate_all():
    for model in ["gcn", "gat", "vanilla"]:
        for target in ["eu_grad", "gym_grad"]:
            fit_model(model, target, True)


if __name__ == "__main__":
    cli()
