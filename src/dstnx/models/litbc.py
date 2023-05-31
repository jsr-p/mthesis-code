from typing import Protocol

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

import dstnx
from dstnx import log_utils, data_utils
from dstnx.models import nn_utils

LOGGER = log_utils.get_logger(name=__name__)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
seed_everything(42)
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # type: ignore


def preds_from_logits(pred_proba):
    return torch.where(pred_proba >= 0.5, 1, 0)


class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim: int = 4):
        super(SimpleBinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.aucroc = torchmetrics.AUROC(task="binary", num_classes=2)

    def full_forward(self, batch) -> tuple[float, float, float]:
        X, y = batch
        logits = self.forward(X)
        loss = self.loss(logits, y)
        pred_proba = torch.sigmoid(logits)
        acc = self.accuracy(preds_from_logits(pred_proba), y)
        aucroc = self.aucroc(pred_proba, y)
        return loss, acc, aucroc

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, p=0.1)
        x = self.layer2(x)
        return x.squeeze()  # returns (N, ) tensor


class GCNBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.aucroc = torchmetrics.AUROC(task="binary", num_classes=2)

    def full_forward(self, batch, mode: str = "train") -> tuple[float, float, float]:
        logits = self.forward(batch)
        mask = self.get_mask(batch, mode)
        loss = self.loss(logits[mask], batch.y[mask])
        pred_proba = torch.sigmoid(logits)
        acc = self.accuracy(preds_from_logits(pred_proba)[mask], batch.y[mask])
        aucroc = self.aucroc(pred_proba[mask], batch.y[mask])
        return loss, acc, aucroc

    def get_mask(self, data, mode: str = "train"):
        return getattr(data, f"{mode}_mask")


class GCN(GCNBase):
    def __init__(self, input_dim, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze()  # returns (N, ) tensor


class GAT(GCNBase):
    def __init__(self, input_dim, hidden_dim, heads):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=input_dim, out_channels=hidden_dim, heads=heads
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=1,  # Binary classification
            heads=1,
        )

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        if return_attention_weights:
            return self.conv2(
                x, edge_index, return_attention_weights=return_attention_weights
            )
        return self.conv2(x, edge_index).squeeze()


class BinaryModel(Protocol):
    def full_forward(self, batch) -> tuple[float, float, float]:
        ...


# Define a PyTorch Lightning module
class BinaryClassifier(pl.LightningModule):
    def __init__(self, model: BinaryModel, gnn: bool = False, batch_size: int = 128):
        super(BinaryClassifier, self).__init__()
        self.model = model
        self.gnn = gnn
        self.batch_size = batch_size
        self._init()

    def _init(self):
        if self.gnn:
            self.train_args = {"mode": "train"}
            self.val_args = {"mode": "val"}
            self.test_args = {"mode": "test"}
        else:
            self.train_args = {}
            self.val_args = {}
            self.test_args = {}

    def training_step(self, batch, _):
        loss, acc, aucroc = self.model.full_forward(batch, **self.train_args)
        self.log_vals(loss, acc, aucroc, "train")
        return loss

    def validation_step(self, batch, _):
        loss, acc, aucroc = self.model.full_forward(batch, **self.val_args)
        self.log_vals(loss, acc, aucroc, "val")
        return loss

    def test_step(self, batch, _):
        loss, acc, aucroc = self.model.full_forward(batch, **self.test_args)
        self.log_vals(loss, acc, aucroc, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def log_vals(self, loss, acc, aucroc, mode: str):
        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{mode}_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{mode}_auc_roc",
            aucroc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )


# --------------------- Training --------------------- #


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    early_stopping: bool = True,
    max_epochs: int = -1,
    model_name: str = "gcn",
    target: str = "eu_grad",
):
    LOGGER.info(f"Training {model_name=} with {early_stopping=}")
    cb_checkpoint = ModelCheckpoint(
        dirpath=dstnx.fp.PL_MODELS,
        filename=f"{model_name}-{target}" + "{epoch}-{val_loss:.2f}",
    )
    logger = TensorBoardLogger(save_dir=dstnx.fp.PL, prefix=f"{model_name}-{target}")
    callbacks = [cb_checkpoint]
    if early_stopping:
        cb_earlystopping = EarlyStopping(monitor="val_loss", mode="min", patience=6)
        callbacks.append(cb_earlystopping)
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=callbacks, logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    result = trainer.test(model, dataloaders=test_loader, verbose=True)
    (result,) = result  # Unpack single element
    data_utils.log_save_json_append(
        filename=f"{target}-nn-res", obj=result, key=f"{model_name}"
    )


if __name__ == "__main__":
    # dataset = BinaryDataset(num_samples=1000, num_features=20)
    dataset = nn_utils.binary_data(iris=False)
    print("Training cases:", len(dataset))
    model = BinaryClassifier(
        model=SimpleBinaryClassifier(input_dim=dataset.X.shape[1]), gnn=False
    )
    # X, y = next(iter(train_loader))
    # print(X)
    train_loader, val_loader, test_loader = nn_utils.split_dataset(dataset)
    train_model(model, train_loader, val_loader, test_loader, early_stopping=False)
