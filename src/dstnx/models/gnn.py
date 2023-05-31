import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP, IMDB, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv

import dstnx


def load_dblp():
    fp = dstnx.fp.PROJ / "data" / "DBLP"
    dataset = DBLP(str(fp), transform=T.Constant(node_types="conference"))
    return dataset


def load_planetoid():
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    print(len(dataset))
    print(dataset.num_classes)
    print(dataset.num_node_features)
    print(dataset[0])
    return dataset


def load_imdb():
    dataset = IMDB(
        root="/tmp/imdb", transform=T.RandomNodeSplit(num_val=0.1, num_test=0.1)
    )
    print(dataset, dataset[0])
    return dataset


def test_loader(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        print(batch)
        print(batch.num_graphs)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train_model(dataset):
    model = GCN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # dataset = load_dblp()
    dataset = load_planetoi()
    # dataset = load_imdb()
    # test_loader(dataset)
    # train_model(dataset)
