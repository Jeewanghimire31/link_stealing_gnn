# gnn_training.py
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Load Dataset
dataset_name = "Cora"  # Change to Pubmed, DBLP, etc.
dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0]

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train GNN
def train_gnn(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            acc = test_gnn(model, data)
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")

# Test GNN
def test_gnn(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    return int(correct) / int(data.test_mask.sum())

# Main
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data.num_features, dataset.num_classes).to(device)
    data = data.to(device)

    print("Training GNN...")
    train_gnn(model, data)
    test_acc = test_gnn(model, data)
    print(f"GNN Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "gnn_model.pth")
    print("Model saved as gnn_model.pth")
