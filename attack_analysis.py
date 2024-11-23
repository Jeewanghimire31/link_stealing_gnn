# attack_analysis.py
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.nn import Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load Dataset
dataset_name = "Cora"  # Change to Pubmed, DBLP, etc.
dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0]

# Load Trained GNN Model
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_features, dataset.num_classes).to(device)
model.load_state_dict(torch.load("gnn_model.pth"))
data = data.to(device)
model.eval()
print("Loaded trained GNN model.")

# Generate Node Pairs for Attacks
pairs = torch.randint(0, data.num_nodes, (5000, 2))  # Example pairs
labels = torch.randint(0, 2, (5000,))  # Example labels

# Extract Features for Attack Model
def extract_attack_features(pairs, model, data):
    posteriors = model(data).detach()
    features = []
    for u, v in pairs:
        f_u, f_v = posteriors[u], posteriors[v]
        features.append(torch.cat([f_u, f_v], dim=0))
    return torch.stack(features)

features = extract_attack_features(pairs, model, data)

# Train and Evaluate Attack Model
def train_attack_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    mlp = Sequential(
        Linear(features.size(1), 64),
        ReLU(),
        Linear(64, 1)
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

    for epoch in range(100):
        mlp.train()
        optimizer.zero_grad()
        preds = mlp(torch.tensor(X_train, dtype=torch.float32)).squeeze()
        loss = F.binary_cross_entropy_with_logits(preds, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            mlp.eval()
            preds_test = mlp(torch.tensor(X_test, dtype=torch.float32)).squeeze()
            auc = roc_auc_score(y_test, torch.sigmoid(preds_test).detach().numpy())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")

    return mlp

print("Training attack model...")
attack_model = train_attack_model(features, labels)
