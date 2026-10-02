import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """One vanilla GCN layer: ÂĤ A X W + b."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        messages = torch.matmul(adj_norm, x)
        return self.linear(messages)


class VanillaGCN(nn.Module):
    """Explicit two-layer vanilla GCN: GCN-1 -> ReLU -> GCN-2."""

    def __init__(self, in_features: int, hidden: int = 36, out_features: int = 2):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden)
        self.gc2 = GraphConvolution(hidden, out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h1 = self.gc1(x, adj_norm)
        h1 = F.relu(h1)
        return self.gc2(h1, adj_norm)

    def forward_with_activations(self, x: torch.Tensor, adj_norm: torch.Tensor):
        h1_linear = self.gc1(x, adj_norm)
        h1 = F.relu(h1_linear)
        logits = self.gc2(h1, adj_norm)
        return logits, h1
