import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim=384,
        hidden_dim=64,
        output_dim=1,
        net_depth=2,
        net_activation=F.relu,
        weight_init="he_uniform",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_idx in range(net_depth):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layer = nn.Linear(in_dim, hidden_dim)
            if weight_init == "he_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            elif weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            else:
                raise NotImplementedError(f"Unknown weight initialization {weight_init}")
            self.layers.append(layer)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity="relu")
        self.net_activation = net_activation
        self.softplus = nn.Softplus()

    def forward(self, x):
        input_with_batch_dim = len(x.shape) == 4
        if not input_with_batch_dim:
            x = x.unsqueeze(0)

        batch_size, h, w, _ = x.shape
        x = x.reshape(-1, x.shape[-1])
        for layer in self.layers:
            x = self.net_activation(layer(x))
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.softplus(self.output_layer(x))
        x = x.reshape(batch_size, h, w)
        if not input_with_batch_dim:
            x = x.squeeze(0)
        return x


def generate_uncertainty_mlp(n_features, device="cuda"):
    return MLPNetwork(input_dim=n_features).to(device)

