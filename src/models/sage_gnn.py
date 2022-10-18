import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, SAGEConv


class SAGEGNNEncoder(torch.nn.Module):
    def __init__(
        self,
        data,
        hidden_channels,
        out_channels,
        dropout=0.0,
        normalize=False,
        project=False
    ):
        super().__init__()

        self.dropout_fn = torch.nn.Dropout(dropout)
        self.convs = torch.nn.ModuleList()
        #self.lins = torch.nn.ModuleList()

        for hidden_channel in hidden_channels:
            self.convs.append(
                SAGEConv(
                    (-1, -1),
                    hidden_channel,
                    normalize=normalize,
                    project=False,
                    aggr="max"
                )
            )
            #self.lins.append(Linear(-1, hidden_channel))

        self.conv2 = SAGEConv(
            (-1, -1), out_channels, normalize=normalize, project=False, aggr="max"
        )
        #self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = self.dropout_fn(conv(x, edge_index))
            x = F.relu(x)

        x = self.dropout_fn(self.conv2(x, edge_index))
        x = F.relu(x)
        return x
        # for lin, conv in zip(self.lins, self.convs):
        #     x = self.dropout_fn(conv(x, edge_index) + lin(x))
        #     x = x.relu()

        # x = self.dropout_fn(self.conv2(x, edge_index) + self.lin2(x))
        # x = x.relu()
        # return x
            