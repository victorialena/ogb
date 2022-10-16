import torch
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    # def forward(self, x_i, x_j):
    #     x = x_i * x_j
    #     for lin in self.lins:
    #         x = F.relu(x)
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #         x = lin(x)
    #     return torch.sigmoid(x)

class BiLinearPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Bilinear(in_channels, in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = self.lins[0](x_i, x_j)
        for lin in self.lins[1:]:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
        return torch.sigmoid(x)


import sys
sys.path.append('/home/victorialena/') #homomorphicReadOut/')
import homomorphicReadOut as hrm

class IsoPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        in_group = hrm.get_feature_group_representations(in_channels)
        out_group = hrm.get_feature_group_invariants(1)
        repr_in = hrm.MatrixRepresentation(in_group, out_group)

        self.lins = torch.nn.ModuleList()
        self.lins.append(hrm.BasisLinear_1d(hidden_channels, group=repr_in, 
                                            basis='invariant', bias_init=True))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    # def forward(self, x_i, x_j):
    #     x = torch.cat([x_i, x_j], dim=1)
    #     for lin in self.lins[:-1]:
    #         x = lin(x)
    #         x = F.relu(x)
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.lins[-1](x)
    #     return torch.sigmoid(x)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
        return torch.sigmoid(x)

class SymPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        group = hrm.get_feature_group_representations(in_channels).representations

        self.lins = torch.nn.ModuleList()
        self.lins.append(hrm.SymLinear(group, in_channels, hidden_channels, bias=True))
        
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    # def forward(self, x_i, x_j):
    #     x = torch.cat([x_i, x_j], dim=1)
    #     for lin in self.lins:
    #         x = F.relu(x)
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #         x = lin(x)
    #     return torch.sigmoid(x)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)