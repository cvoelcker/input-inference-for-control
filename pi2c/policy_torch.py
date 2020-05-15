import torch
from torch import nn
from torch.distributions import Normal


class GaussianMlpPolicy(nn.Module):

    def __init__(self, layers, inp_dim, out_dim, init_mu, init_var):
        super().__init__()
        
        net_layers = []
        in_dim = inp_dim
        for layer in layers:
            net_layers.append(nn.Linear(in_dim, layer))
            in_dim = layer
            net_layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*net_layers)
        self.mu_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim)
                )
        self.var_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim))
        self.init_mu = 0# torch.Tensor(init_mu)
        self.init_var = init_var
        self.out_dim = out_dim

    def forward(self, x, n):
        batch_dim = x.shape[0]
        x = self.net(x)
        mu = self.mu_head(x) + self.init_mu
        var = torch.exp(self.var_head(x)) * self.init_var
        dist = Normal(mu, var)
        samples = dist.rsample((n, )).view(-1, self.out_dim)
        return samples

