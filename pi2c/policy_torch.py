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
        self.init_var = nn.Parameter(torch.log(torch.tensor(init_var, requires_grad=True)))
        self.mu_dampen = nn.Parameter(torch.tensor(1e-2), requires_grad=True)
        self.out_dim = out_dim
        self.i_counter = 0

    def forward(self, x, n):
        batch_dim = x.shape[0]
        b = self.net(x)
        mu = self.mu_head(b) * self.mu_dampen
        var = torch.exp(self.var_head(b) + self.init_var) + 0.1
        dist = Normal(mu, var)
        samples = dist.rsample((n, )).view(-1, self.out_dim)
        # if self.i_counter > 10000:
        #     print(x)
        #     print(mu)
        #     print(var)
        #     print(samples)
        #     print(self.mu_dampen)
        #     exit()
        # else:
        #     self.i_counter += 1
        # print(torch.any(torch.isnan(samples)))
        return samples

