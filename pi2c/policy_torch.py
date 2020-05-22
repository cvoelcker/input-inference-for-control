import torch
from torch import nn
from torch.distributions import Normal


class GaussianMlpPolicy(nn.Module):

    def __init__(self, layers, inp_dim, out_dim, init_mu, init_var):
        super().__init__()
        
        def init_weights(m):
            if type(m) == nn.Linear:
                #torch.nn.init.normal_(m.weight, 0.0, 0.00001)
                m.weight.data.fill_(0.0)
                m.bias.data.fill_(0.0)
        
        net_layers = []
        in_dim = inp_dim
        for layer in layers:
            net_layers.append(nn.Linear(in_dim, layer))
            # in_dim = layer
            # net_layers.append(nn.ReLU())
        self.net = nn.Sequential(*net_layers)
        # self.net.apply(init_weights)
        self.mu_head = nn.Sequential(
                # nn.Linear(in_dim, in_dim),
                # nn.ReLU(),
                nn.Linear(in_dim, out_dim)
                )
        # self.mu_head.apply(init_weights)
        # self.mu_dampen = nn.Parameter(torch.tensor(1e-10, requires_grad=True))
        self.var_head = nn.Sequential(
                # nn.Linear(in_dim, in_dim),
                # nn.ReLU(),
                nn.Linear(in_dim, out_dim))
        self.var_head.apply(init_weights)
        self.init_var = nn.Parameter(torch.log(torch.tensor(init_var, requires_grad=True)))
        self.out_dim = out_dim
        self.i_counter = 0

    def forward(self, x, n):
        batch_dim = x.shape[0]
        mu = self.mu_head(x)
        mu = torch.repeat_interleave(mu, n, 0)
        var = torch.exp(self.init_var) + 0.01
        dist = Normal(0, 1)
        samples = dist.sample((n*batch_dim, )).view(-1, self.out_dim)
        samples = mu + samples * var
        return samples

    def control(self, x):
        return self.mu_head(x)
