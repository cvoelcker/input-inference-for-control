import torch
from torch import nn
from torch.distributions import Normal


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


class Policy(nn.Module):

    def __init__(self, inp_dim, out_dim, var_init, var_strategy='log', mu_strategy='linear', layers=tuple(), offset=0.5):
        super().__init__()
        if mu_strategy == 'linear':
            self.mu = LinearMu(inp_dim, out_dim)
        elif mu_strategy == 'mlp':
            assert len(layer) > 0, 'Cannot compute MLP without layers'
            self.mu = MlpMu(layers, inp_dim, out_dim)
        else:
            raise ValueError('Strategy {} unknown for mean'.format(mu_strategy))

        if var_strategy == 'log':
            self.var = LogVar(var_init, offset)
        elif var_strategy == 'square':
            self.var = SquareVar(var_init, offset)
        else:
            raise ValueError('Strategy {} unknown for var'.format(var_strategy))
        self.out_dim = out_dim
            
    def forward(self, x, n):
        batch_dim = x.shape[0]
        mu = self.mu(x, n)
        var = self.var()
        samples = Normal(0, 1).sample((n * batch_dim,)).view(-1, self.out_dim)
        return mu + samples * var


class LogLinearPolicy(nn.Module):
    def __init__(self, inp_dim, out_dim, var_init, offset=0.5):
        super().__init__()
        self.p = Policy(inp_dim, out_dim, var_init, 'log', 'linear', [], offset)

    def forward(self, x, n):
        return self.p(x, n)


class SquareLinearPolicy(nn.Module):
    def __init__(self, inp_dim, out_dim, var_init):
        super().__init__()
        self.p = Policy(inp_dim, out_dim, var_init, var_strategy='square')

    def forward(self, x, n):
        return self.p(x, n)


class MlpMu(nn.Module):

    def __init__(self, layers, inp_dim, out_dim):
        super().__init__()
        
        net_layers = []
        in_dim = inp_dim
        for layer in layers:
            net_layers.append(nn.Linear(in_dim, layer))
            in_dim = layer
            net_layers.append(nn.ReLU())
        net_layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*net_layers)
        self.net.apply(init_weights)
        self.out_dim = out_dim

    def forward(self, x, n):
        batch_dim = x.shape[0]
        x = self.net(x)
        mu = self.mu_head(x)
        mu = torch.repeat_interleave(mu, n, 0)
        return mu

    def control(self, x):
        return self.net(x)


class LinearMu(nn.Module):

    def __init__(self, inp_dim, out_dim):
        super().__init__()
        
        self.mu_head = nn.Sequential(
                nn.Linear(inp_dim, out_dim)
                )
        self.mu_head.apply(init_weights)
        self.out_dim = out_dim

    def forward(self, x, n):
        batch_dim = x.shape[0]
        mu = self.mu_head(x)
        mu = torch.repeat_interleave(mu, n, 0)
        return mu

    def control(self, x):
        return self.mu_head(x)

    
class LogVar(nn.Module):
    def __init__(self, var_init, min_offset=0.5):
        super().__init__()
        self.var = nn.Parameter(torch.log(torch.tensor(var_init, requires_grad=True)))
        self.offset = min_offset

    def forward(self):
        return torch.exp(self.var) + self.offset


class SquareVar(nn.Module):

    def __init__(self, var_init, min_offest=0.5):
        super().__init__()
        self.var = nn.Parameter(torch.log(torch.tensor(var_init, requires_grad=True)))
        self.offset = min_offset

    def forward(self):
        return self.var ** 2 + self.offset
