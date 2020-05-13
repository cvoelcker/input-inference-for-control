from tqdm import tqdm

from jax import numpy as np
from jax.random import PRNGKey, normal
from jax import grad, jit
from jax.experimental import optimizers

from pi2c.policy_distributions import MLP

KEY = PRNGKey(0)

mlp = MLP(4, 2, 16, KEY)

assert mlp.params is not None

inp = normal(KEY, (16, 4))

def loss(params, nets, x):
    mu, var = MLP.conditional_sample(params, nets, x)
    mu_loss = np.sum(mu**2)
    var_loss = np.sum(var**2)
    return mu_loss + var_loss

for i in tqdm(range(100000)):
    res = grad(loss, argnums=0)(mlp.params, mlp.nets, inp)
    mlp.update_params(res)

print(loss(inp))
