from jax import numpy as np
from jax.random import PRNGKey, normal
from jax import grad
from jax.experimental import optimizers

from pi2c.policy_distributions import MLP

KEY = PRNGKey(0)

mlp = MLP(4, 2, 16, KEY)

assert mlp.params is not None

inp = normal(KEY, (16, 4))

mlp.conditional_sample(inp)

