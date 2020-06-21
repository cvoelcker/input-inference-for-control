import sys

import jax.numpy as np
import jax

seed = jax.random.PRNGKey(0)

def gradient(f):
    return jax.jacrev(f)

def hessian(f):
    return jax.jacfwd(gradient(f))

def diag_hessian(f, x):
    h = jax.vmap(hessian(f))(x)
    return jax.vmap(np.diag)(h)

def score_matching(f, x, weights):
    grads = jax.vmap(gradient(f))(x)
    quad_grads = diag_hessian(f, x)
    alpha = np.sum(weights * -quad_grads)/np.sum(weights * (grads ** 2))
    return alpha
