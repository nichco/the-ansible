import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from itertools import combinations

def combo(variables: list) -> jnp.ndarray:
    """
    Compute the vector of pairwise differences between input variables.

    Given a sequence of arrays (blocks) [x0, x1, ..., x_{N-1}], this function
    computes all pairwise differences x_i - x_j for i < j, in the order
    produced by itertools.combinations(range(N), 2). The per-pair results are
    stacked into a JAX array and the final result is returned as a 1-D array
    (flattened).
    """

    num_blocks = len(variables)
    indices = list(range(num_blocks))
    pairs = list(combinations(indices, 2))

    # remove an arbitrary pair so the constraints are linearly independent
    # (e.g. remove the last pair)
    if len(pairs) > 1:
        print('removing one pair')
        pairs = pairs[:-1]
    
    c = jnp.array([variables[i] - variables[j] for i, j in pairs])
    return c.flatten()