from typing import Tuple, Callable, Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np 

Array = jax.Array


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, *, key):
        lim = jnp.sqrt(1. / (in_size + 1.))
        key_w, _ = jr.split(key)
        self.weight = jr.truncated_normal(
            key_w, shape=(out_size, in_size), lower=-2., upper=2.
        ) * lim
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


class ResidualNetwork(eqx.Module):
    _in: eqx.nn.Linear
    layers: Tuple[eqx.Module]
    dropouts: Tuple[eqx.nn.Dropout]
    _out: eqx.nn.Linear
    activation: Callable

    def __init__(
        self, 
        in_size: int, 
        width_size: int, 
        depth: int, 
        y_dim: int, 
        activation: Callable,
        dropout_p: float =0.,
        *, 
        key: jr.PRNGKey
    ):
        """ Time-embedding may be necessary """
        in_key, *net_keys, out_key = jr.split(key, 2 + depth)
        self._in = Linear(
            in_size + y_dim + 1, width_size, key=in_key
        )
        layers = [
            Linear(
                width_size + y_dim + 1, width_size, key=_key
            )
            for _key in net_keys 
        ]
        self._out = Linear(
            width_size, in_size, key=out_key
        )

        dropouts = [
            eqx.nn.Dropout(p=dropout_p) for _ in layers
        ]
        self.layers = tuple(layers)
        self.dropouts = tuple(dropouts)
        self.activation = activation
    
    def __call__(
        self, 
        t: float | Array, 
        x: Array, 
        y: Array, 
        *, 
        key: jr.PRNGKey = None
    ) -> Array:
        t = jnp.atleast_1d(t)
        xyt = jnp.concatenate([x, y, t])
        h0 = self._in(xyt)
        h = h0
        for l, d in zip(self.layers, self.dropouts):
            # Condition on time at each layer
            hyt = jnp.concatenate([h, y, t])
            h = l(hyt)
            h = d(h, key=key)
            h = self.activation(h)
        o = self._out(h0 + h)
        return o