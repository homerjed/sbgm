from typing import Tuple, Callable, Union, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        *, 
        key: Key
    ):
        lim = jnp.sqrt(1. / (in_size + 1.))
        key_w, _ = jr.split(key)
        self.weight = jr.truncated_normal(
            key_w, shape=(out_size, in_size), lower=-2., upper=2.
        ) * lim
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias


class ResidualNetwork(eqx.Module):
    _in: Linear
    layers: Tuple[Linear]
    dropouts: Tuple[eqx.nn.Dropout]
    _out: Linear
    activation: Callable
    a_dim: Optional[int] = None

    def __init__(
        self, 
        in_size: int, 
        width_size: int, 
        depth: Optional[int] = None, 
        a_dim: Optional[int] = None, 
        activation: Callable = jax.nn.tanh,
        dropout_p: float = 0.,
        *, 
        key: Key
    ):
        """ Time-embedding may be necessary """
        in_key, *net_keys, out_key = jr.split(key, 2 + depth)
        self._in = Linear(
            in_size + a_dim + 1 if a_dim is not None else in_size + 1, 
            width_size, 
            key=in_key
        )
        layers = [
            Linear(
                width_size + a_dim + 1 if a_dim is not None else width_size + 1, 
                width_size, 
                key=_key
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
        self.a_dim = a_dim
    
    def __call__(
        self, 
        t: Union[float, Array], 
        x: Array, 
        q: Optional[Array], 
        a: Optional[Array], 
        *, 
        key: Key = None
    ) -> Array:
        t = jnp.atleast_1d(t)
        if a is not None and self.a_dim is not None:
            xat = jnp.concatenate([x, a, t]) 
        else:
            xat = jnp.concatenate([x, t])
        h0 = self._in(xat)
        h = h0
        for l, d in zip(self.layers, self.dropouts):
            # Condition on time at each layer
            if a is not None and self.a_dim is not None:
                hat = jnp.concatenate([h, a, t]) 
            else:
                hat = jnp.concatenate([h, t])
            h = l(hat)
            h = d(h, key=key)
            h = self.activation(h)
        o = self._out(h0 + h)
        return o