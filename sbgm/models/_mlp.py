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
    q_dim: Optional[int] = None
    a_dim: Optional[int] = None
    t1: float

    def __init__(
        self, 
        in_size: int, 
        width_size: int, 
        depth: Optional[int], 
        q_dim: Optional[int] = None,
        a_dim: Optional[int] = None, 
        activation: Callable = jax.nn.tanh,
        dropout_p: float = 0.,
        t1: float = 1.,
        *, 
        key: Key
    ):
        """
            Parameters:
            -----------
            `in_size` : `int`
                The size of the input features to the network.
            
            `width_size` : `int`
                The width (number of units) of each hidden layer in the network.
            
            `depth` : `Optional[int]`, default: `None`
                The number of residual layers in the network. 
            
            `q_dim` : `Optional[int]`, default: `None`
                The number of elements in the optional conditioning input.
            
            `a_dim` : `Optional[int]`, default: `None`
                The number of parameters in the optional conditioning.
            
            `activation` : `Callable`, default: `jax.nn.tanh`
                The activation function applied after each layer. Defaults to the hyperbolic tangent (`tanh`).
            
            `dropout_p` : `float`, default: `0.`
                The probability of dropping out units in the dropout layers. Defaults to `0` (no dropout).

            `t1` : float, default: `1.0`
                Default maximum time of diffusion process. Scales input times.
            
            `key` : `Key`
                JAX random key used for initialization.
        """

        in_key, *net_keys, out_key = jr.split(key, 2 + depth)

        _in_size = in_size + 1 # TODO: time embedding
        if q_dim is not None:
            _in_size += q_dim
        if a_dim is not None:
            _in_size += a_dim

        _width_size = width_size + 1 # TODO: time embedding
        if q_dim is not None:
            _width_size += q_dim
        if a_dim is not None:
            _width_size += a_dim

        self._in = Linear(_in_size,width_size, key=in_key)
        layers = [
            Linear(_width_size, width_size, key=_key)
            for _key in net_keys 
        ]
        self._out = Linear(width_size, in_size, key=out_key)
        self.layers = tuple(layers)

        dropouts = [
            eqx.nn.Dropout(p=dropout_p) for _ in layers
        ]
        self.dropouts = tuple(dropouts)

        self.activation = activation
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.t1 = t1
    
    def __call__(
        self, 
        t: Union[float, Array], 
        x: Array, 
        q: Optional[Array], 
        a: Optional[Array], 
        *, 
        key: Key = None
    ) -> Array:
        t = jnp.atleast_1d(t / self.t1)

        _qa = []
        if q is not None and self.q_dim is not None:
            _qa.append(q)
        if a is not None and self.a_dim is not None:
            _qa.append(a)

        xqat = jnp.concatenate([x, t] + _qa)
        
        h0 = self._in(xqat)
        h = h0
        for l, d in zip(self.layers, self.dropouts):
            # Condition on time at each layer
            hqat = jnp.concatenate([h, t] + _qa)

            h = l(hqat)
            h = d(h, key=key)
            h = self.activation(h)
        o = self._out(h0 + h)
        return o