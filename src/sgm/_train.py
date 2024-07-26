from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import PyTree, Key, Array
import optax

from ._sde import SDE

Model = eqx.Module
OptState = optax.OptState
TransformUpdateFn = optax.TransformUpdateFn


def apply_ema(
    ema_model: Model, 
    model: Model, 
    ema_rate: float = 0.999
) -> Model:
    # Break models into parameters and 'architecture'
    m_, _m = eqx.partition(model, eqx.is_inexact_array)
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array)
    # Calculate EMA parameters
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    e_ = jtu.tree_map(ema_fn, e_, m_)
    # Combine EMA model parameters and architecture
    return eqx.combine(e_, _m)


def single_loss_fn(
    model: Model, 
    sde: SDE,
    x: Array, 
    q: Array, 
    t: Array, 
    key: Key
) -> Array:
    key_noise, key_apply = jr.split(key)
    mean, std = sde.marginal_prob(x, t) # std = jnp.sqrt(jnp.maximum(std, 1e-5)) 
    noise = jr.normal(key_noise, x.shape)
    y = mean + std * noise
    y_ = model(t, y, q, key=key_apply) # Inference is true in validation
    return sde.weight(t) * jnp.square(y_ + noise / std).mean()


def sample_time(
    key: Key, 
    t0: float, 
    t1: float, 
    n_sample: int
) -> Array:
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(key, (n_sample,), minval=t0, maxval=t1 / n_sample)
    t = t + (t1 / n_sample) * jnp.arange(n_sample)
    return t


@eqx.filter_jit
def batch_loss_fn(
    model: Model, 
    sde: SDE,
    x: Array, 
    q: Array, 
    key: Key
) -> Array:
    batch_size = x.shape[0]
    key_t, key_L = jr.split(key)
    keys_L = jr.split(key_L, batch_size)
    t = sample_time(key_t, sde.t0, sde.t1, batch_size)
    loss_fn = jax.vmap(partial(single_loss_fn, model, sde))
    return loss_fn(x, q, t, keys_L).mean()


@eqx.filter_jit
def make_step(
    model: Model, 
    sde: SDE,
    x: Array, 
    q: Array, 
    key: Key, 
    opt_state: OptState, 
    opt_update: TransformUpdateFn
) -> Tuple[Array, Model, Key, OptState]:
    model = eqx.tree_inference(model, False)
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, sde, x, q, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key, _ = jr.split(key)
    return loss, model, key, opt_state


@eqx.filter_jit
def evaluate(
    model: Model, 
    sde: SDE, 
    x: Array, 
    q: Array, 
    key: Key
) -> Array:
    model = eqx.tree_inference(model, True)
    loss = batch_loss_fn(model, sde, x, q, key)
    return loss 