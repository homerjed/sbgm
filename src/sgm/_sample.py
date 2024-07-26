from typing import Sequence, Callable
import diffrax as dfx  
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

from ._ode import get_solver
from ._sde import SDE


@eqx.filter_jit
def single_ode_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    key: Key,
    q: Array 
) -> Array:
    model = eqx.tree_inference(model, True)

    reverse_sde = sde.reverse(model, probability_flow=True)

    def ode(t, y, args):
        """
            dx = [f(x, t) - 0.5 * g^2(t) * score(x, t, q)] * dt
        """
        (q,) = args
        drift, _ = reverse_sde.sde(y, t, q)
        return drift

    term = dfx.ODETerm(ode)
    solver = get_solver() 
    y1 = sde.prior_sample(key, data_shape) 
    # Reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(
        term, solver, sde.t1, sde.t0, -sde.dt, y1, (q,)
    )
    return sol.ys[0]


@eqx.filter_jit
def single_eu_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    key: Key, 
    q: Array,
    T_sample: int = 1_000
) -> Array:
    """ Euler-Murayama sampler of reverse SDE """
    model = eqx.tree_inference(model, True)

    time_steps = jnp.linspace(sde.t1, sde.t0, T_sample) # Reversed time
    step_size = (sde.t1 - sde.t0) / T_sample 
    # keys = jr.split(key, T_sample)

    _, std_t1 = sde.marginal_prob(jnp.zeros(data_shape), sde.t1)
    # std_t1 = jnp.sqrt(jnp.maximum(std_t1, 1e-5)) # SDEs give std not var
    x = sde.prior_sample(key, data_shape) * std_t1 # x = x * std(t=1.)???
    reverse_sde = sde.reverse(model, probability_flow=False)

    def body_fn(i, val):
        """
            Euler Maryuma iteration:
            dx <- [f(x, t) - g^2(t) * score(x, t, q)] * dt + g(t) * sqrt(dt) * eps_t
            x <- x + dx
            t <- t + dt
        """
        (mean_x, x) = val
        t = time_steps[i]
        
        key_eps = jr.fold_in(key, i) # keys[i]
        eps_t = jr.normal(key_eps, data_shape)
        drift, diffusion = reverse_sde.sde(x, t, q)
        mean_x = x + drift * (-step_size)
        # x = [f(x, t) - g^2(t) * score(x, t, q)] * dt + g(t) * sqrt(dt) * eps_t
        x = mean_x + diffusion * jnp.sqrt(step_size) * eps_t

        return mean_x, x

    mean_x, x = jax.lax.fori_loop(
        0, T_sample, body_fn, init_val=(jnp.zeros_like(x), x)
    )

    # Do not include any noise in the last sampling step.
    return mean_x


def get_eu_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    T_sample: int = 1_000
) -> Callable:
    def _eu_sample_fn(key, q): 
        return single_eu_sample_fn(model, sde, data_shape, key, q, T_sample) 
    return _eu_sample_fn


def get_ode_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int]
) -> Callable:
    def _ode_sample_fn(key, q):
        return single_ode_sample_fn(model, sde, data_shape, key, q)
    return _ode_sample_fn