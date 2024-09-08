from typing import Sequence, Callable, Optional
import diffrax as dfx  
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

from .sde import SDE
from ._ode import get_solver


@eqx.filter_jit
def single_ode_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    key: Key,
    q: Optional[Array] = None,
    a: Optional[Array] = None,
    solver: Optional[dfx.AbstractSolver] = None
) -> Array:
    """ Solve reverse ODE initial-value problem with prior sample """

    model = eqx.tree_inference(model, True)

    reverse_sde = sde.reverse(model, probability_flow=True)

    def ode(t, y, args):
        """
            dx = [f(x, t) - 0.5 * g^2(t) * score(x, t, q)] * dt
        """
        (q, a) = args
        drift, _ = reverse_sde.sde(y, t, q, a)
        return drift

    term = dfx.ODETerm(ode)
    solver = solver if solver is not None else get_solver() 
    y1 = sde.prior_sample(key, data_shape) 

    # Reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(
        term, solver, sde.t1, sde.t0, -sde.dt, y1, (q, a)
    )
    return sol.ys[0]


@eqx.filter_jit
def single_eu_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    key: Key, 
    q: Optional[Array] = None,
    a: Optional[Array] = None,
    T_sample: int = 1_000
) -> Array:
    """ Euler-Murayama sampler of reverse SDE """
    
    model = eqx.tree_inference(model, True)

    time_steps = jnp.linspace(sde.t1, sde.t0, T_sample) # Reversed time
    step_size = (sde.t1 - sde.t0) / T_sample 

    xT = sde.prior_sample(key, data_shape) 
    reverse_sde = sde.reverse(model, probability_flow=False)

    def marginal(i, val):
        """
            Euler Maryuma iteration:
            dx <- [f(x, t) - g^2(t) * score(x, t, q)] * dt + g(t) * sqrt(dt) * eps_t
            x <- x + dx
            t <- t + dt
        """
        (mean_x, x) = val
        t = time_steps[i]
        
        key_eps = jr.fold_in(key, i) 

        eps_t = jr.normal(key_eps, data_shape)
        drift, diffusion = reverse_sde.sde(x, t, q, a)
        mean_x = x - drift * step_size # mu_x = x + drift * -step

        # x = [f(x, t) - g^2(t) * score(x, t, q)] * dt + g(t) * sqrt(dt) * eps_t
        x = mean_x + diffusion * jnp.sqrt(step_size) * eps_t

        return mean_x, x

    mean_x, x = jax.lax.fori_loop(
        0, T_sample, marginal, init_val=(jnp.zeros_like(xT), xT)
    )

    # Do not include any noise in the last sampling step.
    return mean_x


def get_eu_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    T_sample: int = 1_000
) -> Callable:
    def _eu_sample_fn(key, q, a): 
        return single_eu_sample_fn(
            model, sde, data_shape, key, q, a, T_sample
        ) 
    return _eu_sample_fn


def get_ode_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int]
) -> Callable:
    def _ode_sample_fn(key, q, a):
        return single_ode_sample_fn(
            model, sde, data_shape, key, q, a
        )
    return _ode_sample_fn