from typing import Tuple, Callable, Optional, Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx

from ._sde import SDE

Array = jax.Array
Key = jr.PRNGKey
Module = eqx.Module


"""
    ODE objects
    - ODE terms from score network (general to all SDEs -> ODEs)
    - log_likelihood function p(x|q), depends on an SDE
"""


def get_solver() -> dfx.AbstractSolver:
    return dfx.Tsit5(scan_kind="bounded")

# from jaxtyping import Float

def logp_approx(
    t: float | Array, 
    y: Array, 
    args: Tuple[Array, Array, Module, Sequence[int]]
) -> Tuple[Array, Array]:
    """ 
        Approx. trace using Hutchinson's trace estimator. 
        - implement sampling multiple eps and averaging
    """
    y, _ = y 
    eps, q, func, data_shape = args 
    
    fn = lambda y: func(y.reshape(data_shape), t, q)
    f, f_vjp = jax.vjp(fn, y) # f = f(*primals)
    
    # Expectation over multiple eps
    if eps.ndim == len((1,) + tuple(data_shape)):
        (eps_dfdy,) = jax.vmap(f_vjp)(eps.reshape(len(eps), -1))
        print("eps dfdy", eps_dfdy.shape)
        # expectation would be mean over this for all eps
        logp = jax.vmap(
            lambda eps_dfdy, eps: jnp.sum(eps_dfdy * eps.flatten())
        )(eps_dfdy, eps).mean(axis=0)
    else:
        (eps_dfdy,) = f_vjp(eps.flatten())
        print("eps dfdy", eps_dfdy.shape)
        # expectation would be mean over this for all eps
        logp = jnp.sum(eps_dfdy * eps.flatten())
    return f, logp 


def logp_exact(
    t: float | Array, 
    y: Array, 
    args: Tuple[None, Array, Module, Sequence[int]]
) -> Tuple[Array, Array]:
    """ Compute trace directly. """
    y, _ = y
    _, q, func, data_shape = args

    print("exact", y.shape)

    fn = lambda y: func(y.reshape(data_shape), t, q)
    f, f_vjp = jax.vjp(fn, y)  

    print("exact f", f.shape)

    (dfdy,) = jax.vmap(f_vjp)(jnp.eye(y.size)) # J_f(x)?
    logp = jnp.trace(dfdy)

    print("prob", logp.shape)
    return f, logp


@eqx.filter_jit
def log_likelihood(
    key: Key, 
    score_network: eqx.Module, 
    sde: SDE,
    data_shape: Tuple[int], 
    y: Array, 
    q: Array, 
    exact_logp: bool,
    n_eps: Optional[int] = 10
) -> Tuple[Array, Array]:
    """ ODE is 'forward' to compute p(x) => drift f(x, t) of SDE """
    score_network = eqx.tree_inference(score_network, True)

    reverse_sde = sde.reverse(score_network, probability_flow=True)

    def func(y: Array, t: float | Array, q: Array) -> Array:
        print("func ytq", y.shape, t.shape, q.shape)
        t = jnp.atleast_1d(t)
        drift, _ = reverse_sde.sde(y, t, q)
        # drift = -0.5 * beta_fn(t) * (y + score_network(t, y, q))
        return drift.flatten()

    if not exact_logp:
        if n_eps is not None:
            eps_shape = (n_eps,) + y.shape 
        else:
            eps_shape = y.shape
        eps = jr.normal(key, eps_shape)
    else:
        eps = None

    # Likelihood: integrate ODE term forward in time? Change of vars => t=0->t=1
    sol = dfx.diffeqsolve(
        dfx.ODETerm(logp_exact if exact_logp else logp_approx),
        get_solver(), 
        t0=sde.t0,
        t1=sde.t1, 
        dt0=sde.dt, 
        y0=(y.flatten(), 0.), 
        args=(eps, q, func, data_shape),
        adjoint=dfx.DirectAdjoint()
    ) 
    (z,), (delta_log_likelihood,) = sol.ys
    print("dffs", sde.prior_log_prob(z).shape, delta_log_likelihood.shape)
    log_p_y = sde.prior_log_prob(z).sum() + delta_log_likelihood
    return z, log_p_y


def get_log_likelihood_fn(
    model: Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    exact_logp: bool = False,
    n_eps: int = None
) -> Callable:
    model = eqx.tree_inference(model, True)

    def _log_likelihood_fn(y, q, key):
        _, log_probs = log_likelihood(
            key, model, sde, data_shape, y, q, exact_logp, n_eps
        )
        return log_probs
    return _log_likelihood_fn


# @eqx.filter_jit
# def single_sample_ode_fn(model, sde, data_shape, q, key):
#     def drift(t, y, args):
#         (q,) = args
#         # beta = beta_fn(t) 
#         # return -0.5 * beta * (y + model(t, y, q))
#         drift = sde.reverse(model).sde(y, t, q)
#         # drift = -0.5 * beta_fn(t) * (y + score_network(t, y, q))
#         return drift


#     term = dfx.ODETerm(drift)
#     solver = get_solver() #dfx.Tsit5()
#     y1 = sde.prior_sampling(key, data_shape) # jr.normal(key, data_shape)
#     # reverse time, solve from t1 to t0
#     sol = dfx.diffeqsolve(term, solver, sde.t1, sde.t0, -sde.dt, y1, (q,))
#     return sol.ys[0]


# @eqx.filter_jit
# def batch_sample_ode_fn(model, sde, data_shape, Q, key):
#     keys = jr.split(key, len(Q))
#     return jax.vmap(
#         single_sample_ode_fn,
#         in_axes=(None, None, None, 0, 0)
#     )(model, sde, data_shape, Q, keys)
