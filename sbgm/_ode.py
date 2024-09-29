from typing import Tuple, Callable, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import Key, Array

from .sde._sde import SDE


def get_solver() -> dfx.AbstractSolver:
    return dfx.Tsit5()


def log_prob_approx(
    t: Union[float, Array], 
    y: Array, 
    args: Tuple[Array, Array, Array, eqx.Module, Sequence[int]]
) -> Tuple[Array, Array]:
    """ 
        Approx. trace using Hutchinson's trace estimator. 
        - optional multiple-eps sample to average estimated log_prob over
    """
    y, _ = y 
    eps, q, a, func, data_shape = args 
    
    fn = lambda y: func(y.reshape(data_shape), t, q, a)
    f, f_vjp = jax.vjp(fn, y) # f = f(*primals)
    
    # Expectation over multiple eps
    if eps.ndim == len((1,) + tuple(data_shape)):
        (eps_dfdy,) = jax.vmap(f_vjp)(eps.reshape(len(eps), -1))
        # Expectation would be mean over this for all eps
        log_probs = jax.vmap(
            lambda eps_dfdy, eps: jnp.sum(eps_dfdy * eps.flatten())
        )(eps_dfdy, eps)
        log_prob = log_probs.mean(axis=0)
    else:
        (eps_dfdy,) = f_vjp(eps.flatten())
        log_prob = jnp.sum(eps_dfdy * eps.flatten())
        
    return f, log_prob


def log_prob_exact(
    t: Union[float, Array], 
    y: Array, 
    args: Tuple[None, Array, Array, eqx.Module, Sequence[int]]
) -> Tuple[Array, Array]:
    """ Compute trace directly. """
    y, _ = y
    _, q, a, func, data_shape = args

    fn = lambda y: func(y.reshape(data_shape), t, q, a)
    f, f_vjp = jax.vjp(fn, y)  

    (dfdy,) = jax.vmap(f_vjp)(jnp.eye(y.size)) 
    log_prob = jnp.trace(dfdy)

    return f, log_prob


def get_ode(model: eqx.Module, sde: SDE) -> Callable:
    reverse_sde = sde.reverse(model, probability_flow=True)

    def ode(
        y: Array, 
        t: Union[float, Array], 
        q: Optional[Array] = None, 
        a: Optional[Array] = None
    ) -> Array:
        t = jnp.atleast_1d(t)
        drift, _ = reverse_sde.sde(y, t, q, a)
        return drift.flatten()

    return ode


@eqx.filter_jit
def log_likelihood(
    key: Key, 
    model: eqx.Module, 
    sde: SDE,
    data_shape: Tuple[int], 
    x: Array, 
    q: Optional[Array] = None, 
    a: Optional[Array] = None, 
    exact_logp: bool = False,
    n_eps: Optional[int] = 10,
    solver: Optional[dfx.AbstractSolver] = None
) -> Tuple[Array, Array]:
    """
        Computes the log-likelihood of data by solving an ordinary differential equation (ODE) 
        related to the reverse SDE paramterised with a score network. The function supports exact 
        and approximate log-likelihood computation using multiple noise realizations (`eps`).

        Parameters:
        -----------
        `key` : `Key`
            A JAX random key used for generating noise (eps) during the log-likelihood approximation.
        
        `model` : `eqx.Module`
            The trained model, typically a score-based generative model, used to compute the likelihood.
        
        `sde` : `SDE`
            The stochastic differential equation (SDE) defining the forward and reverse diffusion dynamics.
        
        `data_shape` : `Tuple[int]`
            Shape of the input data (x), required for reconstructing dimensions during the ODE solution.
        
        `x` : `Array`
            The input data for which the log-likelihood is being computed.
        
        `q` : `Optional[Array]`, default: `None`
            Optional conditioning variable `q` related to the input data, if applicable.
        
        `a` : `Optional[Array]`, default: `None`
            Optional conditioning variable `a` related to the input data, if applicable.
        
        `exact_logp` : `bool`, default: `False`
            If `True`, computes the exact log-likelihood. Otherwise, uses an approximation with noise realizations.
        
        `n_eps` : `Optional[int]`, default: `10`
            The number of noise (`eps`) realizations used for approximating the log-likelihood when `exact_logp` is `False`. 
            Ignored if `exact_logp` is `True`.
        
        `solver` : `Optional[dfx.AbstractSolver]`, default: `None`
            The differential equation solver to be used for solving the ODE. If `None`, a default solver is used.

        Returns:
        --------
        `z` : `Array`
            The transformed latent variable `z` after solving the ODE.
        
        `log_p_x` : `Array`
            The log-likelihood of the input data `x` computed by combining the prior log probability of `z` 
            and the change in log-likelihood during the ODE solution.

        Notes:
        ------
        - The function works by solving an initial value problem (IVP) for the ODE that corresponds to the 
          log-likelihood of the data.
        - If `exact_logp` is `False`, the log-likelihood is approximated by averaging over multiple noise 
          realizations (`eps`).
        - The `solver` parameter allows for flexible ODE solving, and by default, a suitable solver is chosen 
          (`diffrax.Tsit5()`) if not provided.
    """

    model = eqx.nn.inference_mode(model, True)

    ode = get_ode(model, sde)

    # TODO: multiple eps realisations for averaging
    if not exact_logp:
        if n_eps is not None:
            eps_shape = (n_eps,) + x.shape 
        else:
            eps_shape = x.shape
        eps = jr.normal(key, eps_shape)
    else:
        eps = None

    # Likelihood from solving initial value problem
    sol = dfx.diffeqsolve(
        dfx.ODETerm(
            log_prob_exact if exact_logp else log_prob_approx
        ),
        solver if solver is not None else get_solver(), 
        t0=sde.t0,
        t1=sde.t1, 
        dt0=sde.dt, 
        y0=(x.flatten(), 0.), # Data and initial change in log_prob
        args=(eps, q, a, ode, data_shape),
        # adjoint=dfx.DirectAdjoint()
    ) 
    (z,), (delta_log_likelihood,) = sol.ys
    p_z = sde.prior_log_prob(z).sum()
    log_p_x = p_z + delta_log_likelihood 
    return z, log_p_x


def get_log_likelihood_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    exact_logp: bool = False,
    n_eps: Optional[int] = None,
    solver: Optional[dfx.AbstractSolver] = None
) -> Callable:
    """
        Returns a parameterized log-likelihood function that computes the log-likelihood 
        of input data based on the provided model and stochastic differential equation (SDE).

        Parameters:
        -----------
        `model` : `eqx.Module`
            The trained score network model used to compute the log-likelihood.
        
        `sde` : `SDE`
            The stochastic differential equation (SDE) defining the forward diffusion dynamics.
        
        `data_shape` : `Sequence[int]`
            Shape of the input data for which the log-likelihood will be computed.
        
        `exact_logp` : `bool`, default: `False`
            If `True`, the returned function will compute the exact log-likelihood. Otherwise, it will use 
            an approximation with multiple noise realizations.
        
        `n_eps` : `Optional[int]`, default: `None`
            The number of noise realizations (`eps`) to use when approximating the log-likelihood. Ignored if `exact_logp` is `True`.
        
        `solver` : `Optional[dfx.AbstractSolver]`, default: `None`
            The differential equation solver to use in the returned function for solving the ODE. If `None`, a default solver is used.

        Returns:
        --------
        `Callable`
            A parameterized function that computes the log-likelihood of input data `x`, conditioned on optional 
            parameters `q` and `a`, and using the provided random key `key`.

        The returned function has the following signature:

        ```python
        def _log_likelihood_fn(x: Array, q: Optional[Array], a: Optional[Array], key: Key) -> Array:
            '''
            Computes the log-likelihood of input data `x`, optionally conditioned on `q` and `a`.
            
            Parameters:
            -----------
            `x` : `Array`
                Input data for which to compute the log-likelihood.
            
            `q` : `Optional[Array]`
                Optional conditioning variable `q`, if applicable.
            
            `a` : `Optional[Array]`
                Optional conditioning variable `a`, if applicable.
            
            `key` : `Key`
                A JAX random key for sampling noise during log-likelihood approximation.

            Returns:
            --------
            `log_probs` : `Array`
                The computed log-likelihood of the input data `x`.
            '''
        ```
    """
    def _log_likelihood_fn(x: Array, q: Optional[Array], a: Optional[Array], key: Key) -> Array:
        _, log_probs = log_likelihood(
            key, model, sde, data_shape, x, q, a, exact_logp, n_eps, solver
        )
        return log_probs
    return _log_likelihood_fn