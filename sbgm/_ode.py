from typing import Tuple, Callable, Optional, Sequence, Union, Literal
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from .sde._sde import SDE

"""
    Methods for sampling from and computing the log-likelihood of data
    using a score-based model and a ordinary differential equation (SDE)
    associated with a stochastic differential equation (SDE).
    - Hutchinson's trace estimator is used to approximate the log-likelihood
      if computing the trace directly is not computationally feasible.
    - The ODE is solved using a numerical solver (e.g., Tsit5) to compute the log-likelihood
      given a sample x and conditioning variables q and a.
"""


def get_solver() -> dfx.AbstractSolver:
    return dfx.Tsit5()


def log_prob_approx(
    t: Union[float, Scalar], 
    y: Float[Array, "..."], 
    args: Tuple[
        Float[Array, "..."], 
        Optional[Float[Array, "..."]], 
        Optional[Float[Array, "..."]], 
        Callable[
            [
                Float[Array, "..."], 
                Scalar, 
                Optional[Float[Array, "..."]], 
                Optional[Float[Array, "..."]]
            ], 
            Float[Array, "..."]
        ],
        Sequence[int]
    ]
) -> Tuple[Float[Array, "..."], Scalar]:
    """ 
        Approx. trace using Hutchinson's trace estimator. 
        - optional multiple-eps sample to average estimated log_prob over
    """
    y, _ = y 
    eps, q, a, ode, data_shape = args 
    
    fn = lambda y: ode(y.reshape(data_shape), t, q, a)
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
    t: Union[float, Scalar], 
    y: Float[Array, "..."], 
    args: Tuple[
        None, 
        Optional[Float[Array, "..."]], 
        Optional[Float[Array, "..."]], 
        Callable[
            [
                Float[Array, "..."], 
                Scalar, 
                Optional[Float[Array, "..."]], 
                Optional[Float[Array, "..."]]
            ], 
            Float[Array, "..."]
        ],
        Sequence[int]
    ]
) -> Tuple[Float[Array, "..."], Scalar]:
    """ 
        Compute trace directly. 
    """
    y, _ = y
    _, q, a, ode, data_shape = args

    fn = lambda y: ode(y.reshape(data_shape), t, q, a)
    f, f_vjp = jax.vjp(fn, y)  

    (dfdy,) = jax.vmap(f_vjp)(jnp.eye(y.size)) 
    log_prob = jnp.trace(dfdy)

    return f, log_prob


def get_ode(
    model: eqx.Module, 
    sde: SDE
) -> Callable[
    [
        Float[Array, "..."], 
        Scalar, 
        Optional[Float[Array, "..."]], 
        Optional[Float[Array, "..."]]
    ], 
    Float[Array, "..."]
]:
    reverse_sde = sde.reverse(model, probability_flow=True)

    def ode(
        y: Float[Array, "..."], 
        t: Union[float, Scalar], 
        q: Optional[Array] = None, 
        a: Optional[Array] = None
    ) -> Float[Array, "..."]:
        drift, _ = reverse_sde.sde(y, t, q, a)
        return drift.flatten()

    return ode


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def log_likelihood(
    key: Optional[PRNGKeyArray], 
    model: eqx.Module, 
    sde: SDE,
    data_shape: Sequence[int], 
    x: Float[Array, "..."], 
    q: Optional[Float[Array, "..."]] = None, 
    a: Optional[Float[Array, "..."]] = None, 
    *,
    exact_log_prob: bool = False,
    n_eps: Optional[int] = 10,
    eps_noise_type: Literal["gaussian", "rademacher"] = "gaussian",
    solver: Optional[dfx.AbstractSolver] = None
) -> Tuple[Float[Array, "..."], Scalar]:
    """
        Computes the log-likelihood of data by solving an ordinary differential equation (ODE) 
        related to the reverse SDE paramterised with a score network. The function supports exact 
        and approximate log-likelihood computation using multiple noise realizations (`eps`).

        Parameters:
        -----------
        `key` : `PRNGKeyArray`
            A JAX random key used for generating noise (eps) during the log-likelihood approximation (if it is required).
        
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
        
        `exact_log_prob` : `bool`, default: `False`
            If `True`, computes the exact log-likelihood. Otherwise, uses an approximation with noise realizations.
        
        `n_eps` : `Optional[int]`, default: `10`
            The number of noise (`eps`) realizations used for approximating the log-likelihood when `exact_log_prob` is `False`. 
            Ignored if `exact_log_prob` is `True`. Noise realizations are for the Hutchinson trace estimator.
        
        `eps_noise_type` : `Literal["gaussian", "rademacher"]`, default: `"gaussian"`
            The type of noise to use for the `eps` realizations. Can be either "gaussian" or "rademacher".
        
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
        - If `exact_log_prob` is `False`, the log-likelihood is approximated by averaging over multiple noise 
          realizations (`eps`).
        - The `solver` parameter allows for flexible ODE solving, and by default, a suitable solver is chosen 
          (`diffrax.Tsit5()`) if not provided.
    """

    model = eqx.nn.inference_mode(model, True)

    ode = get_ode(model, sde)

    if not exact_log_prob:
        assert key is not None, (
            "Must provide key for approximate likelihood calculations."
        )

        if n_eps is not None:
            eps_shape = (n_eps,) + x.shape 
        else:
            eps_shape = x.shape

        if eps_noise_type == "gaussian":
            eps = jr.normal(key, eps_shape)
        if eps_noise_type == "rademacher":
            eps = jr.rademacher(key, eps_shape)

        _ode_term = log_prob_approx
    else:
        eps = None
        _ode_term = log_prob_exact

    _solver = solver if solver is not None else get_solver()

    # Likelihood from solving initial value problem
    sol = dfx.diffeqsolve(
        dfx.ODETerm(_ode_term),
        _solver, 
        t0=sde.t0,
        t1=sde.t1, 
        dt0=sde.dt, 
        y0=(x.flatten(), 0.), # Data and initial change in log_prob
        args=(eps, q, a, ode, data_shape),
        # adjoint=dfx.DirectAdjoint()
    ) 
    (z,), (delta_log_likelihood,) = sol.ys

    p_z = sde.prior_log_prob(z)
    log_p_x = p_z + delta_log_likelihood 

    return z, log_p_x


@jaxtyped(typechecker=typechecker)
def get_log_likelihood_fn(
    model: eqx.Module, 
    sde: SDE, 
    *,
    data_shape: Sequence[int], 
    exact_log_prob: bool = False,
    n_eps: Optional[int] = None,
    solver: Optional[dfx.AbstractSolver] = None
) -> Callable[
    [
        Float[Array, "..."], 
        Optional[Float[Array, "..."]], 
        Optional[Float[Array, "..."]], 
        Optional[PRNGKeyArray]
    ],
    Scalar
]:
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
        
        `exact_log_prob` : `bool`, default: `False`
            If `True`, the returned function will compute the exact log-likelihood. Otherwise, it will use 
            an approximation with multiple noise realizations.
        
        `n_eps` : `Optional[int]`, default: `None`
            The number of noise realizations (`eps`) to use when approximating the log-likelihood. Ignored if `exact_log_prob` is `True`.
        
        `solver` : `Optional[dfx.AbstractSolver]`, default: `None`
            The differential equation solver to use in the returned function for solving the ODE. If `None`, a default solver is used.

        Returns:
        --------
        `Callable`
            A parameterized function that computes the log-likelihood of input data `x`, conditioned on optional 
            parameters `q` and `a`, and using the provided random key `key`.

        The returned function has the following signature:

        ```python
        def _log_likelihood_fn(x: Array, q: Optional[Array], a: Optional[Array], key: PRNGKeyArray) -> Array:
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
            
            `key` : `PRNGKeyArray`
                A JAX random key for sampling noise during log-likelihood approximation.

            Returns:
            --------
            `log_probs` : `Array`
                The computed log-likelihood of the input data `x`.
            '''
        ```
    """

    @jaxtyped(typechecker=typechecker)
    def _log_likelihood_fn(
        x: Float[Array, "..."], 
        q: Optional[Float[Array, "..."]] = None, 
        a: Optional[Float[Array, "..."]] = None, 
        key: Optional[PRNGKeyArray] = None
    ) -> Scalar:
        """
            Computes the log-likelihood of input data `x` using a score-based model and stochastic 
            differential equation (SDE), optionally conditioned on `q` and `a`.

            This function is parameterized by a model and SDE and returns the log-likelihood of data 
            by solving an ODE. It supports both exact log-likelihood computation and approximation 
            via multiple noise realizations.

            Parameters:
            -----------
            `x` : `Array`
                The input data for which the log-likelihood is to be computed. This typically represents 
                a batch of data points or images.
            
            `q` : `Optional[Array]`, default: `None`
                Optional conditioning variable `q`, used to condition the log-likelihood calculation if applicable.
            
            `a` : `Optional[Array]`, default: `None`
                Optional conditioning variable `a`, used to condition the log-likelihood calculation if applicable.
            
            `key` : `Optional[PRNGKeyArray]`, default: `None`
                A JAX random key used for generating noise realizations in the case of approximate log-likelihood 
                computations. If the exact log-likelihood is being computed (`exact_log_prob=True` in `get_log_likelihood_fn`), 
                this argument may be ignored.

            Returns:
            --------
            `log_probs` : `Array`
                The computed log-likelihood of the input data `x`. This will be a scalar value for each data point, 
                estimating the probability density of the data under the given score-based model.

            Notes:
            ------
            - If `exact_log_prob` was set to `False` when creating this function, it uses Hutchinson trace approximation 
              with `n_eps` noise realizations to approximate the log-likelihood.
            - The function relies on solving an initial value problem for the reverse-time ODE associated with the SDE.
            - The log-likelihood can be conditioned on additional variables `q` and `a`, which are passed into the SDE.
            
            Example:
            --------
            ```python
            # Get a log-likelihood function for a trained model and SDE
            log_likelihood_fn = get_log_likelihood_fn(model, sde, data_shape=(3, 32, 32))

            # Use the function to compute log-likelihoods for a batch of data `x`
            log_probs = log_likelihood_fn(x=batch_data, q=None, a=None, key=jax.random.PRNGKey(42))
            ```
        """
        _, log_probs = log_likelihood(
            key, 
            model, 
            sde, 
            data_shape, 
            x, 
            q, 
            a, 
            exact_log_prob=exact_log_prob, 
            n_eps=n_eps, 
            solver=solver
        )
        return log_probs
    return _log_likelihood_fn