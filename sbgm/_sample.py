from typing import Sequence, Callable, Optional
import diffrax as dfx  
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker

from .sde import SDE
from ._ode import get_solver

"""
    Implements sampling from the reverse-time stochastic differential equation (SDE) using either an ODE solver or
    Euler-Maruyama method. 
    - This module provides functions to generate samples from a trained score-based generative model
      by solving the reverse-time SDE defined by the model.
"""


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def single_ode_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    key: Key[jnp.ndarray, ""],
    q: Optional[Float[Array, "..."]] = None,
    a: Optional[Float[Array, "..."]] = None,
    solver: Optional[dfx.AbstractSolver] = None
) -> Array:
    """
        Solves the reverse-time ordinary differential equation (ODE) to generate a sample from the prior distribution 
        by solving the initial value problem (IVP) with the provided model and stochastic differential equation (SDE).

        Parameters:
        -----------
        `model` : `eqx.Module`
            The trained model, typically a score-based generative model, used for reverse-time sampling.
        
        `sde` : `SDE`
            The stochastic differential equation (SDE) defining both the forward and reverse diffusion dynamics.
        
        `data_shape` : `Sequence[int]`
            Shape of the data to be sampled, used for constructing the prior sample.
        
        `key` : `Key`
            A JAX random key used to generate the prior sample.
        
        `q` : `Optional[Array]`, default: `None`
            Optional conditioning variable `q` for the ODE, if applicable.
        
        `a` : `Optional[Array]`, default: `None`
            Optional conditioning variable `a` for the ODE, if applicable.
        
        `solver` : `Optional[dfx.AbstractSolver]`, default: `None`
            The differential equation solver to be used for solving the reverse-time ODE. If `None`, a default solver is used.

        Returns:
        --------
        `Array`
            The generated sample at time `t0`, obtained by solving the reverse-time ODE from the prior sample at `t1`.

        Notes:
        ------
        - The function uses the reverse SDE (with probability flow) to define the drift term for the ODE, 
        which corresponds to the reverse diffusion process.
        - The ODE is solved from `t1` (the endpoint of the forward process) to `t0` (the initial point).
        - If `q` and `a` are provided, they condition the reverse-time dynamics during the sampling process.
        - The reverse-time SDE is solved by `diffeqsolve` from the `diffrax` library, with `ODETerm` representing 
        the reverse dynamics.

        Example:
        --------
        ```python
        sampled_data = single_ode_sample_fn(
            model, sde, data_shape=(3, 32, 32), key=jr.PRNGKey(0)
        )
        ```
    """

    model = eqx.nn.inference_mode(model, True)

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


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def single_eu_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int], 
    key: Key[jnp.ndarray, ""], 
    q: Optional[Float[Array, "..."]] = None,
    a: Optional[Float[Array, "..."]] = None,
    T_sample: int = 1_000
) -> Array:
    """
        Implements the Euler-Maruyama sampler for solving the reverse-time stochastic differential equation (SDE) 
        to generate a sample from the prior distribution.

        Parameters:
        -----------
        `model` : `eqx.Module`
            The trained model, typically a score-based generative model, used for reverse-time sampling.
        
        `sde` : `SDE`
            The stochastic differential equation (SDE) defining both the forward and reverse diffusion dynamics.
        
        `data_shape` : `Sequence[int]`
            Shape of the data to be sampled, used for constructing the prior sample.
        
        `key` : `Key`
            A JAX random key used to generate the prior sample and noise at each time step.
        
        `q` : `Optional[Array]`, default: `None`
            Optional conditioning variable `q` for the SDE, if applicable.
        
        `a` : `Optional[Array]`, default: `None`
            Optional conditioning variable `a` for the SDE, if applicable.
        
        `T_sample` : `int`, default: `1_000`
            The number of time steps used for the Euler-Maruyama discretization. Higher values give more accurate 
            approximations of the reverse SDE.

        Returns:
        --------
        `Array`
            The generated sample at time `t0`, obtained by solving the reverse-time SDE using Euler-Maruyama discretization.

        Notes:
        ------
        - The function simulates the reverse diffusion process starting from a prior sample `xT` at time `t1`, 
          and evolves it to `t0` using the Euler-Maruyama method.
        - At each time step `i`, the function applies the Euler-Maruyama update rule: 
          `x <- x + [f(x, t) - g^2(t) * score(x, t, q)] * dt + g(t) * sqrt(dt) * eps_t`, 
          where `f(x, t)` is the drift term, `g(t)` is the diffusion coefficient, and `eps_t` is a random noise term.
        - The reverse-time SDE is defined with `probability_flow=False`, meaning it includes both drift and diffusion components.

        Example:
        --------
        ```python
        sampled_data = single_eu_sample_fn(
            model, sde, data_shape=(32, 32, 3), key=jr.PRNGKey(0)
        )
        ```
    """
    
    model = eqx.nn.inference_mode(model, True)

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
    """
        Returns a callable function that implements Euler-Maruyama sampling of the reverse-time stochastic differential 
        equation (SDE) for generating data samples.

        Parameters:
        -----------
        `model` : `eqx.Module`
            The trained score network model used for reverse-time sampling.
        
        `sde` : `SDE`
            The stochastic differential equation (SDE) defining both the forward diffusion dynamics.
        
        `data_shape` : `Sequence[int]`
            Shape of the data to be sampled, used for constructing the prior sample.
        
        `T_sample` : `int`, default: `1_000`
            The number of time steps used for the Euler-Maruyama discretization. Higher values give more accurate 
            approximations of the reverse SDE.

        Returns:
        --------
        `Callable`
            A function `_eu_sample_fn` which generates a data sample given a random key `key`, and optional 
            conditioning variables `q` and `a`. It has the following signature:
            
            ```python
            def _eu_sample_fn(key: Key, q: Optional[Array], a: Optional[Array]) -> Array
            ```

            This function uses `single_eu_sample_fn` to generate a sample at time `t0`.

        Example:
        --------
        ```python
        eu_sampler = get_eu_sample_fn(model, sde, data_shape=(3, 32, 32))
        sampled_data = eu_sampler(jr.PRNGKey(0), q=None, a=None)
        ```

        Notes:
        ------
        - The returned function calls `single_eu_sample_fn` to simulate the reverse diffusion process starting from 
          a prior sample `xT` at time `t1`, and evolves it to `t0` using the Euler-Maruyama method.
        - The sample function takes in a `key` for random sampling, and optionally `q` and `a` as conditioning variables, 
          which can be used to condition the SDE dynamics.
        - The number of discretization steps `T_sample` controls the fidelity of the sample, with higher values providing 
          more accurate results.
    """
    def _eu_sample_fn(
        key: Key, 
        q: Optional[Float[Array, "..."]] = None, 
        a: Optional[Float[Array, "..."]] = None, 
    ) -> Array: 
        return single_eu_sample_fn(
            model, sde, data_shape, key, q, a, T_sample
        ) 
    return _eu_sample_fn


def get_ode_sample_fn(
    model: eqx.Module, 
    sde: SDE, 
    data_shape: Sequence[int],
    solver: Optional[dfx.AbstractSolver] = None
) -> Callable:
    """
        Returns a callable function that implements sampling from the reverse-time stochastic differential equation 
        (SDE) using the ODE solver approach.

        Parameters:
        -----------
        `model` : `eqx.Module`
            The trained score network model used for reverse-time sampling.

        `sde` : `SDE`
            The stochastic differential equation (SDE) defining both the forward diffusion dynamics.

        `data_shape` : `Sequence[int]`
            Shape of the data to be sampled, used for constructing the prior sample.

        Returns:
        --------
        `Callable`
            A function `_ode_sample_fn` which generates a data sample given a random key `key`, and optional 
            conditioning variables `q` and `a`. It has the following signature:
            
            ```python
            def _ode_sample_fn(key: Key, q: Optional[Array], a: Optional[Array]) -> Array
            ```

            This function uses `single_ode_sample_fn` to generate a sample at time `t0`.

        Example:
        --------
        ```python
        ode_sampler = get_ode_sample_fn(model, sde, data_shape=(3, 32, 32))
        sampled_data = ode_sampler(jr.PRNGKey(0))
        ```

        Notes:
        ------
        - The returned function calls `single_ode_sample_fn` to simulate the reverse diffusion process starting from 
          a prior sample at time `t1`, and evolves it to `t0` using the ODE method.
        - The sample function takes in a `key` for random sampling, and optionally `q` and `a` as conditioning variables, 
          which can be used to condition the SDE dynamics.
    """
    def _ode_sample_fn(
        key: Key, 
        q: Optional[Float[Array, "..."]] = None, 
        a: Optional[Float[Array, "..."]] = None, 
    ) -> Array:
        return single_ode_sample_fn(
            model, sde, data_shape, key, q, a, solver
        )
    return _ode_sample_fn