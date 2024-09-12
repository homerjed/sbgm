from typing import Callable, Optional, Union, Sequence, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

from ._sde import SDE, _get_log_prob_fn


def get_beta_fn(beta_integral_fn: Union[Callable, eqx.Module]) -> Callable:
    """ Obtain beta function from a beta beta integral. """
    def _beta_fn(t):
        _, beta = jax.jvp(
            beta_integral_fn, 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        return beta
    return _beta_fn


class VPSDE(SDE):
    beta_integral_fn: Callable
    beta_fn: Callable
    weight_fn: Callable

    def __init__(
        self, 
        beta_integral_fn: Callable, 
        weight_fn: Optional[Callable] = None, 
        dt: float = 0.1, 
        t0: float = 0., 
        t1: float = 1.
    ):
        """
            Construct a Variance Preserving SDE.

            Args:
        """
        super().__init__(dt=dt, t0=t0, t1=t1)
        self.beta_integral_fn = beta_integral_fn 
        self.beta_fn = get_beta_fn(beta_integral_fn)
        self.weight_fn = weight_fn

    def sde(self, x: Array, t: Array) -> Tuple[Array, Array]:
        """ 
            dx = f(x, t) * dt + g(t) * dw 
            dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
        """
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x 
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x: Array, t: Array) -> Tuple[Array, Array]:
        """ 
            VP SDE p_t(x(t)|x(0)) is
                x(t) ~ G[x(t)|mu(x(0), t), sigma^2(t)] 
            where
                mu(x(0), t) = x(0) * exp(-0.5 * int[beta(s)])
                sigma^2(t) = I * (1 - exp(-int[beta(s)]))

            -> return mean * x and std (not var?)
        """
        beta_integral = self.beta_integral_fn(t)
        mean = jnp.exp(-0.5 * beta_integral) * x 
        std = jnp.sqrt(-jnp.expm1(-beta_integral)) 
        return mean, std

    def weight(self, t: Array, likelihood_weight: bool = False) -> Array:
        # likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = self.beta_fn(t) # beta(t)
            else:
                weight = -jnp.expm1(-self.beta_integral_fn(t))
        return weight

    def prior_sample(self, key: Key, shape: Sequence[int]) -> Array:
        return jr.normal(key, shape)

    def prior_log_prob(self, z: Array) -> Array:
        return _get_log_prob_fn(scale=1.)(z)