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


class SubVPSDE(SDE):
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
            Construct the sub-VP SDE that excels at likelihoods.

            Args:
        """
        super().__init__(dt=dt, t0=t0, t1=t1)
        self.beta_integral_fn = beta_integral_fn
        self.beta_fn = get_beta_fn(beta_integral_fn)
        self.weight_fn = weight_fn

    def sde(self, x: Array, t: Array) -> Tuple[Array, Array]:
        """ 
            dx = f(x, t) * dt + g(t) * dw 
            dx = -0.5 * beta(t) * x * dt + sqrt(beta(t) * (1 - exp(-2 * int[beta(s)]))) * dw
        """
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x
        # NOTE: Bug in Song++ code (should be -2. * BETA INTEGRAL)
        diffusion = jnp.sqrt(beta_t * -jnp.expm1(-2. * self.beta_integral_fn(t))) 
        return drift, diffusion

    def marginal_prob(self, x: Array, t: Array) -> Tuple[Array, Array]:
        """ 
            Sub-VP SDE p_t(x(t)|x(0)) is 
                x(t) ~ G[x(t)|mu(x(0), t), sigma^2(t)]
            where
                mu(x(0), t) = x(0) * exp(-0.5 * int[beta(s)])
                sigma^2(t) = [1 - exp(-int[beta(s)])]^2  
        """
        beta_integral = self.beta_integral_fn(t)
        mean = jnp.exp(-0.5 * beta_integral) * x 
        std = jnp.sqrt(jnp.square(-jnp.expm1(-2. * beta_integral))) # Eq 29 https://arxiv.org/pdf/2011.13456.pdf
        return mean, std

    def weight(self, t: Array, likelihood_weight: bool = False) -> Array:
        # Likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = self.beta_fn(t) * -jnp.expm1(-2. * self.beta_integral_fn(t))
            else:
                weight = jnp.square(1. - jnp.exp(-self.beta_integral_fn(t)))
        return weight 

    def prior_sample(self, key: Key, shape: Sequence[int]) -> Array:
        return jr.normal(key, shape)

    def prior_log_prob(self, z: Array) -> Array:
        return _get_log_prob_fn(scale=1.)(z)