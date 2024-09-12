from typing import Callable, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

from ._sde import SDE, _get_log_prob_fn


def get_diffusion_fn(sigma_fn: Union[Callable, eqx.Module]) -> Callable:
    """ Get diffusion coefficient function for VE SDE: dx = sqrt(d[sigma^2(t)]/dt)dw """
    def _diffusion_fn(t):
        _, dsigmadt = jax.jvp(
            lambda t: jnp.square(sigma_fn(t)), 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        return jnp.sqrt(dsigmadt) # = sqrt(d[sigma^2(t)]/dt) ?
    return _diffusion_fn


class VESDE(SDE):
    sigma_fn: Callable
    weight_fn: Callable

    def __init__(
        self, 
        sigma_fn, 
        weight_fn: Optional[Callable] = None, 
        dt: float = 0.1, 
        t0: float = 0., 
        t1: float = 1.
    ):
        """
            Construct a Variance Exploding SDE.

            dx = sqrt(d[sigma_fn(t) ** 2]/dt)

            Args:
            sigma: default variance value
            dt: timestep width
        """
        super().__init__(dt=dt, t0=t0, t1=t1)
        self.sigma_fn = sigma_fn
        self.weight_fn = weight_fn

    def sde(self, x: Array, t: Array) -> Tuple[Array, Array]:
        drift = jnp.zeros_like(x)
        _, dsigma2dt = jax.jvp(
            lambda t: jnp.square(self.sigma_fn(t)), 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        diffusion = jnp.sqrt(dsigma2dt)
        return drift, diffusion

    def marginal_prob(self, x: Array, t: Array) -> Tuple[Array, Array]:
        """ 
            SDE:
                dx = sqrt(d[sigma^2(t)]/dt) * dw
            sigma(t) = exp(t) for example
                x(t) ~ G[x(t)|x(0), [sigma^2(t) - sigma^2(0)] * I 

            x(t) ~ G[x(t)|x(0), [sigma^2(t) - sigma^2(0)] * I
        """
        std = jnp.sqrt(jnp.square(self.sigma_fn(t)) - jnp.square(self.sigma_fn(0.))) 
        return x, std 

    def weight(self, t: Array, likelihood_weight: bool = False) -> Array:
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = jnp.square(self.sigma_fn(t))
            else:
                weight = jnp.square(self.sigma_fn(t)) # Same for likelihood weighting
        return weight

    def prior_sample(self, key: Key, shape: Sequence[int]) -> Array:
        return jr.normal(key, shape) * self.sigma_fn(self.t1) 

    def prior_log_prob(self, z: Array) -> Array:
        return _get_log_prob_fn(scale=self.sigma_fn(self.t1))(z)