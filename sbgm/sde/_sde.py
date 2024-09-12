from typing import Sequence, Tuple, Self, Callable, Optional, Union
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array


class SDE(eqx.Module):
    """
        SDE abstract class.
    """
    dt: float
    t0: float
    t1: float

    def __init__(self, dt: float = 0.01, t0: float = 0., t1: float = 1.):
        """
            Construct an SDE.
        """
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.dt = dt

    def sde(self, x: Array, t: Union[float, Array]) -> Tuple[Array, Array]:
        pass

    def marginal_prob(self, x: Array, t: Union[float, Array]) -> Tuple[Array, Array]:
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$. """
        pass

    def prior_sample(self, key: Key, shape: Sequence[int]) -> Array:
        """ Generate one sample from the prior distribution, $p_T(x)$. """
        pass

    def weight(self, t: Union[float, Array], likelihood_weight: bool = False) -> Array:
        """ Weighting for loss """
        pass

    def prior_log_prob(self, z: Array) -> Array:
        """
            Compute log-density of the prior distribution.

            Useful for computing the log-likelihood via probability flow ODE.

            Args:
            z: latent code
            Returns:
            log probability density
        """
        pass

    def reverse(self, score_fn: eqx.Module, probability_flow: bool = False) -> Self:
        """
            Create the reverse-time SDE/ODE.

            Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        sde_fn = self.sde

        if hasattr(self, "beta_integral_fn"):
            _sde_fn = self.beta_integral_fn 
        if hasattr(self, "sigma_fn"):
            _sde_fn = self.sigma_fn

        _dt = self.dt
        _t0 = self.t0
        _t1 = self.t1

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__, SDE):
            probability_flow: bool

            def __init__(self):
                self.probability_flow = probability_flow
                super().__init__(_sde_fn, dt=_dt, t0=_t0, t1=_t1)

            def sde(
                self, 
                x: Array, 
                t: Union[float, Array], 
                q: Optional[Array] = None,
                a: Optional[Array] = None
            ) -> Tuple[Array, Array]:
                """ 
                    Create the drift and diffusion functions for the reverse SDE/ODE. 
                    - forward time SDE:
                        dx = f(x, t) * dt + g(t) * dw
                    - reverse time SDE:
                        dx = [f(x, t) - g^2(t) * score(x, t)] * dt + g(t) * dw
                    - ode of SDE:
                        dx = [f(x, t) - 0.5 * g^2(t) * score(x, t)] * dt (ODE => No dw)
                """
                coeff = 0.5 if self.probability_flow else 1.
                drift, diffusion = sde_fn(x, t)
                score = score_fn(t, x, q, a)
                # Drift coefficient of reverse SDE and probability flow only different by a factor
                drift = drift - jnp.square(diffusion) * score * coeff
                # Set the diffusion function to zero for ODEs (dw=0)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

        return RSDE()


def _get_log_prob_fn(scale: float = 1.) -> Callable:
    def _log_prob_fn(z: Array) -> Array:
        return jax.scipy.stats.norm.logpdf(z, loc=0., scale=scale).sum()
    return _log_prob_fn


# if __name__ == "__main__":
#     import os 
#     import matplotlib.pyplot as plt 
#     import numpy as np

#     figs_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/sgm/figs/"

#     # Plot SDEs with time
#     beta_integral_fn = lambda t: t
#     beta_fn = get_beta_fn(beta_integral_fn)
#     sigma_fn = lambda t: jnp.exp(t)

#     times = dict(t0=0., t1=4., dt=0.1)

#     vp_sde = VPSDE(beta_integral_fn, **times)
#     ve_sde = VESDE(sigma_fn=sigma_fn)
#     subvp_sde = SubVPSDE(beta_integral_fn, **times)

#     x = jnp.ones((1,))
#     T = jnp.linspace(1e-5, times["t1"], 1000)

#     def get_sde_drift_and_diffusion_fn(sde):
#         return jax.vmap(sde.sde, in_axes=(None, 0))

#     def get_sde_mean_and_std(sde):
#         return jax.vmap(sde.marginal_prob, in_axes=(None, 0))

#     fig, axs = plt.subplots(1, 4, figsize=(21., 4.), dpi=200)
#     ax = axs[0]
#     ax.plot(T, jax.vmap(beta_fn)(T), linestyle=":", label=r"$\beta(t)$")
#     ax_ = ax.twinx()
#     ax.legend(frameon=False, loc="upper left")
#     ax_.plot(T, jax.vmap(beta_integral_fn)(T), label=r"$\int_0^t\beta(s)ds$")
#     ax_.legend(frameon=False, loc="lower right")
#     plt.title("SDEs")
#     for ax, _sde in zip(axs[1:], [ve_sde, vp_sde, subvp_sde]):
#         mu, std = get_sde_mean_and_std(_sde)(x, T)
#         ax.set_title(str(_sde.__class__.__name__))
#         ax.plot(T, mu, label=r"$\mu(t)$")
#         ax.plot(T, std, label=r"$\sigma(t)$")
#         ax.legend(frameon=False)
#     plt.savefig(os.path.join(figs_dir, "sdes.png"), bbox_inches="tight")
#     plt.close()