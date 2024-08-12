from typing import Sequence, Tuple, Self, Callable, Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class SDE(eqx.Module):
    """
        SDE abstract class.
    """
    dt: float
    t0: float
    t1: float
    N: int

    def __init__(self, dt: float = 0.01, t0: float = 0., t1: float = 1., N: int = 1000):
        """
            Construct an SDE.
        """
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.N = N

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

    def discretize(self, x: Array, t: Union[float, Array]) -> Tuple[Array, Array]:
        """
            Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

            Useful for reverse diffusion sampling and probabiliy flow sampling.
            Defaults to Euler-Maruyama discretization.

            Args:
            x: a jax array
            t: a jax float representing the time step (from 0 to `self.T`)

            Returns:
            f, G
        """
        drift, diffusion = self.sde(x, t)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(jnp.atleast_1d(self.dt))
        return f, G

    def reverse(self, score_fn: eqx.Module, probability_flow: bool = False) -> Self:
        """
            Create the reverse-time SDE/ODE.

            Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        sde_fn = self.sde
        discretize_fn = self.discretize

        if isinstance(self, (VPSDE, SubVPSDE)):
            _sde_fn = self.beta_integral_fn 
        if isinstance(self, VESDE):
            _sde_fn = self.sigma_fn

        _dt = self.dt
        _t0 = self.t0
        _t1 = self.t1
        _N = self.N

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__, SDE):
            probability_flow: bool

            def __init__(self):
                self.N = _N
                self.probability_flow = probability_flow
                super().__init__(
                    _sde_fn, dt=_dt, t0=_t0, t1=_t1, N=_N # _beta_integral_fn
                ) # **vars(self.__class__)

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

            def discretize(
                self, 
                x: Array, 
                t: Union[float, Array],
                q: Optional[Array] = None,
                a: Optional[Array] = None
            ) -> Tuple[Array, Array]:
                """ Create discretized iteration rules for the reverse diffusion sampler. """
                f, G = discretize_fn(x, t)
                rev_f = f - G ** 2. * score_fn(t, x, q, a) * (0.5 if self.probability_flow else 1.)
                rev_G = jnp.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


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


def _get_log_prob_fn(scale: float = 1.) -> Callable:
    def _log_prob_fn(z: Array) -> Array:
        return jax.scipy.stats.norm.logpdf(z, loc=0., scale=scale).sum()
    return _log_prob_fn


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
        t1: float = 1.,
        N: int = 1000
    ):
        """
            Construct a Variance Preserving SDE.

            Args:
            beta_min: value of beta(t=0)
            beta_max: value of beta(t=1)
            N: number of discretization steps
        """
        super().__init__(dt=dt, t0=t0, t1=t1, N=N)
        self.beta_integral_fn = beta_integral_fn 
        self.beta_fn = get_beta_fn(beta_integral_fn)
        self.weight_fn = weight_fn

    def sde(self, x, t):
        """ 
            dx = f(x, t) * dt + g(t) * dw 
            dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
        """
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x 
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
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

    def weight(self, t, likelihood_weight=False):
        # likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = self.beta_fn(t) # beta(t)
            else:
                weight = -jnp.expm1(-self.beta_integral_fn(t))
        return weight

    def prior_sample(self, key, shape):
        return jr.normal(key, shape)

    def prior_log_prob(self, z):
        return jax.vmap(_get_log_prob_fn(scale=1.))(z)

    def discretize(self, x, t):
        """ DDPM discretization. """
        timestep = t * (self.N - 1) / self.t1
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = jnp.sqrt(beta)
        f = jnp.sqrt(alpha) * x - x
        G = sqrt_beta
        return f, G


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
        t1: float = 1., 
        N: int = 1000
    ):
        """
            Construct the sub-VP SDE that excels at likelihoods.

            Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(dt=dt, t0=t0, t1=t1, N=N)
        self.beta_integral_fn = beta_integral_fn
        self.beta_fn = get_beta_fn(beta_integral_fn)
        self.weight_fn = weight_fn

    def sde(self, x, t):
        """ 
            dx = f(x, t) * dt + g(t) * dw 
            dx = -0.5 * beta(t) * x * dt + sqrt(beta(t) * (1 - exp(-2 * int[beta(s)]))) * dw
        """
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x
        # NOTE: Bug in Song++ code (should be -2. * BETA INTEGRAL)
        diffusion = jnp.sqrt(beta_t * -jnp.expm1(-2. * self.beta_integral_fn(t))) 
        return drift, diffusion

    def marginal_prob(self, x, t):
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

    def weight(self, t, likelihood_weight=False):
        # Likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = self.beta_fn(t) * -jnp.expm1(-2. * self.beta_integral_fn(t))
            else:
                weight = jnp.square(1. - jnp.exp(-self.beta_integral_fn(t)))
        return weight 

    def prior_sample(self, key, shape):
        return jr.normal(key, shape)

    def prior_log_prob(self, z):
        return jax.vmap(_get_log_prob_fn(scale=1.))(z)


def get_diffusion_fn(sigma_fn):
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
        t1: float = 1.,
        N: int = 1000
    ):
        """
            Construct a Variance Exploding SDE.

            dx = sqrt(d[sigma_fn(t) ** 2]/dt)

            Args:
            sigma: default variance value
            dt: timestep width
        """
        super().__init__(dt=dt, t0=t0, t1=t1, N=N)
        self.sigma_fn = sigma_fn
        self.weight_fn = weight_fn

    def sde(self, x, t):
        drift = jnp.zeros_like(x)
        _, dsigma2dt = jax.jvp(
            lambda t: jnp.square(self.sigma_fn(t)), 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        diffusion = jnp.sqrt(dsigma2dt)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """ 
            SDE:
                dx = sqrt(d[sigma^2(t)]/dt) * dw
            sigma(t) = exp(t) for example
                x(t) ~ G[x(t)|x(0), [sigma^2(t) - sigma^2(0)] * I 

            x(t) ~ G[x(t)|x(0), [sigma^2(t) - sigma^2(0)] * I
        """
        std = jnp.sqrt(jnp.square(self.sigma_fn(t)) - jnp.square(self.sigma_fn(0.))) 
        return x, std 

    def weight(self, t, likelihood_weight=False):
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = jnp.square(self.sigma_fn(t))
            else:
                weight = jnp.square(self.sigma_fn(t)) # Same for likelihood weighting
        return weight

    def prior_sample(self, key, shape):
        return jr.normal(key, shape) * self.sigma_fn(self.t1) 

    def prior_log_prob(self, z):
        return jax.vmap(_get_log_prob_fn(scale=self.sigma_fn(self.t1)))(z)

    def discretize(self, x, t):
        """ SMLD(NCSN) discretization. """
        timestep = (t * (self.N - 1) / self.T)
        sigma = self.discrete_sigmas[timestep]
        adjacent_sigma = jnp.where(
            timestep == 0., 
            jnp.zeros_like(t), 
            self.discrete_sigmas[timestep - 1]
        )
        f = jnp.zeros_like(x)
        G = jnp.sqrt(sigma ** 2. - adjacent_sigma ** 2.)
        return f, G


if __name__ == "__main__":
    import os 
    import matplotlib.pyplot as plt 
    import numpy as np

    figs_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/sgm/figs/"

    # Plot SDEs with time
    beta_integral_fn = lambda t: t
    beta_fn = get_beta_fn(beta_integral_fn)
    sigma_fn = lambda t: jnp.exp(t)

    times = dict(t0=0., t1=4., dt=0.1)

    vp_sde = VPSDE(beta_integral_fn, **times)
    ve_sde = VESDE(sigma_fn=sigma_fn)
    subvp_sde = SubVPSDE(beta_integral_fn, **times)

    x = jnp.ones((1,))
    T = jnp.linspace(1e-5, times["t1"], 1000)

    def get_sde_drift_and_diffusion_fn(sde):
        return jax.vmap(sde.sde, in_axes=(None, 0))

    def get_sde_mean_and_std(sde):
        return jax.vmap(sde.marginal_prob, in_axes=(None, 0))

    fig, axs = plt.subplots(1, 4, figsize=(21., 4.), dpi=200)
    ax = axs[0]
    ax.plot(T, jax.vmap(beta_fn)(T), linestyle=":", label="beta")
    ax_ = ax.twinx()
    ax.legend(frameon=False, loc="upper left")
    ax_.plot(T, jax.vmap(beta_integral_fn)(T), label="integral")
    ax_.legend(frameon=False, loc="lower right")
    plt.title("SDEs")
    for ax, _sde in zip(axs[1:], [ve_sde, vp_sde, subvp_sde]):
        mu, std = get_sde_mean_and_std(_sde)(x, T)
        ax.set_title(str(_sde.__class__.__name__))
        ax.plot(T, mu, label=r"$\mu(t)$")
        ax.plot(T, std, label=r"$\sigma(t)$")
        ax.legend(frameon=False)
    plt.savefig(os.path.join(figs_dir, "sdes.png"), bbox_inches="tight")
    plt.close()