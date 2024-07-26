from typing import Sequence, Tuple, Self, Callable, Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

Module = eqx.Module


def get_beta_fn(beta_integral: Union[Callable, Module]) -> Callable:
    """ Turn a beta integral into a beta function """
    def _beta_fn(t):
        _, beta = jax.jvp(
            beta_integral, 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        return beta
    return _beta_fn


def _get_log_prob_fn(scale: float = 1.) -> Callable:
    def _log_prob_fn(z: Array) -> Array:
        # NOTE: Summing over data-dims... prefer multivariate...
        return jax.scipy.stats.norm.logpdf(z, loc=0., scale=scale).sum()
    return _log_prob_fn


class SDE(eqx.Module):
    """
        SDE abstract class. Functions are designed for a mini-batch of inputs.
    """
    dt: float
    t0: float
    t1: float
    N: int

    def __init__(self, dt: float = 0.1, t0: float = 0., t1: float = 1., N: int = 1000):
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

    def reverse(self, score_fn: Module, probability_flow: bool = False) -> Self:
        """
            Create the reverse-time SDE/ODE.

            Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        # N = (self.t1 - self.t0) / self.dt
        sde_fn = self.sde
        discretize_fn = self.discretize
        _beta_integral = self.beta_integral
        _dt = self.dt
        _t0 = self.t0
        _t1 = self.t1
        _N = self.N

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__, SDE):
            N: int
            probability_flow: bool

            def __init__(self):
                self.N = _N
                self.probability_flow = probability_flow
                super().__init__(
                    _beta_integral, dt=_dt, t0=_t0, t1=_t1, N=_N
                ) # **vars(self.__class__)

            def sde(
                self, 
                x: Array, 
                t: Union[float, Array], 
                q: Optional[Array] = None
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
                score = score_fn(t, x, q)
                # Drift coefficient of reverse SDE and probability flow only different by a factor
                drift = drift - jnp.square(diffusion) * score * coeff
                # Set the diffusion function to zero for ODEs (dw=0)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(
                self, 
                x: Array, 
                t: Union[float, Array],
                q: Array
            ) -> Tuple[Array, Array]:
                """ Create discretized iteration rules for the reverse diffusion sampler. """
                f, G = discretize_fn(x, t)
                rev_f = f - G ** 2. * score_fn(t, x, q) * (0.5 if self.probability_flow else 1.)
                rev_G = jnp.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    beta_integral: Callable
    beta_fn: Callable

    def __init__(
        self, 
        beta_integral: Callable, 
        weight: Callable = None, 
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
        self.beta_integral = beta_integral
        self.beta_fn = get_beta_fn(beta_integral)
        # self.weight = get_weight_fn(weight)
        # N = int((t1 - t0) / dt)
        # Betas at each time point; not assuming linear beta_fn
        # self.discrete_betas = jax.vmap(self.beta_fn)(jnp.linspace(t0, t1, N))
        # DDPM parameters at each t
        # self.alphas = 1. - self.discrete_betas
        # self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        # self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        # self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

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
        log_mean_coeff = self.beta_integral(t)
        mean = jnp.exp(-0.5 * log_mean_coeff) * x 
        std = jnp.sqrt(-jnp.expm1(-log_mean_coeff)) 
        return mean, std

    def weight(self, t, likelihood_weight=False):
        # likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if likelihood_weight:
            weight = self.beta_fn(t) # beta(t)
        else:
            weight = -jnp.expm1(-self.beta_integral(t))
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
    def __init__(
        self, 
        beta_integral: Callable, 
        weight: Callable = None, 
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
        self.beta_integral = beta_integral
        self.beta_fn = get_beta_fn(beta_integral)
        # self.weight = weight

        # N = int((t1 - t0) / dt)
        # # Betas at each time point; not assuming linear beta_fn
        # self.discrete_betas = jax.vmap(self.beta_fn)(jnp.linspace(t0, t1, N))
        # # DDPM parameters
        # self.alphas = 1. - self.discrete_betas
        # self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        # self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        # self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

    def sde(self, x, t):
        """ 
            dx = f(x, t) * dt + g(t) * dw 
            dx = -0.5 * beta(t) * x * dt + sqrt(beta(t) * (1 - exp(-2 * int[beta(s)]))) * dw
        """
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x
        # Bug in SDE CODE? SHOULD BE -2. * BETA INTEGRAL?
        diffusion = jnp.sqrt(beta_t * (-jnp.expm1(-2. * self.beta_integral(t)))) # -jnp.exp1m instead?
        return drift, diffusion

    def marginal_prob(self, x, t):
        """ 
            Sub-VP SDE p_t(x(t)|x(0)) is 
                x(t) ~ G[x(t)|mu(x(0), t), sigma^2(t)]
            where
                mu(x(0), t) = x(0) * exp(-0.5 * int[beta(s)])
                sigma^2(t) = [1 - exp(-int[beta(s)])]^2  
        """
        log_mean_coeff = self.beta_integral(t)
        mean = jnp.exp(-0.5 * log_mean_coeff) * x 
        std = jnp.sqrt(jnp.square(-jnp.expm1(-2. * log_mean_coeff))) # Eq 29 https://arxiv.org/pdf/2011.13456.pdf
        return mean, std

    def weight(self, t, likelihood_weight=False):
        # likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if likelihood_weight:
            weight = beta_fn(t) * (-jnp.expm1(-2. * self.beta_integral(t)))
        else:
            weight = jnp.square(1. - jnp.exp(-self.beta_integral(t)))
        return weight 

    def prior_sample(self, key, shape):
        return jr.normal(key, shape)

    def prior_log_prob(self, z):
        return jax.vmap(_get_log_prob_fn(scale=1.))(z)


class VESDE2(SDE):
    def __init__(self, sigma, dt=0.1, t0=0., t1=1.):
        """
            Construct a Variance Exploding SDE.

            Args:
            sigma: default variance value
            dt: timestep width
        """
        super().__init__(dt=dt, t0=t0, t1=t1)
        self.N = int((t1 - t0) / dt)
        self.sigma = sigma
        self.sigma_fn = lambda t: self.sigma ** t
        self.discrete_sigmas = jax.vmap(self.sigma_fn)(jnp.linspace(t0, t1, self.N))
        self.t1 = t1

    def sde(self, x, t):
        drift = jnp.zeros_like(x)
        diffusion = self.sigma_fn(t) 
        return drift, diffusion

    def marginal_prob(self, x, t):
        """ x(t) ~ G[x(t)|x(0), [sigma^2(t) - sigma^2(0)] * I """
        # This assumes sigma ** 5 sigma_fn always..
        std = jnp.sqrt(
            # This is sqrt(d\dt[sigma ** t])?
            # d\dt[sigma ** (2 * t)] 
            # = log(sigma ** 2) * sigma ** (2 * t) 
            # = 2 * log(sigma) * sigma ** (2 * t) 
            (self.sigma ** (2. * t) - 1.) / (2. * jnp.log(self.sigma))
        )
        return x, std

    def weight(self, t, likelihood_weight=False):
        weight = self.sigma_fn(t) # Same for likelihood weighting
        return weight

    def prior_sample(self, key, shape):
        return jr.normal(key, shape) * self.sigma_fn(self.t1)

    def prior_log_prob(self, z):
        return jax.vmap(_get_log_prob_fn(scale=self.sigma_max))(z)

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


def get_diffusion_fn(sigma_fn):
    """ Get diffusion coefficient function for VE SDE """
    def _diffusion_fn(t):
        _, sigma = jax.jvp(
            sigma_fn, 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        return jnp.sqrt(sigma)
    return _diffusion_fn


class VESDE(SDE):
    def __init__(self, sigma_fn, dt=0.1, t0=0., t1=1.):
        """
            Construct a Variance Exploding SDE.

            Args:
            sigma: default variance value
            dt: timestep width
        """
        super().__init__(dt=dt, t0=t0, t1=t1)
        self.N = int((t1 - t0) / dt)
        self.sigma_fn = sigma_fn
        self.diffusion_fn = get_diffusion_fn(sigma_fn)
        self.discrete_sigmas = jax.vmap(self.sigma_fn)(jnp.linspace(t0, t1, self.N))
        self.t1 = t1

    def sde(self, x, t):
        """
            SDE:
                dx = sqrt(d[sigma^2(t)]/dt) * dw
        """
        drift = jnp.zeros_like(x)
        diffusion = self.diffusion_fn(t) #self.sigma_fn(t) 
        return drift, diffusion

    def marginal_prob(self, x, t):
        """ 
            SDE:
                dx = sqrt(d[sigma^2(t)]/dt) * dw
            sigma^t = exp(t) for example
                x(t) ~ G[x(t)|x(0), [sigma^2(t) - sigma^2(0)] * I 
        """
        # This assumes sigma ** 5 sigma_fn always..
        std = jnp.sqrt(self.sigma_fn(t) - self.sigma_fn(self.t0))
        return x, std

    def weight(self, t, likelihood_weight=False):
        weight = self.sigma_fn(t) # Same for likelihood weighting
        return weight

    def prior_sample(self, key, shape):
        return jr.normal(key, shape) * self.sigma_fn(self.t1)

    def prior_log_prob(self, z):
        return jax.vmap(_get_log_prob_fn(scale=self.sigma_max))(z)

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


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt 
#     import numpy as np
#     """
#         TODO: 
#         - replace init'ing beta_min, beta_max with t0, t1 and a beta_fn(t)
#         - beta_fn should actually depend on a beta_int time schedule, autodiff to get beta(t)

#         - replace all 1 - exp(...) with jnp.expm1(-(...))
#     """

#     # Test expand() function
#     x = jnp.ones((1, 32, 32))
#     y = expand(jnp.ones((1,)), x)
#     # assert x.shape == y.shape
#     print(y.shape)

#     # Plot SDEs with time
#     beta_integral = lambda t: t ** 2. + t
#     beta_fn = get_beta_fn(beta_integral)
#     # ve_sde = VESDE(beta_integral)
#     times = dict(t0=0., t1=20., dt=0.1)
#     vp_sde = VPSDE(beta_integral, **times)
#     ve_sde = VESDE(sigma=25.)
#     subvp_sde = SubVPSDE(beta_integral, **times)

#     x = jnp.ones((1,))
#     T = jnp.linspace(1e-5, 1., 100)

#     def get_sde_drift_and_diffusion_fn(sde):
#         return jax.vmap(sde.sde, in_axes=(None, 0))

#     def get_sde_mean_and_std(sde):
#         return jax.vmap(sde.marginal_prob, in_axes=(None, 0))

#     fig, axs = plt.subplots(1, 4, figsize=(21., 4.))
#     ax = axs[0]
#     ax.plot(T, jax.vmap(beta_fn)(T), linestyle=":", label="beta")
#     ax_ = ax.twinx()
#     ax.legend()
#     ax_.plot(T, jax.vmap(beta_integral)(T), label="integral")
#     ax_.legend()
#     plt.title("SDEs")
#     for ax, _sde in zip(axs[1:], [ve_sde, vp_sde, subvp_sde]):
#         mu, std = get_sde_mean_and_std(_sde)(x, T)
#         # print(mu)
#         # print(std)
#         # print("\n")
#         ax.set_title(str(_sde.__class__.__name__))
#         ax.plot(T, mu, label="$\mu(t)$")
#         ax.plot(T, std, label="$\sigma(t)$")
#     ax.legend()
#     plt.savefig("sdes.png")
#     plt.close()

#     if 0:
#         """ Reverse SDE has no marginal prob fn because it is the same in either direction? only the SDE reverses... """
#         fig, axs = plt.subplots(1, 3, figsize=(10., 4.))
#         plt.title("SDEs")
#         for ax, _sde in zip(axs, [ve_sde, vp_sde, subvp_sde]):
#             _rsde = _sde.reverse(lambda x: x)
#             mu, std = get_sde_mean_and_std(_rsde)(x, T)
#             ax.set_title(str(_sde.__class__.__name__))
#             ax.plot(T, mu, label="$\mu(t)$")
#             ax.plot(T, std, label="$\sigma(t)$")
#         ax.legend()
#         plt.savefig("sdes_reverse.png")
#         plt.close()

#     """
#         Make the classic fig...
#         - calculate p_t(x) for a scalar x=1 across all t
#     """
#     n_times = 1000
#     x = jnp.linspace(-2., 2., 1000)
#     times = jnp.linspace(0., 1., n_times)
#     X = np.zeros((n_times, len(x)))
#     sde = VPSDE(beta_integral=beta_integral)
#     # sde = VESDE(sigma=25.)

#     print("X", X.shape)
#     def gaussian_prob(mu, std):
#         return lambda x: jax.scipy.stats.norm.pdf(x, loc=mu, scale=std).squeeze()

#     if 0:
#         key = jr.PRNGKey(0)
#         points = np.zeros((n_times, 2))
#         p = sde.prior_sample(key, (1,))
#         p = jnp.array([100.])
#         for i, t in enumerate(times):
#             # Calculate the marginals, returns mu, std of p_t(x(t)) (background of plots)
#             mu, std = sde.marginal_prob(x=jnp.ones((1,)), t=t)
#             X[i] = jax.vmap(gaussian_prob(mu, std))(x)

#             # Stochastically evolve a point
#             mu, std = sde.marginal_prob(x=p, t=t) # mu = p * mu_t...
#             p = mu + jr.normal(jr.fold_in(key, i), p.shape) * std
#             points[i] = float(i), p.item()
#             print("\r", i, p, end="")

#             # Stochastically evolve by reverse SDE sampling?!
#             # eps_t = jr.normal(jr.fold_in(key, i), p.shape)
#             # drift, diffusion = sde.reverse(model, probability_flow=False).sde(p, times[i])
#             # mean_x = x + jnp.square(diffusion) * model(t, x, q) * step_size 
#             # x = drift + jnp.sqrt(step_size) * diffusion * eps_t

#             # print("\r", i, (jr.normal(jr.fold_in(key, i)) * std + mu).item(), end="")

#             # points[i, :] = float(i), float((jr.normal(jr.fold_in(key, i)) * std + x * mu).item())

#             # drift, diffusion = sde.sde(x=x, t=times[i])
#             # n = jr.normal(jr.fold_in(key, i), (1,))
#             # x = x * drift + n * diffusion

#         plt.figure(dpi=200)
#         plt.imshow(X.T, cmap="gist_stern", )#extent=[0., 1., -4., 4.])
#         plt.scatter(*points.T, color="royalblue", s=0.1)
#         # plt.plot(points[:, 1], color="w")
#         plt.ylabel("$p_t(x)$")
#         plt.xlabel("$t$")
#         plt.tight_layout()
#         plt.savefig("sdes_marginals.png")
#         plt.close()


#     import tensorflow_probability.substrates.jax as tfp
#     tfd = tfp.distributions
#     # Sample interesting data, to diffuse
#     a, b, c = 0.2, 0.3, 0.5
#     mixture = tfd.Mixture(
#         cat=tfd.Categorical(probs=[a, b, c]),
#         components=[
#             tfd.Normal(loc=-1., scale=0.1),
#             tfd.Normal(loc=+1., scale=0.5),
#             tfd.Normal(loc=0., scale=0.2),
#         ]
#     )

#     key = jr.PRNGKey(0)
#     d = mixture.sample((1000,), seed=key)
#     print(d.shape)

#     points = np.zeros((n_times, len(d)))
#     for i, t in enumerate(times):
#         # Calculate the marginals, returns mu, std of p_t(x(t)) (background of plots)
#         mu_, std_ = sde.marginal_prob(x=jnp.ones((1,)), t=t)

#         mu, std = sde.marginal_prob(x=d, t=t)
#         d = mu + jr.normal(jr.fold_in(key, i), d.shape) * std
#         # X[i] = gaussian_prob(mu_, std_)(d)
#         X[i] = jax.vmap(gaussian_prob(mu_, std_))(d)
#         # X[i] = d

#     plt.figure(dpi=200)
#     plt.imshow(X.T, cmap="gist_stern", )#extent=[0., 1., -4., 4.])
#     # plt.scatter(*points.T, color="royalblue", s=0.1)
#     # plt.plot(points[:, 1], color="w")
#     plt.ylabel("$p_t(x)$")
#     plt.xlabel("$t$")
#     plt.tight_layout()
#     plt.savefig("sdes_marginals.png")
#     plt.close()

#     # plt.plot(d)
#     # plt.savefig("sdes_marginals.png")
#     # plt.close()


