import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx


@eqx.filter_jit
def observed_information(key, q, sample_fn, log_likelihood_fn, n_samples):
    # Get samples from model, at 'q' in parameter space
    key_sample, key_J = jr.split(key)
    keys = jr.split(key_sample, n_samples)
    # sample_fn = get_sample_fn(model, int_beta, data_shape, dt0, t1)
    x = jax.vmap(sample_fn, in_axes=(0, None))(keys, q)
    # Calculate observed Fisher information under model (args ordered for argnum=0)
    keys = jr.split(key_J, n_samples)
    # L_x_q = jax.vmap(log_likelihood_fn, in_axes=(None, 0, 0))(q, x, keys)
    Js = jax.vmap(jax.hessian(log_likelihood_fn), in_axes=(None, 0, 0))(q, x, keys)
    F = Js.mean(axis=0)
    return F


def get_observed_information_fn(sample_fn, log_likelihood_fn, n_samples):
    return eqx.filter_jit(
        lambda key, q: observed_information(
            key, q, sample_fn, log_likelihood_fn, n_samples
        )
    )


def logdet(F):
    a, b = jnp.linalg.slogdet(F)
    return a * b