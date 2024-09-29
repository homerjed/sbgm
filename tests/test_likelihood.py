import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx

from sbgm.sde import VPSDE
from sbgm._ode import get_log_likelihood_fn


def test_likelihood():

    key = jr.key(0)

    sde = VPSDE(beta_integral_fn=lambda t: t)

    data_shape = (1, 32, 32)
    a_dim = 5
    q_dim = 10

    class MLP(eqx.Module):
        net: eqx.nn.MLP

        def __init__(self, *args, **kwargs):
            self.net = eqx.nn.MLP(*args, **kwargs)

        def __call__(self, t, x, q=None, a=None, key=None):
            _inputs = [jnp.atleast_1d(t), x.flatten()]
            if a is not None:
                _inputs += [a]
            if q is not None:
                _inputs += [q.flatten()]

            out = self.net(jnp.concatenate(_inputs))
            out = out.reshape(data_shape)
            return out

    model = MLP(
        1024 + a_dim + q_dim + 1, 
        out_size=1024, 
        width_size=1024, 
        depth=1,
        activation=jax.nn.tanh,
        key=key
    )

    X = jnp.ones((10,) + data_shape)
    A = jnp.ones((10,) + (a_dim,))
    Q = jnp.ones((10,) + (q_dim,))

    log_likelihood_fn = get_log_likelihood_fn(
        model, sde, data_shape, exact_logp=True
    )
    L_X = jax.vmap(log_likelihood_fn)(X, A, Q)

    assert L_X.shape == (len(X),)

    model = MLP(
        1024 + 1, 
        out_size=1024, 
        width_size=1024, 
        depth=1,
        activation=jax.nn.tanh,
        key=key
    )

    X = jnp.ones((10,) + data_shape)

    log_likelihood_fn = get_log_likelihood_fn(
        model, sde, data_shape, exact_logp=True
    )
    L_X = jax.vmap(log_likelihood_fn)(X)

    assert L_X.shape == (len(X),)

    model = MLP(
        1024 + a_dim + q_dim + 1, 
        out_size=1024, 
        width_size=1024, 
        depth=1,
        activation=jax.nn.tanh,
        key=key
    )

    X = jnp.ones((10,) + data_shape)
    A = jnp.ones((10,) + (a_dim,))
    Q = jnp.ones((10,) + (q_dim,))

    keys = jr.split(key, len(X))

    log_likelihood_fn = get_log_likelihood_fn(
        model, sde, data_shape, exact_logp=False
    )
    L_X = jax.vmap(log_likelihood_fn)(X, A, Q, keys)

    assert L_X.shape == (len(X),)

    model = MLP(
        1024 + 1, 
        out_size=1024, 
        width_size=1024, 
        depth=1,
        activation=jax.nn.tanh,
        key=key
    )

    X = jnp.ones((10,) + data_shape)

    keys = jr.split(key, len(X))

    log_likelihood_fn = get_log_likelihood_fn(
        model, sde, data_shape, exact_logp=False
    )
    L_X = jax.vmap(log_likelihood_fn, in_axes=(0, None, None, 0))(X, None, None, keys)

    assert L_X.shape == (len(X),)