import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx

from sbgm.sde import VPSDE
from sbgm._sample import get_eu_sample_fn, get_ode_sample_fn


def test_sampling():

    import diffrax
    print(jax.__version__, eqx.__version__, diffrax.__version__)
    
    key = jr.key(0)

    sde = VPSDE(beta_integral_fn=lambda t: t)

    n_sample = 10

    data_shape = (1, 32, 32)

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

    a_dim = None
    q_dim = None

    _in_size = 1024 + 1
    if q_dim is not None:
        _in_size += q_dim 
    if a_dim is not None:
        _in_size += a_dim 

    model = MLP(
        _in_size,
        out_size=1024, 
        width_size=1024, 
        depth=1,
        activation=jax.nn.tanh,
        key=key
    )

    key_samples = jr.split(key, n_sample)

    eu_sample_fn = get_eu_sample_fn(model, sde, data_shape)
    eu_samples = jax.vmap(eu_sample_fn)(key_samples)

    assert eu_samples.shape == (n_sample,) + data_shape

    ode_sample_fn = get_ode_sample_fn(model, sde, data_shape)
    ode_samples = jax.vmap(ode_sample_fn)(key_samples)

    assert ode_samples.shape == (n_sample,) + data_shape

    a_dim = 5
    q_dim = 10

    _in_size = 1024 + 1
    if q_dim is not None:
        _in_size += q_dim 
    if a_dim is not None:
        _in_size += a_dim 

    model = MLP(
        _in_size,
        out_size=1024, 
        width_size=1024, 
        depth=1,
        activation=jax.nn.tanh,
        key=key
    )

    key_samples = jr.split(key, n_sample)

    Q = jnp.ones((n_sample,) + (q_dim,))
    A = jnp.ones((n_sample,) + (a_dim,))

    eu_sample_fn = get_eu_sample_fn(model, sde, data_shape)
    eu_samples = jax.vmap(eu_sample_fn)(key_samples, Q, A)

    assert eu_samples.shape == (n_sample,) + data_shape

    ode_sample_fn = get_ode_sample_fn(model, sde, data_shape)
    ode_samples = jax.vmap(ode_sample_fn)(key_samples, Q, A)

    assert ode_samples.shape == (n_sample,) + data_shape