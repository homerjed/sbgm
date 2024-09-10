import jax.numpy as jnp
import jax.random as jr 
from sklearn.datasets import make_moons

from .utils import ScalerDataset, _InMemoryDataLoader


def key_to_seed(key):
    return int(jnp.asarray(jr.key_data(key)).sum())

def moons(key):
    key_train, key_valid = jr.split(key)
    data_shape = (2,)
    context_shape = None 
    parameter_dim = 1

    Xt, Yt = make_moons(
        40_000, noise=0.05, random_state=key_to_seed(key_train)
    )
    Xv, Yv = make_moons(
        40_000, noise=0.05, random_state=key_to_seed(key_valid)
    )

    min = Xt.min()
    max = Xt.max()
    mean = Xt.mean()
    std = Xt.std()

    # (We do need to handle normalisation ourselves though.)
    # train_data = (Xt - min) / (max - min)
    # valid_data = (Xv - min) / (max - min)
    train_data = (Xt - mean) / std
    valid_data = (Xv - mean) / std
    
    train_dataloader = _InMemoryDataLoader(
        X=jnp.asarray(train_data), 
        A=jnp.asarray(Yt)[:, jnp.newaxis], 
        key=key_train
    )
    valid_dataloader = _InMemoryDataLoader(
        X=jnp.asarray(valid_data), 
        A=jnp.asarray(Yv)[:, jnp.newaxis], 
        key=key_valid
    )

    class _Scaler:
        forward: callable 
        reverse: callable
        def __init__(self, a, b):
            # [0, 1] -> [-1, 1]
            self.forward = lambda x: 2. * (x - a) / (b - a) - 1.
            # [-1, 1] -> [0, 1]
            self.reverse = lambda y: (y + 1.) * 0.5 * (b - a) + a

    def label_fn(key, n):
        Q = None
        A = jr.choice(key, jnp.array([0., 1.]), (n,))[:, jnp.newaxis]
        return Q, A

    return ScalerDataset(
        name="moons",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        scaler=_Scaler(a=Xt.min(), b=Xt.max()),
        label_fn=label_fn
    )