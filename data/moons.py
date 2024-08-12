import jax.numpy as jnp
import jax.random as jr 
from sklearn.datasets import make_moons

from .utils import ScalerDataset, _InMemoryDataLoader


def moons(key):
    key_train, key_valid = jr.split(key)
    data_shape = (2,)
    context_shape = (1,)

    Xt, Yt = make_moons(
        5_000, noise=0.05, random_state=int(key_train.sum())
    )
    Xv, Yv = make_moons(
        5_000, noise=0.05, random_state=int(key_valid.sum())
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
        jnp.asarray(train_data), 
        jnp.asarray(Yt)[:, jnp.newaxis], 
        key=key_train
    )
    valid_dataloader = _InMemoryDataLoader(
        jnp.asarray(valid_data), 
        jnp.asarray(Yv)[:, jnp.newaxis], 
        key=key_valid
    )

    class _Scaler:
        forward: callable 
        reverse: callable
        def __init__(self):
            # [0, 1] -> [-1, 1]
            self.forward = lambda x: x
            # [-1, 1] -> [0, 1]
            self.reverse = lambda y: y

    return ScalerDataset(
        name="moons",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=_Scaler()
    )