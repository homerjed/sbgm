import os
import jax.random as jr 
import jax.numpy as jnp
from jaxtyping import Key
from torchvision import transforms, datasets

from .utils import Scaler, ScalerDataset, TorchDataLoader


def cifar10(path: str, key: Key) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    n_pix = 32 # Native resolution for CIFAR10 
    data_shape = (3, n_pix, n_pix)
    parameter_dim = 1

    scaler = Scaler(x_min=0., x_max=1.)

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(), # This magically [0,255] -> [0,1]??
            transforms.Lambda(scaler.forward) # [0,1] -> [-1,1]
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )
    train_dataset = datasets.CIFAR10(
        os.path.join(path, "datasets/cifar10/"),
        train=True, 
        download=True, 
        transform=train_transform
    )
    valid_dataset = datasets.CIFAR10(
        os.path.join(path, "datasets/cifar10/"),
        train=False, 
        download=True, 
        transform=valid_transform
    )

    train_dataloader = TorchDataLoader(
        train_dataset, data_shape, context_shape=None, parameter_dim=parameter_dim, key=key_train
    )
    valid_dataloader = TorchDataLoader(
        valid_dataset, data_shape, context_shape=None, parameter_dim=parameter_dim, key=key_valid
    )

    def label_fn(key, n):
        Q = None
        A = jr.choice(key, jnp.arange(10), (n,))[:, jnp.newaxis]
        return Q, A

    return ScalerDataset(
        name="cifar10",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        parameter_dim=parameter_dim,
        context_shape=None,
        scaler=scaler,
        label_fn=label_fn
    )