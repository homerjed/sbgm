import abc
from typing import Tuple, Union, NamedTuple, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
import numpy as np
import torch

Array = np.ndarray


def expand_if_scalar(x):
    return x[:, jnp.newaxis] if x.ndim == 1 else x


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(self, X, Q=None, A=None, *, key):
        self.X = X 
        self.Q = Q 
        self.A = A 
        self.key = key

    def loop(self, batch_size):
        dataset_size = self.X.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                # x, q, a
                yield (
                    self.X[batch_perm], 
                    self.Q[batch_perm] if self.Q is not None else None, 
                    self.A[batch_perm] if self.A is not None else None 
                )
                start = end
                end = start + batch_size


class _TorchDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        dataset, 
        data_shape,
        context_shape,
        parameter_dim,
        *, 
        num_workers=None, 
        key
    ):
        self.dataset = dataset
        self.context_shape = context_shape 
        self.parameter_dim = parameter_dim # Indices representing dataset having q, a or not
        min = torch.iinfo(torch.int32).min
        max = torch.iinfo(torch.int32).max
        self.seed = jr.randint(key, (), min, max).item() # key.sum().item() ?
        self.num_workers = num_workers

    def loop(self, batch_size, num_workers=2):
        generator = torch.Generator().manual_seed(self.seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=self.num_workers if self.num_workers is not None else num_workers,
            shuffle=True,
            drop_last=True,
            generator=generator
        )
        while True:
            for tensors in dataloader:

                x, *qa = tensors
                if self.context_shape and self.parameter_dim:
                    q, a = qa
                else:
                    if self.context_shape:
                        (q,) = qa
                    else:
                        q = None
                    if self.parameter_dim:
                        (a,) = qa
                    else:
                        a = None
                yield ( 
                    jnp.asarray(x),
                    expand_if_scalar(jnp.asarray(q)) if self.context_shape else None,
                    expand_if_scalar(jnp.asarray(a)) if self.parameter_dim else None
                )


class Scaler:
    forward: callable 
    reverse: callable
    def __init__(self, x_min=0., x_max=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


class Normer:
    forward: callable 
    reverse: callable
    def __init__(self, x_mean=0., x_std=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: (x - x_mean) / x_std
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: y * x_std + x_mean


@dataclass
class ScalerDataset:
    name: str
    train_dataloader: Union[_TorchDataLoader | _InMemoryDataLoader]
    valid_dataloader: Union[_TorchDataLoader | _InMemoryDataLoader]
    data_shape: Tuple[int]
    context_shape: Tuple[int]
    parameter_dim: int
    scaler: Scaler
    label_fn: Callable