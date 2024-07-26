import abc
from typing import Tuple 
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
    def __init__(self, data, targets, *, key):
        self.data = data 
        self.targets = targets 
        self.key = key

    def loop(self, batch_size):
        dataset_size = self.data.shape[0]
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
                yield self.data[batch_perm], self.targets[batch_perm]
                start = end
                end = start + batch_size


class _TorchDataLoader(_AbstractDataLoader):
    def __init__(self, dataset, *, key):
        self.dataset = dataset
        # min = torch.iinfo(torch.int32).min
        # max = torch.iinfo(torch.int32).max
        self.seed = int(key.sum().item()) #jr.randint(key, (), min, max).item() # key.sum().item() ?

    def loop(self, batch_size):
        generator = torch.Generator().manual_seed(self.seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )
        while True:
            for tensor, label in dataloader:
                yield (
                    jnp.asarray(tensor), 
                    expand_if_scalar(jnp.asarray(label))
                )


Loader = _TorchDataLoader | _InMemoryDataLoader


@dataclass
class Dataset:
    name: str
    train_dataloader: Loader
    valid_dataloader: Loader
    data_shape: Tuple[int]
    context_shape: Tuple[int]
    mean: jax.Array
    std: jax.Array
    max: jax.Array
    min: jax.Array


class Scaler:
    forward: callable 
    reverse: callable
    def __init__(self, x_min=0., x_max=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


@dataclass
class ScalerDataset:
    name: str
    train_dataloader: Loader
    valid_dataloader: Loader
    data_shape: Tuple[int]
    context_shape: Tuple[int]
    scaler: Scaler