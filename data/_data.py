import os 
import abc
from typing import Tuple, Callable, Union
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import Key, Array
import numpy as np
import torch


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
        min = torch.iinfo(torch.int32).min
        max = torch.iinfo(torch.int32).max
        self.seed = jr.randint(key, (), min, max).item() 

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


@dataclass
class Dataset:
    name: str
    train_dataloader: Union[_TorchDataLoader, _InMemoryDataLoader]
    valid_dataloader: Union[_TorchDataLoader, _InMemoryDataLoader]
    data_shape: Tuple[int]
    context_shape: Tuple[int]
    mean: Array
    std: Array
    max: Array
    min: Array


class Scaler:
    forward: Callable 
    reverse: Callable
    def __init__(self, x_min=0., x_max=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


@dataclass
class ScalerDataset:
    name: str
    train_dataloader: _TorchDataLoader | _InMemoryDataLoader
    valid_dataloader: _TorchDataLoader | _InMemoryDataLoader
    data_shape: Tuple[int]
    context_shape: Tuple[int]
    scaler: Scaler


def get_quijote_or_field_labels(dataset_name: str) -> np.ndarray:
    """ Get labels only of fields dataset """
    data_dir = "/project/ls-gruen/users/jed.homer/data/fields/"
    quijote_dir = "/project/ls-gruen/users/jed.homer/quijote_pdfs/data/"

    if "LN" in dataset_name or "G" in dataset_name:
        Q = np.load(os.path.join(data_dir, "field_parameters.npy"))
    if "Quijote" in dataset_name:
        Q = np.load(os.path.join(quijote_dir, "ALL_LATIN_PDFS_PARAMETERS.npy"))
    return Q


def get_grf_labels(n_pix: int) -> np.ndarray:
    data_dir = "/project/ls-gruen/users/jed.homer/data/fields/"
    Q = np.load(os.path.join(data_dir, f"G_{n_pix=}.npy"))
    A = np.load(os.path.join(data_dir, f"field_parameters_{n_pix=}.npy"))
    return Q, A


def get_labels(key: Key, dataset_name: str, config) -> Array:
    # Sample random labels or use parameter prior for labels
    if "fields" in dataset_name:
        Q = get_quijote_or_field_labels(dataset_name)
        ix = jr.choice(key, jnp.arange(len(Q)), (config.sample_size ** 2,))
        Q = Q[ix]
        A = None
    if "grfs" in dataset_name:
        Q, A = get_grf_labels(config.n_pix)
        ix = jr.choice(key, jnp.arange(len(Q)), (config.sample_size ** 2,))
        Q = Q[ix]
        A = A[ix]
    if "moons" in dataset_name:
        Q = jr.choice(key, jnp.array([0., 1.]), (config.sample_size ** 2,))[:, jnp.newaxis]
        A = None
    if dataset_name in ["mnist", "cifar10", "flowers"]:
        # CIFAR10, MNIST, ...
        Q = None
        A = jr.choice(key, jnp.arange(10), (config.sample_size ** 2,))[:, jnp.newaxis]
    return Q, A