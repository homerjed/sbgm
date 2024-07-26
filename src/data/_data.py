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


# def flowers(key, n_pix):
#     key_train, key_valid = jr.split(key)
#     data_shape = (3, n_pix, n_pix)
#     context_shape = (1,)

#     scaler = Scaler()

#     train_transform = transforms.Compose(
#         [
#             transforms.Resize((n_pix, n_pix)),
#             transforms.RandomCrop(n_pix, padding=4, padding_mode='reflect'),
#             # transforms.Grayscale(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(scaler.forward)
#         ]
#     )
#     valid_transform = transforms.Compose(
#         [
#             transforms.Resize((n_pix, n_pix)),
#             transforms.RandomCrop(n_pix, padding=4, padding_mode='reflect'),
#             # transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Lambda(scaler.forward)
#         ]
#     )
#     train_dataset = datasets.Flowers102(
#         "datasets/" + "flowers", 
#         split="train", 
#         download=True, 
#         transform=train_transform
#     )
#     valid_dataset = datasets.Flowers102(
#         "datasets/" + "flowers", 
#         split="val", 
#         download=True, 
#         transform=valid_transform
#     )

#     train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
#     valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)
#     return ScalerDataset(
#         name="flowers",
#         train_dataloader=train_dataloader,
#         valid_dataloader=valid_dataloader,
#         data_shape=data_shape,
#         context_shape=context_shape,
#         scaler=scaler
#     )


# def cifar10(key):
#     key_train, key_valid = jr.split(key)
#     n_pix = 32 # Native resolution for CIFAR10 
#     data_shape = (3, n_pix, n_pix)
#     context_shape = (1,)

#     scaler = Scaler(x_min=0., x_max=1.)

#     train_transform = transforms.Compose(
#         [
#             transforms.Resize((n_pix, n_pix)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ToTensor(), # This magically [0,255] -> [0,1]??
#             transforms.Lambda(scaler.forward)
#         ]
#     )
#     valid_transform = transforms.Compose(
#         [
#             transforms.Resize((n_pix, n_pix)),
#             transforms.ToTensor(),
#             transforms.Lambda(scaler.forward)
#         ]
#     )
#     train_dataset = datasets.CIFAR10(
#         "datasets/" + "cifar10", 
#         train=True, 
#         download=True, 
#         transform=train_transform
#     )
#     valid_dataset = datasets.CIFAR10(
#         "datasets/" + "cifar10", 
#         train=False, 
#         download=True, 
#         transform=valid_transform
#     )

#     train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
#     valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)
#     return ScalerDataset(
#         name="cifar10",
#         train_dataloader=train_dataloader,
#         valid_dataloader=valid_dataloader,
#         data_shape=data_shape,
#         context_shape=context_shape,
#         scaler=scaler
#     )


# def mnist(key):
#     key_train, key_valid = jr.split(key)
#     data_shape = (1, 28, 28)
#     context_shape = (1,)
#     # Scale data to be in range [-1, 1]
#     # mean = 0.1307
#     # std = 0.3081
#     # max = 1.
#     # min = 0.

#     scaler = Scaler() # [0, 1] -> [-1, 1]

#     # MNIST is small enough that the whole dataset can be placed in memory, so
#     # we can actually use a faster method of data loading.
#     train_dataset = datasets.MNIST(
#         "datasets/" + "mnist", train=True, download=True
#     )
#     valid_dataset = datasets.MNIST(
#         "datasets/" + "mnist", train=False, download=True
#     )

#     def tensor_to_array(tensor):
#         return jnp.asarray(tensor.numpy())

#     # Scale the data to the range [0, 1] 
#     train_data = tensor_to_array(train_dataset.data)[:, jnp.newaxis, ...] / 255.
#     train_targets = tensor_to_array(train_dataset.targets)[:, jnp.newaxis]
#     valid_data = tensor_to_array(valid_dataset.data)[:, jnp.newaxis, ...] / 255.
#     valid_targets = tensor_to_array(valid_dataset.targets)[:, jnp.newaxis]

#     min, max = train_data.min(), train_data.max()
#     # train_data = (train_data - mean) / std
#     # valid_data = (valid_data - mean) / std
#     train_data = (train_data - min) / (max - min)
#     valid_data = (valid_data - min) / (max - min)

#     train_dataloader = _InMemoryDataLoader(
#         train_data, train_targets, key=key_train
#     )
#     valid_dataloader = _InMemoryDataLoader(
#         valid_data, valid_targets, key=key_valid
#     )
#     return ScalerDataset(
#         name="mnist",
#         train_dataloader=train_dataloader,
#         valid_dataloader=valid_dataloader,
#         data_shape=data_shape,
#         context_shape=context_shape,
#         scaler=scaler
#         # mean=mean,
#         # std=std,
#         # max=max,
#         # min=min
#     )


# def moons(key):
#     key_train, key_valid = jr.split(key)
#     data_shape = (2,)
#     context_shape = (1,)

#     Xt, Yt = make_moons(
#         5_000, noise=0.05, random_state=int(key_train.sum())
#     )
#     Xv, Yv = make_moons(
#         5_000, noise=0.05, random_state=int(key_valid.sum())
#     )

#     min = Xt.min()
#     max = Xt.max()
#     mean = Xt.mean()
#     std = Xt.std()

#     # (We do need to handle normalisation ourselves though.)
#     # train_data = (Xt - min) / (max - min)
#     # valid_data = (Xv - min) / (max - min)
#     train_data = (Xt - mean) / std
#     valid_data = (Xv - mean) / std
    
#     train_dataloader = _InMemoryDataLoader(
#         jnp.asarray(train_data), 
#         jnp.asarray(Yt)[:, jnp.newaxis], 
#         key=key_train
#     )
#     valid_dataloader = _InMemoryDataLoader(
#         jnp.asarray(valid_data), 
#         jnp.asarray(Yv)[:, jnp.newaxis], 
#         key=key_valid
#     )

#     class _Scaler:
#         forward: callable 
#         reverse: callable
#         def __init__(self):
#             # [0, 1] -> [-1, 1]
#             self.forward = lambda x: x
#             # [-1, 1] -> [0, 1]
#             self.reverse = lambda y: y

#     return ScalerDataset(
#         name="moons",
#         train_dataloader=train_dataloader,
#         valid_dataloader=valid_dataloader,
#         data_shape=data_shape,
#         context_shape=context_shape,
#         scaler=_Scaler()
#     )


# class MapDataset(torch.utils.data.Dataset):
#     def __init__(self, tensors, transform=None):
#         # Tuple of (images, targets), turn them into tensors
#         self.tensors = tuple(
#             torch.as_tensor(tensor) 
#             for tensor in tensors
#         )
#         self.transform = transform
#         assert all(
#             self.tensors[0].size(0) == tensor.size(0) 
#             for tensor in self.tensors
#         )

#     def __getitem__(self, index):
#         x = self.tensors[0][index]
#         y = self.tensors[1][index]

#         if self.transform:
#             x = self.transform(x)

#         return x, y

#     def __len__(self):
#         return self.tensors[0].size(0)


# def get_data(field_type: str, n_pix: int) -> Tuple[Array, Array]:
#     """
#         Load lognormal, gaussian or Quijote fields
#     """
#     data_dir = "/project/ls-gruen/users/jed.homer/data/fields/"
#     quijote_dir = "/project/ls-gruen/users/jed.homer/quijote_pdfs/data/"

#     filename = field_type + "_fields.npy"

#     if field_type.upper() in ["LN", "G"]:
#         X = np.load(os.path.join(data_dir, filename))[:, np.newaxis, ...]
#         Q = np.load(os.path.join(data_dir, "field_parameters.npy"))
#     if field_type.upper() == "QUIJOTE":
#         X = np.load(os.path.join(data_dir, "quijote_fields.npy"))[:, np.newaxis, ...]
#         Q = np.load(os.path.join(quijote_dir, "ALL_LATIN_PDFS_PARAMETERS.npy"))

#     dx = int(256 / n_pix)
#     X = X.reshape((-1, 1, n_pix, dx, n_pix, dx)).mean(axis=(3, 5))
#     return X, Q


def get_quijote_or_field_labels(dataset_name: str) -> np.ndarray:
    """ Get labels only of fields dataset """
    data_dir = "/project/ls-gruen/users/jed.homer/data/fields/"
    quijote_dir = "/project/ls-gruen/users/jed.homer/quijote_pdfs/data/"

    if "LN" in dataset_name or "G" in dataset_name:
        Q = np.load(os.path.join(data_dir, "field_parameters.npy"))
    if "Quijote" in dataset_name:
        Q = np.load(os.path.join(quijote_dir, "ALL_LATIN_PDFS_PARAMETERS.npy"))
    return Q


# def fields(key, field_type, n_pix, split=0.5):
#     key_train, key_valid = jr.split(key)
#     data_shape = (1, n_pix, n_pix)
#     # Scale data to be in range [-1, 1]
#     # mean = (0.5,)
#     # std = (0.5,)

#     X, Q = get_data(field_type, n_pix)

#     print("Fields data:", X.shape, Q.shape)

#     min = X.min()
#     max = X.max()

#     print(X.min(), X.max())
#     X = (X - min) / (max - min)
#     print(X.min(), X.max())

#     scaler = Scaler() # [0,1] -> [-1,1]

#     data_shape = X.shape[1:]
#     context_shape = Q.shape[1:]

#     train_transform = transforms.Compose(
#         [
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.Lambda(scaler.forward)
#         ]
#     )
#     valid_transform = transforms.Compose(
#         [transforms.Lambda(scaler.forward)]
#     )

#     n_train = int(split * len(X))
#     train_dataset = MapDataset(
#         (X[:n_train], Q[:n_train]), transform=train_transform
#     )
#     valid_dataset = MapDataset(
#         (X[n_train:], Q[n_train:]), transform=valid_transform
#     )
#     train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
#     valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)

#     # Don't have many maps
#     # train_dataloader = _InMemoryDataLoader(
#     #     data=X[:n_train], targets=Q[:n_train], key=key_train
#     # )
#     # valid_dataloader = _InMemoryDataLoader(
#     #     data=X[n_train:], targets=Q[n_train:], key=key_valid
#     # )
#     return ScalerDataset(
#         name=field_type + " fields",
#         train_dataloader=train_dataloader,
#         valid_dataloader=valid_dataloader,
#         data_shape=data_shape,
#         context_shape=context_shape,
#         scaler=scaler
#         # mean=jnp.array(mean)[:, jnp.newaxis, jnp.newaxis],
#         # std=jnp.array(std)[:, jnp.newaxis, jnp.newaxis],
#         # max=max,
#         # min=min
#     )

def get_labels(key, dataset, config):
    # Sample random labels or use parameter prior for labels
    if "fields" in dataset.name:
        Q = get_quijote_or_field_labels(dataset.name)
        ix = jr.choice(key, jnp.arange(len(Q)), (config.sample_size ** 2,))
        Q = Q[ix]
    if "moons" in dataset.name:
        Q = jr.choice(key, jnp.array([0., 1.]), (config.sample_size ** 2,))[:, jnp.newaxis]
    if dataset.name in ["mnist", "cifar10", "flowers"]:
        # CIFAR10, MNIST, ...
        Q = jr.choice(key, jnp.arange(10), (config.sample_size ** 2,))[:, jnp.newaxis]
    return Q