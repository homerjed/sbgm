import abc
from functools import partial
from typing import Tuple, Callable, Optional, Generator, Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker
import numpy as np
import torch


def expand_if_scalar(x):
    return x[:, jnp.newaxis] if x.ndim == 1 else x


def exists(v):
    return v is not None 


def default(v, d):
    return v if exists(v) else d


class Scaler:
    forward: Callable 
    reverse: Callable
    def __init__(self, x_min: Scalar | float = 0., x_max: Scalar | float = 1.):
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


class Normer:
    forward: Callable 
    reverse: Callable
    def __init__(self, x_mean: Scalar | float = 0., x_std: Scalar | float = 1.):
        self.forward = lambda x: (x - x_mean) / x_std
        self.reverse = lambda y: y * x_std + x_mean


class Identity:
    forward: Callable 
    reverse: Callable
    def __init__(self):
        self.forward = lambda x: x
        self.reverse = lambda x: x


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        X: np.ndarray | Array, 
        Q: Optional[np.ndarray | Array] = None, 
        A: Optional[np.ndarray | Array] = None, 
        *, 
        process_fn: Optional[Scaler | Normer | Identity] = None,
        key: PRNGKeyArray
    ):
        self.X = jnp.asarray(X)
        self.Q = jnp.asarray(Q) if exists(Q) else Q
        self.A = jnp.asarray(A) if exists(A) else A
        self.process_fn = default(process_fn, Identity()) 
        self.key = key

    def loop(
        self, batch_size: int
    ) -> Generator[Tuple[Array, Optional[Array], Optional[Array]], None, None]:
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
                    self.process_fn.forward(self.X[batch_perm]) if exists(self.process_fn) else self.X[batch_perm], 
                    self.Q[batch_perm] if self.Q is not None else None, 
                    self.A[batch_perm] if self.A is not None else None 
                )
                start = end
                end = start + batch_size


class TorchDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        data_shape: Sequence[int],
        context_shape: Optional[Sequence[int]],
        parameter_dim: Optional[int],
        *, 
        process_fn: Optional[Scaler | Normer | Identity] = None,
        num_workers: Optional[int] = None, 
        key: PRNGKeyArray
    ):
        self.dataset = dataset
        self.context_shape = context_shape 
        self.parameter_dim = parameter_dim 
        self.seed = jr.randint(key, (), 0, 1_000_000).item() 
        self.process_fn = default(process_fn, Identity()) 
        self.num_workers = num_workers

    def loop(
        self, batch_size: int, num_workers: int = 2
    ) -> Generator[
        Tuple[Array, Optional[Array], Optional[Array]], None, None
    ]:
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
                x = jnp.asarray(x)
                yield ( 
                    self.process_fn.forward(x) if exists(self.process_fn) else x,
                    expand_if_scalar(jnp.asarray(q)) if self.context_shape else None,
                    expand_if_scalar(jnp.asarray(a)) if self.parameter_dim else None
                )


@jaxtyped(typechecker=typechecker)
@dataclass
class ScalerDataset:
    name: str
    train_dataloader: TorchDataLoader | InMemoryDataLoader
    valid_dataloader: TorchDataLoader | InMemoryDataLoader
    data_shape: Tuple[int, ...]
    context_shape: Tuple[int, ...] | None
    parameter_dim: int | None
    process_fn: Scaler | Normer | Identity | None
    label_fn: Callable


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        # Tuple of (images, contexts, targets), turn them into tensors
        self.tensors = tuple(
            torch.as_tensor(tensor) for tensor in tensors
        )
        self.transform = transform
        assert all(
            self.tensors[0].size(0) == tensor.size(0) 
            for tensor in self.tensors
        )

    def __getitem__(self, index):
        x = self.tensors[0][index] # Fields
        q = self.tensors[1][index] # Parameters
        a = self.tensors[2][index] # Parameters

        if self.transform:
            x = self.transform(x)

        return x, q, a

    def __len__(self):
        return self.tensors[0].size(0)


@jaxtyped(typechecker=typechecker)
def dataset_from_tensors(
    X: Float[Array, "n ..."],
    Q: Optional[Float[Array, "n ..."]] = None,
    A: Optional[Float[Array, "n _"]] = None,
    *,
    key: PRNGKeyArray,
    process_fn: Optional[Scaler | Normer | Identity] = None,
    split: float = 0.8,
    in_memory: bool = False,
    name: Optional[str] = None
):
    """
        Creates a ScalerDataset object from in-memory tensors with optional conditioning and parameter targets.

        Splits the data into training and validation subsets, applies optional preprocessing, 
        and returns a dataset with a label function for downstream usage.

        Parameters
        ----------
        X : Float[Array, "n ..."]
            Input data array of shape (n, ...) where n is the number of data points.
        Q : Optional[Float[Array, "n ..."]]
            Conditioning variables (e.g. features or context), optional. Must align with `X` if provided.
        A : Optional[Float[Array, "n _"]]
            Parameters or labels (e.g. target variables) associated with each sample, optional.
        key : PRNGKeyArray
            PRNG key for random operations such as sampling indices or splitting.
        process_fn : Optional[Scaler | Normer | Identity], default=None
            Optional processing function to normalize or transform the inputs.
        split : float, default=0.8
            Fraction of data to use for training. The remainder is used for validation.
        in_memory : bool, default=False
            Whether to load the entire dataset into memory or use an iterable-style loader.
        name : Optional[str], default=None
            Name identifier for the dataset.

        Returns
        -------
        ScalerDataset
            A dataset wrapper object containing training/validation dataloaders, 
            shape metadata, optional preprocessing, and a label generation function.

        Notes
        -----
        - If `Q` and `A` are both provided, `label_fn` will randomly sample a subset of them.
        - If `in_memory=True`, `InMemoryDataLoader` will be used instead of `TorchDataLoader`.
        - Requires the custom `ScalerDataset`, `InMemoryDataLoader`, `TorchDataLoader`, and `TensorDataset` classes to be defined elsewhere.
    """
    key_train, key_valid = jr.split(key)

    n_train = int(split * X.shape[0])
    data_shape = X.shape[1:]  # Exclude the first dimension (n)
    context_shape = Q.shape[1:] if exists(Q) else None
    parameter_dim = A.shape[1] if exists(A) else None

    train_set = (
        X[:n_train], 
        Q[:n_train] if Q is not None else None, 
        A[:n_train] if A is not None else None,
    )
    valid_set = (
        X[n_train:], 
        Q[n_train:] if Q is not None else None, 
        A[n_train:] if A is not None else None,
    )

    def label_fn(
        Q: Optional[Float[Array, "n ..."]],
        A: Optional[Float[Array, "n _"]],
        key: PRNGKeyArray, 
        n: int
    ) -> Tuple[Array, Array]:
        if exists(Q) and exists(A):
            ix = jr.choice(key, jnp.arange(len(Q)), (n,))
            Q = Q[ix]
            A = A[ix]
        return Q, A

    if in_memory:
        train_dataloader = InMemoryDataLoader(
            *train_set, key=key_train
        )
        valid_dataloader = InMemoryDataLoader(
            *valid_set, key=key_valid
        )
    else:
        train_dataloader = TorchDataLoader(
            TensorDataset(train_set), 
            data_shape=data_shape,
            context_shape=context_shape, 
            parameter_dim=parameter_dim,
            key=key_train
        )
        valid_dataloader = TorchDataLoader(
            TensorDataset(valid_set), 
            data_shape=data_shape,
            context_shape=context_shape, 
            parameter_dim=parameter_dim,
            key=key_valid
        )

    return ScalerDataset(
        name=name if exists(name) else "dataset",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        label_fn=partial(label_fn, Q=Q, A=A),
        process_fn=process_fn
    )