import os 
from functools import partial
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import PRNGKeyArray, Float, Array
import numpy as np
import torch
from torchvision import transforms
import powerbox

from .utils import Scaler, Normer, ScalerDataset, TorchDataLoader, InMemoryDataLoader


def get_fields(key: PRNGKeyArray, Q, n_pix: int, n_fields: int):
    G = np.zeros((n_fields, 1, n_pix, n_pix))
    L = np.zeros((n_fields, 1, n_pix, n_pix))

    key = jr.key_data(key) 

    print("Building fields...")
    for n in range(n_fields):
        A, B = Q[n]
        pk_fn = lambda k: A * k ** -B
        seed = n + jnp.sum(key)

        G[n] = powerbox.PowerBox(
            N=n_pix,                 
            dim=2,                   
            pk=pk_fn,
            boxlength=1.0,           
            seed=seed,               
        ).delta_x()

        L[n] = powerbox.LogNormalPowerBox(
            N=n_pix,                 
            dim=2,                   
            pk=pk_fn,
            boxlength=1.0,           
            seed=seed,               
        ).delta_x()

        print(f"\r {n=}", end="")

    G = jnp.asarray(G)
    L = jnp.asarray(L)

    return G, L
    

def get_data(
    key: PRNGKeyArray, 
    n_pix: int, 
    n_fields: int, 
    data_dir: Optional[str] = None
) -> Tuple[
    Float[Array, "n 1 n_pix n_pix"],
    Float[Array, "n 1 n_pix n_pix"],
    Float[Array, "n 2"]
]:
    """
        Load Gaussian and lognormal fields
    """

    key_A, key_B = jr.split(key)
    Q = jnp.stack(
        [
            jr.uniform(key_A, (n_fields,), minval=1., maxval=3.),
            jr.uniform(key_B, (n_fields,), minval=1., maxval=3.)
        ],
        axis=1
    )
    G, L = get_fields(key, Q, n_pix, n_fields=n_fields)

    if data_dir is not None:
        np.save(os.path.join(data_dir, f"G_{n_pix=}.npy"), G)
        np.save(os.path.join(data_dir, f"LN_{n_pix=}.npy"), L)
        np.save(os.path.join(data_dir, f"field_parameters_{n_pix=}.npy"), Q)

    return G, L, Q


class MapDataset(torch.utils.data.Dataset):
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
        x = self.tensors[0][index] # GRFs
        q = self.tensors[1][index] # LNs
        a = self.tensors[2][index] # Parameters

        if self.transform:
            x = self.transform(x)

        return x, q, a

    def __len__(self):
        return self.tensors[0].size(0)


def grfs(
    key: PRNGKeyArray, 
    n_pix: int, 
    split: float = 0.5, 
    n_fields: int = 10_000, 
    *, 
    in_memory: bool = True
) -> ScalerDataset:

    key_data, key_train, key_valid = jr.split(key, 3)

    data_shape = (1, n_pix, n_pix)
    context_shape = (1, n_pix, n_pix)
    parameter_dim = 2

    Q, X, A = get_data(key_data, n_pix, n_fields) 

    print("\nFields data:", X.shape, Q.shape)

    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)  # Standardize fields
    Q = (Q - jnp.mean(Q, axis=0)) / jnp.std(Q, axis=0)  # Standardize fields

    scaler = Normer() #Scaler() # [0,1] -> [-1,1]

    n_train = int(split * n_fields)

    # If we don't have many maps or they're not too big
    if in_memory:
        train_dataloader = InMemoryDataLoader(
            X=X[:n_train], Q=Q[:n_train], A=A[:n_train], key=key_train
        )
        valid_dataloader = InMemoryDataLoader(
            X=X[n_train:], Q=Q[n_train:], A=A[n_train:], key=key_valid
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Lambda(scaler.forward)
            ]
        )
        valid_transform = transforms.Compose(
            [transforms.Lambda(scaler.forward)]
        )
        train_dataset = MapDataset(
            (X[:n_train], Q[:n_train], A[:n_train]), transform=train_transform
        )
        valid_dataset = MapDataset(
            (X[n_train:], Q[n_train:], A[n_train:]), transform=valid_transform
        )
        train_dataloader = TorchDataLoader(
            train_dataset, 
            data_shape=data_shape, 
            context_shape=context_shape, 
            parameter_dim=parameter_dim, 
            key=key_train
        )
        valid_dataloader = TorchDataLoader(
            valid_dataset, 
            data_shape=data_shape, 
            context_shape=context_shape,
            parameter_dim=parameter_dim, 
            key=key_valid
        )

    def label_fn(
        key: PRNGKeyArray, 
        n: int,
        Q: Float[Array, "n 1 h w"], 
        A: Float[Array, "n p"],
    ) -> Tuple[Float[Array, "b 1 h w"], Float[Array, "b p"]]:
        # Sample conditioning fields and parameters
        ix = jr.choice(key, jnp.arange(len(Q)), (n,))
        return Q[ix], A[ix]

    return ScalerDataset(
        name="grfs",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        process_fn=scaler,
        label_fn=partial(label_fn, Q=Q, A=A)
    )