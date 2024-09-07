import os 
from typing import Tuple 
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import Key, Array
import numpy as np
import torch
from torchvision import transforms
import powerbox

from .utils import Scaler, ScalerDataset, _TorchDataLoader

data_dir = "/project/ls-gruen/users/jed.homer/data/fields/"


def get_fields(key: Key, Q, n_pix: int, n_fields: int):
    G = np.zeros((n_fields, 1, n_pix, n_pix))
    L = np.zeros((n_fields, 1, n_pix, n_pix))
    key = jr.key_data(key) # Convert jr.key() to jr.PRNGKey()
    print("Building fields...")
    for n in range(n_fields):
        A, B = Q[n]
        pk_fn = lambda k: A * k ** -B
        G[n] = powerbox.PowerBox(
            N=n_pix,                 
            dim=2,                   
            pk=pk_fn,
            boxlength=1.0,           
            seed=n + jnp.sum(key),               
            # ensure_physical=True       
        ).delta_x()
        L[n] = powerbox.LogNormalPowerBox(
            N=n_pix,                 
            dim=2,                   
            pk=pk_fn,
            boxlength=1.0,           
            seed=n + jnp.sum(key),               
            # ensure_physical=True       
        ).delta_x()
        print(f"\r {n=}", end="")
    return G, L
    

def get_data(key: Key, n_pix: int) -> Tuple[np.ndarray, np.ndarray]:
    """
        Load Gaussian and lognormal fields
    """

    if 0:
        G = np.load(os.path.join(data_dir, f"G_{n_pix=}.npy"))
        L = np.load(os.path.join(data_dir, f"LN_{n_pix=}.npy"))
        Q = np.load(os.path.join(data_dir, f"field_parameters_{n_pix=}.npy"))
    else:
        key_A, key_B = jr.split(key)
        Q = np.stack(
            [
                jr.uniform(key_A, (10_000,), minval=1., maxval=3.),
                jr.uniform(key_B, (10_000,), minval=1., maxval=3.)
            ],
            axis=1
        )
        G, L = get_fields(key, Q, n_pix, n_fields=10_000)

        np.save(os.path.join(data_dir, f"G_{n_pix=}.npy"), G)
        np.save(os.path.join(data_dir, f"LN_{n_pix=}.npy"), L)
        np.save(os.path.join(data_dir, f"field_parameters_{n_pix=}.npy"), Q)

    # X = X.reshape((-1, 1, n_pix, dx, n_pix, dx)).mean(axis=(3, 5))
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


def get_grf_labels(n_pix: int) -> np.ndarray:
    Q = np.load(os.path.join(data_dir, f"G_{n_pix=}.npy"))
    A = np.load(os.path.join(data_dir, f"field_parameters_{n_pix=}.npy"))
    return Q, A


def grfs(key, n_pix, split=0.5):
    key_data, key_train, key_valid = jr.split(key, 3)

    data_shape = (1, n_pix, n_pix)
    context_shape = (1, n_pix, n_pix)
    parameter_dim = 2

    Q, X, A = get_data(key_data, n_pix) 

    print("Fields data:", X.shape, Q.shape)

    min = X.min()
    max = X.max()
    X = (X - min) / (max - min) # ... -> [0, 1]

    # min = Q.min()
    # max = Q.max()
    # Q = (Q - min) / (max - min) # ... -> [0, 1]

    scaler = Scaler() # [0,1] -> [-1,1]

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

    n_train = int(split * len(X))
    train_dataset = MapDataset(
        (X[:n_train], Q[:n_train], A[:n_train]), transform=train_transform
    )
    valid_dataset = MapDataset(
        (X[n_train:], Q[n_train:], A[n_train:]), transform=valid_transform
    )
    train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
    valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)

    # Don't have many maps
    # train_dataloader = _InMemoryDataLoader(
    #     data=X[:n_train], targets=Q[:n_train], key=key_train
    # )
    # valid_dataloader = _InMemoryDataLoader(
    #     data=X[n_train:], targets=Q[n_train:], key=key_valid
    # )

    def label_fn(key, n):
        Q, A = get_grf_labels(n_pix)
        ix = jr.choice(key, jnp.arange(len(Q)), (n,))
        Q = Q[ix]
        A = A[ix]
        return Q, A

    return ScalerDataset(
        name="grfs",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        scaler=scaler,
        label_fn=label_fn
    )