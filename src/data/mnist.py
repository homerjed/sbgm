import jax.random as jr 
import jax.numpy as jnp
from torchvision import datasets

from .utils import Scaler, ScalerDataset, _InMemoryDataLoader


def mnist(key: jr.PRNGKey) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    data_shape = (1, 28, 28)
    context_shape = (1,)

    scaler = Scaler() # [0, 1] -> [-1, 1]

    # MNIST is small enough that the whole dataset can be placed in memory, so
    # we can actually use a faster method of data loading.
    train_dataset = datasets.MNIST(
        "datasets/" + "mnist", train=True, download=True
    )
    valid_dataset = datasets.MNIST(
        "datasets/" + "mnist", train=False, download=True
    )

    def tensor_to_array(tensor):
        return jnp.asarray(tensor.numpy())

    # Scale the data to the range [0, 1] 
    train_data = tensor_to_array(train_dataset.data)[:, jnp.newaxis, ...] / 255.
    train_targets = tensor_to_array(train_dataset.targets)[:, jnp.newaxis]
    valid_data = tensor_to_array(valid_dataset.data)[:, jnp.newaxis, ...] / 255.
    valid_targets = tensor_to_array(valid_dataset.targets)[:, jnp.newaxis]

    min, max = train_data.min(), train_data.max()
    # train_data = (train_data - mean) / std
    # valid_data = (valid_data - mean) / std
    train_data = (train_data - min) / (max - min)
    valid_data = (valid_data - min) / (max - min)

    train_dataloader = _InMemoryDataLoader(
        train_data, train_targets, key=key_train
    )
    valid_dataloader = _InMemoryDataLoader(
        valid_data, valid_targets, key=key_valid
    )
    return ScalerDataset(
        name="mnist",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler
    )