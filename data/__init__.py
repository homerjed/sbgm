from jaxtyping import Key
from ml_collections import ConfigDict

from .cifar10 import cifar10
from .mnist import mnist 
from .flowers import flowers
from .moons import moons
from .grfs import grfs
from .quijote import quijote
from .utils import Scaler, ScalerDataset, _InMemoryDataLoader, _TorchDataLoader


def get_dataset(
    datasets_path: str,
    key: Key, 
    config: ConfigDict
) -> ScalerDataset:

    dataset_name = config.dataset_name.lower()

    if dataset_name == "flowers":
        dataset = flowers(key, n_pix=config.n_pix)
    if dataset_name == "cifar10":
        dataset = cifar10(datasets_path, key)
    if dataset_name == "mnist":
        dataset = mnist(datasets_path, key)
    if dataset_name == "moons":
        dataset = moons(key)
    if dataset_name == "grfs":
        dataset = grfs(key, n_pix=config.n_pix)
    if dataset_name == "quijote":
        dataset = quijote(key, n_pix=config.n_pix, split=0.9)

    return dataset