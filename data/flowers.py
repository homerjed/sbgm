import jax.random as jr 
import jax.numpy as jnp
from jaxtyping import Key
from torchvision import transforms, datasets

from .utils import Scaler, ScalerDataset, TorchDataLoader


def flowers(key: Key, n_pix: int) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    data_shape = (3, n_pix, n_pix)
    context_shape = (1,)

    scaler = Scaler()

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomCrop(n_pix, padding=4, padding_mode='reflect'),
            # transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomCrop(n_pix, padding=4, padding_mode='reflect'),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )
    train_dataset = datasets.Flowers102(
        "datasets/" + "flowers", 
        split="train", 
        download=True, 
        transform=train_transform
    )
    valid_dataset = datasets.Flowers102(
        "datasets/" + "flowers", 
        split="val", 
        download=True, 
        transform=valid_transform
    )

    train_dataloader = TorchDataLoader(train_dataset, key=key_train)
    valid_dataloader = TorchDataLoader(valid_dataset, key=key_valid)

    def label_fn(key, n):
        Q = None
        A = jr.choice(key, jnp.arange(10), (n,))[:, jnp.newaxis]
        return Q, A

    return ScalerDataset(
        name="flowers",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler,
        label_fn=label_fn
    )
