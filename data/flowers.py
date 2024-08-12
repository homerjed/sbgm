import jax.random as jr 
from torchvision import transforms, datasets

from .utils import Scaler, ScalerDataset, _TorchDataLoader


def flowers(key: jr.PRNGKey, n_pix: int) -> ScalerDataset:
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

    train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
    valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)
    return ScalerDataset(
        name="flowers",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler
    )
