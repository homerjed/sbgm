import os
from typing import Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
import optax
import numpy as np 
import ml_collections

import sbgm
import data 
import configs 


def get_model(
    model_key: Key, 
    model_type: str, 
    data_shape: Sequence[int], 
    context_shape: Sequence[int], 
    parameter_dim: int,
    config: ml_collections.ConfigDict
) -> eqx.Module:
    if model_type == "Mixer":
        model = sbgm.models.Mixer2d(
            data_shape,
            **config.model_args,
            t1=config.t1,
            q_dim=context_shape,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNet":
        model = sbgm.models.UNet(
            data_shape=data_shape,
            is_biggan=config.model.is_biggan,
            dim_mults=config.model.dim_mults,
            hidden_size=config.model.hidden_size,
            heads=config.model.heads,
            dim_head=config.model.dim_head,
            dropout_rate=config.model.dropout_rate,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attn_resolutions,
            final_activation=config.model.final_activation,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNetXY":
        model = sbgm.models.UNetXY(
            data_shape=data_shape,
            is_biggan=config.model.is_biggan,
            dim_mults=config.model.dim_mults,
            hidden_size=config.model.hidden_size,
            heads=config.model.heads,
            dim_head=config.model.dim_head,
            dropout_rate=config.model.dropout_rate,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attn_resolutions,
            final_activation=config.model.final_activation,
            q_dim=context_shape[0], # Just grab channel assuming 'q' is a map like x
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "mlp":
        model = sbgm.models.ResidualNetwork(
            in_size=np.prod(data_shape),
            **config.model_args,
            y_dim=parameter_dim,
            key=model_key
        )
    if model_type == "DiT":
        raise NotImplementedError
    return model


def get_dataset(
    datasets_path: str,
    key: Key, 
    config: ml_collections.ConfigDict
) -> data.ScalerDataset:
    dataset_name = config.dataset_name
    if dataset_name == "flowers":
        dataset = data.flowers(key, n_pix=config.n_pix)
    if dataset_name == "cifar10":
        dataset = data.cifar10(datasets_path, key)
    if dataset_name == "mnist":
        dataset = data.mnist(datasets_path, key)
    if dataset_name == "moons":
        dataset = data.moons(key)
    if dataset_name == "grfs":
        dataset = data.grfs(key, n_pix=config.n_pix)
    if dataset_name == "quijote":
        dataset = data.quijote(key, n_pix=config.n_pix, split=0.9)
    return dataset


def get_sde(config: ml_collections.ConfigDict) -> sbgm.sde.SDE:
    name = config.sde + "SDE"
    assert name in ["VESDE", "VPSDE", "SubVPSDE"]
    sde = getattr(sbgm.sde, name)
    return sde(
        beta_integral_fn=config.beta_integral,
        dt=config.dt,
        t0=config.t0, 
        t1=config.t1,
        N=config.N
    )


def get_opt(config: ml_collections.ConfigDict):
    return getattr(optax, config.opt)(config.lr, **config.opt_kwargs)


def main():
    """
        Fit a score-based diffusion model.
    """

    datasets_path = "/project/ls-gruen/users/jed.homer/"
    root_dir = "./"

    config = [
        configs.mnist_config(),
        configs.grfs_config(),
        configs.flowers_config(),
        configs.cifar10_config(),
        configs.quijote_config()
    ][-1]

    key = jr.key(config.seed)
    data_key, model_key, train_key = jr.split(key, 3)

    dataset = get_dataset(
        datasets_path, data_key, config
    )
    sharding = sbgm.shard.get_sharding()
    reload_opt_state = False # Restart training or not
        
    # Diffusion model 
    model = get_model(
        model_key, 
        config.model.model_type, 
        dataset.data_shape, 
        dataset.context_shape, 
        dataset.parameter_dim,
        config
    )

    # Stochastic differential equation (SDE)
    sde = get_sde(config.sde)

    # Fit model to dataset
    model = sbgm.train.train(
        train_key,
        model,
        sde,
        dataset,
        config,
        reload_opt_state=reload_opt_state,
        sharding=sharding,
        save_dir=root_dir
    )


if __name__ == "__main__":
    main()