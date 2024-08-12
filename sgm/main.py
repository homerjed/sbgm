import os
from copy import deepcopy
from typing import Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
import optax
import numpy as np 
from tqdm import trange

import sgm
from sgm import utils 
import data
import configs


def get_model(
    model_key: Key, 
    model_type: str, 
    data_shape: Sequence[int], 
    context_shape: Sequence[int], 
    parameter_dim: int,
    config: configs.Config
) -> eqx.Module:
    if model_type == "Mixer":
        model = sgm.models.Mixer2d(
            data_shape,
            **config.model_args,
            t1=config.t1,
            q_dim=context_shape,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNet":
        model = sgm.models.UNet(
            data_shape=data_shape,
            **config.model_args,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNetXY":
        model = sgm.models.UNetXY(
            data_shape=data_shape,
            **config.model_args,
            q_dim=context_shape[0], # Just grab channel assuming 'q' is a map like x
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "mlp":
        model = sgm.models.ResidualNetwork(
            in_size=np.prod(data_shape),
            **config.model_args,
            y_dim=parameter_dim,
            key=model_key
        )
    if model_type == "DiT":
        raise NotImplementedError
    return model


def get_dataset(
    dataset_name: str, key: Key, config: configs.Config
) -> data.ScalerDataset:
    if dataset_name == "flowers":
        dataset = data.flowers(key, n_pix=config.n_pix)
    if dataset_name == "cifar10":
        dataset = data.cifar10(key)
    if dataset_name == "mnist":
        dataset = data.mnist(key)
    if dataset_name == "moons":
        dataset = data.moons(key)
    if dataset_name == "grfs":
        dataset = data.grfs(key, n_pix=config.n_pix)
    return dataset


def get_sde(config: configs.Config) -> sgm.sde.SDE:
    name = config.sde + "SDE"
    assert name in ["VESDE", "VPSDE", "SubVPSDE"]
    sde = getattr(sgm.sde, name)
    return sde(
        beta_integral_fn=config.beta_integral,
        dt=config.dt,
        t0=config.t0, 
        t1=config.t1,
        N=config.N
    )


def get_opt(
    config: configs.Config
) -> optax.GradientTransformation:
    return getattr(optax, config.opt)(config.lr, **config.opt_kwargs)


def main():
    """
        Fit a score-based diffusion model.
    """
    config = [
        configs.MNISTConfig,
        configs.GRFConfig,
        configs.FlowersConfig,
        configs.CIFAR10Config
    ][0]

    root_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/sgm/"

    key = jr.key(config.seed)
    data_key, model_key, train_key, sample_key, valid_key = jr.split(key, 5)

    # Non-config args
    dataset          = get_dataset(config.dataset_name, data_key, config)
    data_shape       = dataset.data_shape
    context_shape    = dataset.context_shape
    parameter_dim    = dataset.parameter_dim
    opt              = get_opt(config)
    sharding         = sgm.shard.get_sharding()
    reload_opt_state = False # Restart training or not

    # Diffusion model 
    model = get_model(
        model_key, 
        config.model_type, 
        data_shape, 
        context_shape, 
        parameter_dim,
        config
    )
    # SDE, get getattr(sgm.sde, config.sde)
    # sde = get_sde(config)
    sde = sgm.sde.VESDE(
        # config.beta_integral, 
        # lambda t: t ** 2.,
        lambda t: 25. ** t - 1., #jnp.exp(t),
        # weight_fn=lambda t: 1. - jnp.exp(-t),
        dt=config.dt, 
        t0=config.t0, 
        t1=config.t1, 
        N=config.N
    ) 

    # Experiment and image save directories
    exp_dir, img_dir = utils.make_dirs(root_dir, config)

    # Plot SDE over time 
    utils.plot_sde(sde, filename=os.path.join(exp_dir, "sde.png"))

    # Plot a sample of training data
    utils.plot_train_sample(
        dataset, 
        sample_size=config.sample_size,
        cmap=config.cmap,
        vs=None,
        filename=os.path.join(img_dir, "data.png")
    )

    # Model and optimiser save filenames
    model_filename = os.path.join(
        exp_dir, f"sgm_{dataset.name}_{config.model_type}.eqx"
    )
    state_filename = os.path.join(
        exp_dir, f"state_{dataset.name}_{config.model_type}.obj"
    )

    print("Model n_params =", utils.count_params(model))

    # Reload optimiser and state if so desired
    if not reload_opt_state:
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        start_step = 0
    else:
        state = utils.load_opt_state(filename=state_filename)
        model = utils.load_model(model, model_filename)

        opt, opt_state, start_step = state.values()

        print("Loaded model and optimiser state.")

    train_total_value = 0
    valid_total_value = 0
    train_total_size = 0
    valid_total_size = 0
    train_losses = []
    valid_losses = []
    dets = []

    if config.use_ema:
        ema_model = deepcopy(model)

    with trange(start_step, config.n_steps, colour="red") as steps:
        for step, train_batch, valid_batch in zip(
            steps, 
            dataset.train_dataloader.loop(config.batch_size), 
            dataset.valid_dataloader.loop(config.batch_size)
        ):
            # Train
            x, q, a = sgm.shard.shard_batch(train_batch, sharding)
            _Lt, model, train_key, opt_state = sgm.train.make_step(
                model, sde, x, q, a, train_key, opt_state, opt.update
            )

            train_total_value += _Lt.item()
            train_total_size += 1
            train_losses.append(train_total_value / train_total_size)

            if config.use_ema:
                ema_model = sgm.apply_ema(ema_model, model)

            # Validate
            x, q, a = sgm.shard.shard_batch(valid_batch, sharding)
            _Lv = sgm.train.evaluate(
                ema_model if config.use_ema else model, sde, x, q, a, valid_key
            )

            valid_total_value += _Lv.item()
            valid_total_size += 1
            valid_losses.append(valid_total_value / valid_total_size)

            steps.set_postfix(
                {
                    "Lt" : f"{train_losses[-1]:.3E}",
                    "Lv" : f"{valid_losses[-1]:.3E}"
                }
            )

            if (step % config.print_every) == 0 or step == config.n_steps - 1:
                # Sample model
                key_Q, key_sample = jr.split(sample_key) # Fixed key
                sample_keys = jr.split(key_sample, config.sample_size ** 2)

                # Sample random labels or use parameter prior for labels
                Q, A = data.get_labels(key_Q, dataset.name, config)

                # EU sampling
                if config.eu_sample:
                    sample_fn = sgm.sample.get_eu_sample_fn(
                        ema_model if config.use_ema else model, sde, dataset.data_shape
                    )
                    eu_sample = jax.vmap(sample_fn)(sample_keys, Q, A)

                # ODE sampling
                if config.ode_sample:
                    sample_fn = sgm.sample.get_ode_sample_fn(
                        ema_model if config.use_ema else model, sde, dataset.data_shape
                    )
                    ode_sample = jax.vmap(sample_fn)(sample_keys, Q, A)

                # Sample images and plot
                utils.plot_model_sample(
                    eu_sample,
                    ode_sample,
                    dataset,
                    config,
                    filename=os.path.join(img_dir, f"samples_{step:06d}"),
                )

                # Save model
                utils.save_model(
                    ema_model if config.use_ema else model, model_filename
                )

                # Save optimiser state
                utils.save_opt_state(
                    opt, 
                    opt_state, 
                    i=step, 
                    filename=state_filename
                )

                # Plot losses etc
                utils.plot_metrics(train_losses, valid_losses, dets, step, exp_dir)


if __name__ == "__main__":
    main()