import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import PyTree, Key, Array
from ml_collections import ConfigDict
import optax
from tqdm import trange

from .sde import SDE
from ._sample import get_eu_sample_fn, get_ode_sample_fn
from ._shard import shard_batch
from ._misc import (
    make_dirs, 
    plot_sde, 
    plot_train_sample, 
    plot_model_sample, 
    plot_metrics,
    load_model, 
    load_opt_state, 
    save_model, 
    save_opt_state
)

Model = eqx.Module
OptState = optax.OptState
TransformUpdateFn = optax.TransformUpdateFn


def apply_ema(
    ema_model: Model, 
    model: Model, 
    ema_rate: float = 0.999
) -> Model:
    # Break models into parameters and 'architecture'
    m_, _m = eqx.partition(model, eqx.is_inexact_array)
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array)
    # Calculate EMA parameters
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    e_ = jtu.tree_map(ema_fn, e_, m_)
    # Combine EMA model parameters and architecture
    return eqx.combine(e_, _m)


def single_loss_fn(
    model: Model, 
    sde: SDE,
    x: Array, 
    q: Array, 
    a: Array, 
    t: Array, 
    key: Key
) -> Array:
    key_noise, key_apply = jr.split(key)
    mean, std = sde.marginal_prob(x, t) # std = jnp.sqrt(jnp.maximum(std, 1e-5)) 
    noise = jr.normal(key_noise, x.shape)
    y = mean + std * noise
    y_ = model(t, y, q=q, a=a, key=key_apply) # Inference is true in validation
    return sde.weight(t) * jnp.square(y_ + noise / std).mean()


def sample_time(
    key: Key, 
    t0: float, 
    t1: float, 
    n_sample: int
) -> Array:
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(key, (n_sample,), minval=t0, maxval=t1 / n_sample)
    t = t + (t1 / n_sample) * jnp.arange(n_sample)
    return t


@eqx.filter_jit
def batch_loss_fn(
    model: Model, 
    sde: SDE,
    x: Array, 
    q: Array, 
    a: Array,
    key: Key
) -> Array:
    batch_size = x.shape[0]
    key_t, key_L = jr.split(key)
    keys_L = jr.split(key_L, batch_size)
    t = sample_time(key_t, sde.t0, sde.t1, batch_size)
    loss_fn = jax.vmap(partial(single_loss_fn, model, sde))
    return loss_fn(x, q, a, t, keys_L).mean()


@eqx.filter_jit
def make_step(
    model: Model, 
    sde: SDE,
    x: Array, 
    q: Array, 
    a: Array, 
    key: Key, 
    opt_state: OptState, 
    opt_update: TransformUpdateFn
) -> Tuple[Array, Model, Key, OptState]:
    model = eqx.nn.inference_mode(model, False)
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, sde, x, q, a, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key, _ = jr.split(key)
    return loss, model, key, opt_state


@eqx.filter_jit
def evaluate(
    model: Model, 
    sde: SDE, 
    x: Array, 
    q: Array, 
    a: Array, 
    key: Key
) -> Array:
    model = eqx.nn.inference_mode(model, True)
    loss = batch_loss_fn(model, sde, x, q, a, key)
    return loss 


def get_opt(config: ConfigDict):
    return getattr(optax, config.opt)(config.lr, **config.opt_kwargs)


def train(
    key: Key, 
    # Diffusion model and SDE
    model: eqx.Module, 
    sde: SDE,
    # Dataset
    dataset: dataclass,
    # Experiment config
    config: ConfigDict,
    # Reload optimiser or not
    reload_opt_state: bool = False,
    # Sharding of devices to run on
    sharding: Optional[jax.sharding.Sharding] = None,
    # Location to save model, figs, .etc in
    save_dir: Optional[str] = None,
    plot_train_data: bool = False
):
    """
        Trains a diffusion model built from a score network (`model`) using a stochastic 
        differential equation (SDE, `sde`) with a given dataset, with support for various 
        configurations and saving options.

        Parameters:
        -----------
        `key` : `Key`
            A JAX random key used for sampling and model initialization.
        
        `model` : `eqx.Module`
            The model to be trained, which is typically a `UNet`, `ResidualNetwork` or custom module.
        
        `sde` : `SDE`
            The SDE that governs the diffusion process. Defines forward and reverse dynamics.
        
        `dataset` : `dataclass`
            The dataset to train on which contains the data loaders.
        
        `config` : `ConfigDict`
            Experiment configuration settings, such as model parameters, training steps, batch size, and SDE specifics.
        
        `reload_opt_state` : `bool`, default: `False`
            Whether to reload the optimizer state and model from previous checkpoint files. Defaults to starting from scratch.
        
        `sharding` : `Optional[jax.sharding.Sharding]`, default: `None`
            Optional device sharding for distributed training across multiple devices.
        
        `save_dir` : `Optional[str]`, default: `None`
            Directory path to save the model, optimizer state, and training figures. If `None`, a default directory is created.
        
        `plot_train_data` : `bool`, default: `False`
            If `True`, plots a sample of the training data at the start of training.
        
        Returns:
        --------
        `model` : `eqx.Module`
            The trained model after the specified number of training steps.
        
        Notes:
        ------
        - The function supports optional early stopping and evaluation using exponential 
          moving averages (EMA) of the model.
        - It saves the model and optimizer state at regular intervals, as well as plots 
          training metrics like losses and sampled outputs.
        - This function handles both EU (Euler-Maruyama) and ODE sampling methods, 
          depending on the config settings.
        - The function can reload previously saved optimizer state and continue 
          training from where it left off.
    """

    print(f"Training SGM with {config.sde.sde} SDE on {config.dataset_name} dataset.")

    # Experiment and image save directories
    exp_dir, img_dir = make_dirs(save_dir, config)

    # Model and optimiser save filenames
    model_filename = os.path.join(
        exp_dir, f"sgm_{dataset.name}_{config.model.model_type}.eqx"
    )
    state_filename = os.path.join(
        exp_dir, f"state_{dataset.name}_{config.model.model_type}.obj"
    )

    # Plot SDE over time 
    plot_sde(sde, filename=os.path.join(exp_dir, "sde.png"))

    # Plot a sample of training data
    if plot_train_data:
        plot_train_sample(
            dataset, 
            sample_size=config.sample_size,
            cmap=config.cmap,
            vs=None,
            filename=os.path.join(img_dir, "data.png")
        )

    # Reload optimiser and state if so desired
    opt = get_opt(config)
    if not reload_opt_state:
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        start_step = 0
    else:
        state = load_opt_state(filename=state_filename)
        model = load_model(model, model_filename)

        opt, opt_state, start_step = state.values()

        print("Loaded model and optimiser state.")

    train_key, sample_key, valid_key = jr.split(key, 3)

    train_total_value = 0
    valid_total_value = 0
    train_total_size = 0
    valid_total_size = 0
    train_losses = []
    valid_losses = []

    if config.use_ema:
        ema_model = deepcopy(model)

    with trange(start_step, config.n_steps, colour="red") as steps:
        for step, train_batch, valid_batch in zip(
            steps, 
            dataset.train_dataloader.loop(config.batch_size), 
            dataset.valid_dataloader.loop(config.batch_size)
        ):
            # Train
            x, q, a = shard_batch(train_batch, sharding)
            _Lt, model, train_key, opt_state = make_step(
                model, sde, x, q, a, train_key, opt_state, opt.update
            )

            train_total_value += _Lt.item()
            train_total_size += 1
            train_losses.append(train_total_value / train_total_size)

            if config.use_ema:
                ema_model = apply_ema(ema_model, model)

            # Validate
            x, q, a = shard_batch(valid_batch, sharding)
            _Lv = evaluate(
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
                Q, A = dataset.label_fn(key_Q, config.sample_size ** 2)

                # EU sampling
                if config.eu_sample:
                    sample_fn = get_eu_sample_fn(
                        ema_model if config.use_ema else model, sde, dataset.data_shape
                    )
                    eu_sample = jax.vmap(sample_fn)(sample_keys, Q, A)

                # ODE sampling
                if config.ode_sample:
                    sample_fn = get_ode_sample_fn(
                        ema_model if config.use_ema else model, sde, dataset.data_shape
                    )
                    ode_sample = jax.vmap(sample_fn)(sample_keys, Q, A)

                # Sample images and plot
                if config.eu_sample or config.ode_sample:
                    plot_model_sample(
                        eu_sample,
                        ode_sample,
                        dataset,
                        config,
                        filename=os.path.join(img_dir, f"samples_{step:06d}"),
                    )

                # Save model
                save_model(
                    ema_model if config.use_ema else model, model_filename
                )

                # Save optimiser state
                save_opt_state(
                    opt, 
                    opt_state, 
                    i=step, 
                    filename=state_filename
                )

                # Plot losses etc
                plot_metrics(train_losses, valid_losses, step, exp_dir)

    return model