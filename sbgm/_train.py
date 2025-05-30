import os
from typing import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array, Float, PyTree, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict
import optax
from tqdm.auto import trange

from .sde import SDE
from ._sample import get_eu_sample_fn, get_ode_sample_fn
from ._misc import (
    make_dirs, 
    plot_sde, 
    plot_train_sample, 
    plot_model_sample, 
    plot_metrics,
    save_model, 
    save_opt_state,
    load_model, 
    load_opt_state 
)

Model = eqx.Module

"""
    Trainer functions for training diffusion models using stochastic differential equations (SDEs).
    - Train from an `ml_collections.ConfigDict` config object or from keyword arguments directly.
    - Many bells and whistles such as:
        - Exponential Moving Average (EMA) of model parameters,
        - Sampling and saving model outputs at regular intervals,
        - Support for accumulating gradients over minibatches for large datasets / datavectors,
        - Sharding of model and data across multiple devices for distributed training.
"""


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
    e_ = jax.tree.map(ema_fn, e_, m_)
    # Combine EMA model parameters and architecture
    return eqx.combine(e_, _m)


def accumulate_gradients_scan(
    model: Model,
    sde: SDE,
    xqat: Tuple[
        Float[Array, "b ..."], 
        Optional[Float[Array, "b ..."]], 
        Optional[Float[Array, "b ..."]], 
        Float[Array, "b"]
    ],
    key: Key,
    n_minibatches: int,
    *,
    grad_fn: Callable
) -> Tuple[Float[Array, ""], PyTree]:
    batch_size = xqat[0].shape[0]
    minibatch_size = batch_size // n_minibatches

    keys = jr.split(key, n_minibatches)

    def _minibatch_step(minibatch_idx):
        """ Gradients and metrics for a single minibatch. """
        _xqat = jax.tree.map(
            # Slicing with variable index (jax.Array).
            lambda x: jax.lax.dynamic_slice_in_dim(  
                x, 
                start_index=minibatch_idx * minibatch_size, 
                slice_size=minibatch_size, 
                axis=0
            ),
            xqat, # This works for tuples of batched data e.g. (x, q, a)
        )
        L, step_grads = grad_fn(
            model, sde, *_xqat, keys[minibatch_idx]
        )
        return step_grads, L

    def _scan_step(carry, minibatch_idx):
        """ Scan step function for looping over minibatches. """
        step_grads, L = _minibatch_step(minibatch_idx)
        carry = jax.tree.map(jnp.add, carry, (step_grads, L))
        return carry, None

    # Determine initial shapes for gradients and metrics.
    grads_shapes, L_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    L = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), L_shape)

    # Loop over minibatches to determine gradients and metrics.
    (grads, L), _ = jax.lax.scan(
        _scan_step, 
        init=(grads, L), 
        xs=jnp.arange(n_minibatches), 
        length=n_minibatches
    )

    # Average gradients over minibatches.
    grads = jax.tree.map(lambda g: g / n_minibatches, grads)
    L = jax.tree.map(lambda m: m / n_minibatches, L)

    return L, grads # Same signature as unaccumulated 


def single_loss_fn(
    model: Model, 
    sde: SDE,
    x: Float[Array, "..."], 
    q: Optional[Float[Array, "..."]], 
    a: Optional[Float[Array, "..."]],
    t: Float[Array, ""],
    key: Key
) -> Float[Array, ""]:
    key_noise, key_apply = jr.split(key)
    mean, std = sde.marginal_prob(x, t) 
    noise = jr.normal(key_noise, x.shape)
    y = mean + std * noise
    y_ = model(t, y, q=q, a=a, key=key_apply) # Inference is true in validation
    return sde.weight(t) * jnp.mean(jnp.square(y_ + noise / std))


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


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def batch_loss_fn(
    model: Model, 
    sde: SDE,
    x: Float[Array, "b ..."], 
    q: Optional[Float[Array, "b ..."]], 
    a: Optional[Float[Array, "b ..."]],
    t: Float[Array, "b"],
    key: Key[jnp.ndarray, "..."]
) -> Array:
    batch_size = x.shape[0]
    keys_L = jr.split(key, batch_size)
    loss_fn = jax.vmap(partial(single_loss_fn, model, sde))
    return loss_fn(x, q, a, t, keys_L).mean()


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def make_step(
    model: Model, 
    sde: SDE,
    xqa: Tuple[
        Float[Array, "b ..."], 
        Optional[Float[Array, "b ..."]], 
        Optional[Float[Array, "b ..."]], 
    ],
    key: Key[jnp.ndarray, "..."], 
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation, 
    *,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> Tuple[
    Float[Array, ""], Model, Key[jnp.ndarray, "..."], optax.OptState
]:
    model = eqx.nn.inference_mode(model, False)

    key_apply, key_t = jr.split(key)

    grad_fn = eqx.filter_value_and_grad(batch_loss_fn)

    if replicated_sharding:
        model, opt_state = eqx.filter_shard((model, opt_state), replicated_sharding)
    if sharding:
        xqa = eqx.filter_shard(xqa, sharding)

    x, q, a = xqa
    t = sample_time(key_t, sde.t0, sde.t1, x.shape[0])

    if accumulate_gradients:
        loss, grads = accumulate_gradients_scan(
            model, 
            sde,
            (x, q, a, t), 
            key_apply, 
            n_minibatches=n_minibatches, 
            grad_fn=grad_fn
        ) 
    else:
        loss, grads = grad_fn(model, sde, x, q, a, t, key_apply)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    key, _ = jr.split(key)
    return loss, model, key, opt_state


@eqx.filter_jit
def evaluate(
    model: Model, 
    sde: SDE, 
    xqa: Tuple[
        Float[Array, "b ..."], 
        Optional[Float[Array, "b ..."]], 
        Optional[Float[Array, "b ..."]], 
    ],
    key: Key[jnp.ndarray, "..."],
    *,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> Array:
    model = eqx.nn.inference_mode(model, True)

    key_apply, key_t = jr.split(key)

    if replicated_sharding:
        model = eqx.filter_shard(model, replicated_sharding)
    if sharding:
        x, q, a = eqx.filter_shard(xqa, sharding)

    x, q, a = xqa

    t = sample_time(key_t, sde.t0, sde.t1, x.shape[0])

    loss = batch_loss_fn(model, sde, x, q, a, t, key_apply)
    return loss 


def get_opt(config: ConfigDict) -> optax.GradientTransformation:
    return getattr(optax, config.opt)(config.lr, **config.opt_kwargs)


def train_from_config(
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
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
    *,
    # Location to save model, figs, .etc in
    save_dir: Optional[str] = None,
    plot_train_data: bool = False
) -> Model:
    """
        Trains a diffusion model built from a score network (`model`) using a stochastic 
        differential equation (SDE, `sde`) with a given dataset. Requires a config object.

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
        
        `sharding` : `Optional[jax.sharding.NamedSharding]`, default: `None`
            Optional device sharding for distributed training across multiple devices. Shards sections of batches across each device.

        `replicated_sharding` : `Optional[jax.sharding.NamedSharding]`, default: `None`
            Optional device sharding for distributed training across multiple devices. Shards all model arrays across each device.
        
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

    print(f"Training SGM with a {config.sde.sde} SDE on {config.dataset_name} dataset.")

    # Experiment and image save directories
    exp_dir, img_dir = make_dirs(save_dir, config.dataset_name)

    # Model and optimiser save filenames
    model_filename = os.path.join(
        exp_dir, f"{dataset.name}_{config.model.model_type}.eqx"
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
    if reload_opt_state:
        state = load_opt_state(filename=state_filename)

        opt, opt_state, start_step = state.values()
        model = load_model(model, model_filename)

        print("Loaded model and optimiser state.")
    else:
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        start_step = 0

    train_key, sample_key, valid_key = jr.split(key, 3)

    train_total_value = 0
    valid_total_value = 0
    total_size = 0
    train_losses = []
    valid_losses = []

    if config.use_ema:
        ema_model = deepcopy(model)

    if replicated_sharding:
        ema_model = eqx.filter_shard(ema_model, replicated_sharding)

    with trange(start_step, config.n_steps, colour="red") as steps:
        for step, train_batch, valid_batch in zip(
            steps, 
            dataset.train_dataloader.loop(config.batch_size), 
            dataset.valid_dataloader.loop(config.batch_size)
        ):
            # Train
            _Lt, model, train_key, opt_state = make_step(
                model, 
                sde, 
                train_batch,
                train_key, 
                opt_state, 
                opt,
                accumulate_gradients=config.accumulate_gradients,
                n_minibatches=config.n_minibatches,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            train_total_value += _Lt.item()
            total_size += 1
            train_losses.append(train_total_value / total_size)

            if config.use_ema:
                ema_model = apply_ema(ema_model, model)

            # Validate
            _Lv = evaluate(
                ema_model if config.use_ema else model, 
                sde, 
                valid_batch,
                valid_key,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            valid_total_value += _Lv.item()
            valid_losses.append(valid_total_value / total_size)

            steps.set_postfix(
                {"Lt" : f"{train_losses[-1]:.3E}", "Lv" : f"{valid_losses[-1]:.3E}"}
            )

            if (
                ((step % config.sample_and_save_every) == 0) 
                or 
                (step == config.n_steps - 1)
                or 
                (step == 100)
            ):
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
                        cmap=config.cmap,
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

    return ema_model if config.use_ema else model


def train(
    key: Key, 
    # Diffusion model and SDE
    model: eqx.Module, 
    sde: SDE,
    # Dataset
    dataset: dataclass,
    # Training
    opt: optax.GradientTransformation,
    n_steps: int,
    batch_size: int,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    use_ema: bool = True,
    sample_and_save_every: int = 1_000,
    # Sampling
    sample_size: int = 1,
    eu_sample: bool = False,
    ode_sample: bool = False,
    # Reload optimiser or not
    reload_opt_state: bool = False,
    # Sharding of devices to run on
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
    *,
    # Location to save model, figs, .etc in
    save_dir: Optional[str] = None,
    # Plotting
    plot_train_data: bool = False,
    cmap: str = "gray_r"
) -> Model:
    """
        Trains a diffusion model using a stochastic differential equation (SDE) based on 
        the provided score network model and dataset, with support for optimizer state reloading, 
        Exponential Moving Average (EMA), and periodic model sampling and evaluation.

        Parameters:
        -----------
        `key` : `Key`
            A JAX random key for sampling and model initialization.

        `model` : `eqx.Module`
            The score network model to be trained.

        `sde` : `SDE`
            The stochastic differential equation (SDE) defining the forward and reverse diffusion processes.
        
        `dataset` : `dataclass`
            A dataset object containing the data loaders for training and validation.
        
        `opt` : `optax.GradientTransformation`
            The optimizer transformation function (from Optax) for updating model parameters.
        
        `n_steps` : `int`
            The total number of training steps.
        
        `batch_size` : `int`
            The size of the mini-batches to be used for each training step.
        
        `use_ema` : `bool`, default: `True`
            Whether to use Exponential Moving Average (EMA) of model parameters for validation and sampling.
        
        `sample_and_save_every` : `int`, default: `1_000`
            The frequency in training steps in which the model is saved and sampled.
        
        `sample_size` : `int`, default: `None`
            Number of samples to generate during the sampling phase at each logging step.

        `reload_opt_state` : `bool`, default: `False`
            Whether to reload the model and optimizer state from a previous checkpoint to continue training.
        
        `sharding` : `Optional[jax.sharding.NamedSharding]`, default: `None`
            Optional device sharding for distributed training across multiple devices. Shards sections of batches across each device.

        `replicated_sharding` : `Optional[jax.sharding.NamedSharding]`, default: `None`
            Optional device sharding for distributed training across multiple devices. Shards all model arrays across each device.

        `save_dir` : `Optional[str]`, default: `None`
            Directory path to save the model, optimizer state, and training logs. If `None`, a default path is generated.
        
        `plot_train_data` : `bool`, default: `False`
            If `True`, plots a sample of the training data at the beginning of training.
        
        `cmap` : `str`, default: `"gray_r"`
            The colormap to be used for plotting sampled data. Ignored for non-image data.

        Returns:
        --------
        `model` : `eqx.Module`
            The trained model after completing the specified number of training steps.

        Notes:
        ------
        - The function trains a model using a diffusion process governed by an SDE and saves checkpoints 
          and intermediate samples during the training process.
        - Supports both Euler-Maruyama (EU) and ODE sampling techniques depending on the type of diffusion process.
        - If `use_ema` is enabled, EMA of the model parameters is applied for better stability and performance.
        - Supports restarting the training process by reloading the optimizer state and model from previously saved checkpoints.
    """

    print(f"Training SGM with a {sde.__class__.__name__} on {dataset.name} dataset.")

    # Experiment and image save directories
    exp_dir, img_dir = make_dirs(save_dir, dataset.name) # Dataset name from config

    # Model and optimiser save filenames
    model_type = model.__class__.__name__
    model_filename = os.path.join(
        exp_dir, f"{dataset.name}_{model_type}.eqx"
    )
    state_filename = os.path.join(
        exp_dir, f"state_{dataset.name}_{model_type}.obj"
    )

    # Plot SDE over time 
    plot_sde(sde, filename=os.path.join(exp_dir, "sde.png"))

    # Plot a sample of training data
    if plot_train_data:
        plot_train_sample(
            dataset, 
            sample_size=sample_size,
            cmap=cmap,
            vs=None,
            filename=os.path.join(img_dir, "data.png")
        )

    # Reload optimiser and state if so desired
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
    total_size = 0
    train_losses = []
    valid_losses = []

    if use_ema:
        ema_model = deepcopy(model)

    if replicated_sharding:
        ema_model = eqx.filter_shard(ema_model, replicated_sharding)

    with trange(start_step, n_steps, colour="red") as steps:
        for step, train_batch, valid_batch in zip(
            steps, 
            dataset.train_dataloader.loop(
                n_minibatches * batch_size if accumulate_gradients else batch_size
            ), 
            dataset.valid_dataloader.loop(batch_size)
        ):
            # Train
            _Lt, model, train_key, opt_state = make_step(
                model, 
                sde, 
                train_batch,
                train_key, 
                opt_state, 
                opt,
                accumulate_gradients=accumulate_gradients,
                n_minibatches=n_minibatches,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            train_total_value += _Lt.item()
            total_size += 1
            train_losses.append(train_total_value / total_size)

            if use_ema:
                ema_model = apply_ema(ema_model, model)

            # Validate
            _Lv = evaluate(
                ema_model if use_ema else model, 
                sde, 
                valid_batch,
                valid_key,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            valid_total_value += _Lv.item()
            valid_losses.append(valid_total_value / total_size)

            steps.set_postfix(
                {"Lt" : f"{train_losses[-1]:.3E}", "Lv" : f"{valid_losses[-1]:.3E}"}
            )

            if (
                ((step % sample_and_save_every) == 0) 
                or 
                (step == n_steps - 1)
                or 
                (step == 100)
            ):                
                # Sample model
                key_Q, key_sample = jr.split(sample_key) # Fixed key
                sample_keys = jr.split(key_sample, sample_size ** 2)

                # Sample random labels or use parameter prior for labels
                Q, A = dataset.label_fn(key_Q, sample_size ** 2)

                # EU sampling
                if eu_sample:
                    sample_fn = get_eu_sample_fn(
                        ema_model if use_ema else model, sde, dataset.data_shape
                    )
                    eu_samples = jax.vmap(sample_fn)(sample_keys, Q, A)
                else:
                    eu_samples = None

                # ODE sampling
                if ode_sample:
                    sample_fn = get_ode_sample_fn(
                        ema_model if use_ema else model, sde, dataset.data_shape
                    )
                    ode_samples = jax.vmap(sample_fn)(sample_keys, Q, A)
                else:
                    ode_samples = None

                # Sample images and plot
                if eu_sample or ode_sample:
                    plot_model_sample(
                        eu_samples,
                        ode_samples,
                        dataset,
                        cmap=cmap,
                        filename=os.path.join(img_dir, f"samples_{step:06d}"),
                    )

                # Save model
                save_model(
                    ema_model if use_ema else model, model_filename
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

    return ema_model if use_ema else model