import os
from copy import deepcopy
from typing import Sequence, Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx
from jaxtyping import Key, Array
import optax
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange

import sgm
import data
import configs
from sgm._misc import samples_onto_ax, plot_metrics, save_opt_state, load_opt_state


def get_sharding():
    n_devices = len(jax.local_devices())
    print(f"Running on {n_devices} local devices: \n\t{jax.local_devices()}")

    use_sharding = n_devices > 1
    # Sharding mesh: speed and allow training on high resolution?
    if use_sharding:
        # Split array evenly across data dimensions, this reshapes automatically
        mesh = Mesh(jax.devices(), ('x',))
        sharding = NamedSharding(mesh, P('x'))
    else:
        sharding = None
    print(f"Sharding:\n {sharding}")
    return sharding


def shard_batch(
    batch: Tuple[Array, Array], 
    sharding: Optional[NamedSharding] = None
) -> Tuple[Array, Array]:
    if sharding:
        # batch = jax.device_put(batch, sharding)
        batch = eqx.filter_shard(batch, sharding)
    return batch


def count_params(model: eqx.Module) -> int:
    return np.log10(
        sum(
            x.size for x in jax.tree_util.tree_leaves(model) 
            if eqx.is_array(x)
        )
    )


def sample_and_plot(
    # Sampling
    key: Key, 
    # Data
    dataset: data.ScalerDataset,
    # Config
    config: configs.Config,
    # Model and SDE
    model: eqx.Module,
    sde: sgm.sde.SDE,
    # Plotting
    filename: str
):
    def plot_sample(samples, Q, mode, dataset_name):
        fig, ax = plt.subplots(dpi=300)
        if dataset_name != "moons":
            samples_onto_ax(samples, fig, ax, vs=None, cmap=config.cmap)
        else:
            ax.scatter(*samples.T, c=Q, cmap=config.cmap)
        plt.savefig(filename + "_" + mode, bbox_inches="tight")
        plt.close()

    def rescale(sample):
        # data [0,1] was normed to [-1, 1]
        sample = dataset.scaler.reverse(sample) # [-1,1] -> [0, 1]
        if dataset.name != "moons":
            sample = jnp.clip(sample, 0., 1.) 
        return sample

    key_Q, key_sample = jr.split(key)
    sample_keys = jr.split(key_sample, config.sample_size ** 2)

    # Sample random labels or use parameter prior for labels
    Q = data.get_labels(key_Q, dataset, config)

    # EU sampling
    if config.eu_sample:
        sample_fn = sgm.sample.get_eu_sample_fn(model, sde, dataset.data_shape)
        sample = jax.vmap(sample_fn)(sample_keys, Q)
        sample = rescale(sample)
        plot_sample(sample, Q, mode="eu", dataset_name=dataset.name)

    # ODE sampling
    if config.ode_sample:
        sample_fn = sgm.sample.get_ode_sample_fn(model, sde, dataset.data_shape)
        sample = jax.vmap(sample_fn)(sample_keys, Q)
        sample = rescale(sample)
        plot_sample(sample, Q, mode="ode", dataset_name=dataset.name)


def plot_train_sample(dataset, sample_size, vs, cmap, filename):
    fig, ax = plt.subplots(dpi=300)

    # Unscale data from dataloader
    X, Q = next(dataset.train_dataloader.loop(sample_size ** 2))
    print("batch X", X.min(), X.max())
    X = dataset.scaler.reverse(X)[:sample_size ** 2]
    print("batch X (scaled)", X.min(), X.max())

    if dataset.name != "moons":
        samples_onto_ax(X, fig, ax, vs, cmap)
    else: 
        ax.scatter(*X.T, c=Q, cmap="PiYG")
    del X, Q
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def make_dirs(root_dir: str, config: configs.Config) -> Tuple[str, str]:
    img_dir = os.path.join(root_dir, "exps/", config.dataset_name + "/") 
    exp_dir = os.path.join(root_dir, "imgs/", config.dataset_name + "/") 
    for _dir in [img_dir, exp_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)
    return exp_dir, img_dir


def plot_sde(sde, filename):
    # Plot SDE with time
    plt.figure()
    T = jnp.linspace(sde.t0, sde.t1, 1000)
    mu, std = jax.vmap(sde.marginal_prob)(jnp.ones_like(T), T)
    plt.plot(T, mu)
    plt.plot(T, std)
    plt.savefig(filename)
    plt.close()


def p_x_q_plot(key, log_prob_fn, filename):
    X, Y = jnp.mgrid[-3.:3.:100j, -2.:2.:100j]
    X = jnp.vstack([X.ravel(), Y.ravel()]).T
    print("X", X.shape)

    # Calculate likelihoods
    Q = jnp.zeros((len(X), 1))
    p_x_0 = jax.vmap(log_prob_fn)(X, Q, jr.split(key, len(X)))
    p_x_0 = jnp.exp(p_x_0) / jnp.exp(p_x_0).sum()
    Q = jnp.ones((len(X), 1))
    p_x_1 = jax.vmap(log_prob_fn)(X, Q, jr.split(key, len(X)))
    p_x_1 = jnp.exp(p_x_1) / jnp.exp(p_x_1).sum()
    print("like", p_x_0.shape)

    fig, axs = plt.subplots(1, 2, figsize=(13., 6.), dpi=200)
    ax = axs[0]
    ax.set_title("0")
    s = ax.scatter(*X.T, c=p_x_0, cmap="PuOr")
    fig.colorbar(s)
    ax = axs[1]
    ax.set_title("1")
    s = ax.scatter(*X.T, c=p_x_1, cmap="PuOr")
    fig.colorbar(s)
    plt.savefig(filename)
    plt.close()


def load_model(model: eqx.Module, filename: str) -> eqx.Module:
    model = eqx.tree_deserialise_leaves(filename, model)
    return model


def save_model(model: eqx.Module, filename: str) -> None:
    eqx.tree_serialise_leaves(filename, model)


def get_model(
    model_key: Key, 
    model_type: str, 
    data_shape: Sequence[int], 
    context_dim: Sequence[int], 
    config: configs.Config
) -> eqx.Module:
    if model_type == "Mixer":
        model = sgm.models.Mixer2d(
            data_shape,
            **config.model_args,
            t1=config.t1,
            context_dim=context_dim,
            key=model_key
        )
    if model_type == "UNet":
        model = sgm.models.UNet(
            data_shape=data_shape,
            **config.model_args,
            condition_dim=context_dim,
            key=model_key
        )
    if model_type == "mlp":
        model = sgm.models.ResidualNetwork(
            in_size=np.prod(data_shape),
            **config.model_args,
            y_dim=context_dim,
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
    return dataset


def get_sde(config: configs.Config) -> sgm.sde.SDE:
    sde = getattr(sgm.sde, config.sde + "SDE")
    return sde(
        beta_integral=config.beta_integral,
        dt=config.dt,
        t0=config.t0, 
        t1=config.t1,
        N=config.N
    )


def main():
    """
        Fit a score-based diffusion model.
    """
    # config = configs.FlowersConfig
    config = configs.MNISTConfig
    # config = configs.CIFAR10Config

    root_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/sgm/"

    key = jr.key(config.seed)
    data_key, model_key, train_key, sample_key, valid_key = jr.split(key, 5)

    # Non-config args
    dataset          = get_dataset(config.dataset_name, data_key, config)
    data_shape       = dataset.data_shape
    context_dim      = np.prod(dataset.context_shape)
    opt              = getattr(optax, config.opt)(config.lr)
    sharding         = get_sharding()
    reload_opt_state = False # Restart training or not

    # Diffusion model 
    model = get_model(
        model_key, 
        config.model_type, 
        data_shape, 
        context_dim, 
        config
    )
    # SDE, get getattr(sgm.sde, config.sde)
    sde = sgm.sde.VPSDE(
        config.beta_integral, 
        dt=config.dt, 
        t0=config.t0, 
        t1=config.t1, 
        N=config.N
    ) 

    # Experiment and image save directories
    exp_dir, img_dir = make_dirs(root_dir, config)

    # Plot SDE over time 
    plot_sde(sde, filename=os.path.join(exp_dir, "sde.png"))

    # Plot a sample of training data
    plot_train_sample(
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

    print("Model n_params =", count_params(model))

    # Reload optimiser and state if so desired
    if not reload_opt_state:
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        start_step = 0
    else:
        state = load_opt_state(filename=state_filename)
        model = load_model(model, model_filename)

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
            dataset.train_dataloader.loop(
                config.batch_size, num_workers=8
            ), 
            dataset.valid_dataloader.loop(
                config.batch_size, num_workers=8
            )
        ):
            # Train
            x, q = shard_batch(train_batch, sharding)
            _Lt, model, train_key, opt_state = sgm.train.make_step(
                model, sde, x, q, train_key, opt_state, opt.update
            )

            train_total_value += _Lt.item()
            train_total_size += 1
            train_losses.append(train_total_value / train_total_size)

            if config.use_ema:
                ema_model = sgm.apply_ema(ema_model, model)

            # Validate
            x, q = shard_batch(valid_batch, sharding)
            _Lv = sgm.train.evaluate(
                ema_model if config.use_ema else model, sde, x, q, valid_key
            )

            # Record
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
                # Sample images and plot
                sample_and_plot(
                    sample_key, 
                    dataset,
                    config,
                    ema_model if config.use_ema else model,
                    sde,
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
                plot_metrics(train_losses, valid_losses, dets, step, exp_dir)

                if config.dataset_name == "moons":
                    p_x_q_fn = sgm.ode.get_log_likelihood_fn(
                        ema_model if config.use_ema else model,
                        sde, 
                        data_shape, 
                        exact_logp=config.exact_logp
                    )
                    p_x_q_plot(
                        sample_key, 
                        log_prob_fn=p_x_q_fn, 
                        filename=os.path.join(img_dir, f"p_x_q_{step}.png")
                    )

                # Calculate likelihoods
                # p_x_q_fn = get_log_likelihood_fn(
                #     model, sde, data_shape, exact_logp=config.exact_logp
                # )
                # p_x_qs = jax.vmap(p_x_q_fn)(x, q, jr.split(key, len(x)))
                # plt.figure()
                # plt.hist(p_x_qs.flatten(), bins=16)
                # plt.savefig("p_x_qs.png")
                # plt.close()


if __name__ == "__main__":
    main()