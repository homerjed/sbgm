import os
import cloudpickle
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import einops
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_opt(config: ConfigDict):
    return getattr(optax, config.opt)(config.lr, **config.opt_kwargs)


def load_model(model, filename):
    model = eqx.tree_deserialise_leaves(filename, model)
    return model


def save_model(model, filename):
    eqx.tree_serialise_leaves(filename, model)


def save_opt_state(opt, opt_state, i, filename="state.obj"):
    """ Save an optimiser and its state for a model, to train later """
    state = {
        "opt" : opt, "opt_state" : opt_state, "step" : i
    }
    f = open(filename, 'wb')
    cloudpickle.dump(state, f)
    f.close()


def load_opt_state(filename="state.obj"):
    f = open(filename, 'rb')
    state = cloudpickle.load(f)
    f.close()
    return state


def count_params(model):
    return np.log10(
        sum(
            x.size for x in jax.tree_util.tree_leaves(model) 
            if eqx.is_array(x)
        )
    )


def plot_model_sample(eu_sample, ode_sample, dataset, config, filename):

    def plot_sample(samples, mode):
        fig, ax = plt.subplots(dpi=300)
        samples_onto_ax(samples, fig, ax, vs=None, cmap=config.cmap)
        plt.savefig(filename + "_" + mode, bbox_inches="tight")
        plt.close()

    def rescale(sample):
        # Input data [0,1] was normed to [-1, 1]
        sample = dataset.scaler.reverse(sample) # [-1,1] -> [0, 1]
        sample = jnp.clip(sample, 0., 1.) 
        return sample

    # EU sampling
    if eu_sample is not None:
        eu_sample = rescale(eu_sample)
        plot_sample(eu_sample, mode="eu")

    # ODE sampling
    if ode_sample is not None:
        ode_sample = rescale(ode_sample)
        plot_sample(ode_sample, mode="ode")


def plot_train_sample(dataset, sample_size, vs, cmap, filename):
    # Unscale data from dataloader (ignoring parameters)
    X, Q, A = next(dataset.train_dataloader.loop(sample_size ** 2))
    print("batch X", X.min(), X.max())
    X = dataset.scaler.reverse(X)[:sample_size ** 2]
    print("batch X (scaled)", X.min(), X.max())

    fig, ax = plt.subplots(dpi=300)
    samples_onto_ax(X, fig, ax, vs, cmap)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    del X, Q


def plot_sde(sde, filename):
    # Plot SDE with time
    plt.figure()
    T = jnp.linspace(sde.t0, sde.t1, 1000)
    mu, std = jax.vmap(sde.marginal_prob)(jnp.ones_like(T), T)
    plt.title(str(sde.__class__.__name__))
    plt.plot(T, mu, label=r"$\mu(t)$")
    plt.plot(T, std, label=r"$\sigma(t)$")
    plt.xlabel(r"$t$")
    plt.savefig(filename)
    plt.close()


def make_dirs(root_dir, config):
    # Make experiment and image save directories
    img_dir = os.path.join(root_dir, "exps/", config.dataset_name + "/") 
    exp_dir = os.path.join(root_dir, "imgs/", config.dataset_name + "/") 
    for _dir in [img_dir, exp_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)
    return exp_dir, img_dir


def imgs_to_grid(X):
    """ Arrange images to one grid image """
    # Assumes square number of imgs
    N, c, h, w = X.shape
    n = int(np.sqrt(N))
    X_grid = einops.rearrange(
        X, 
        "(n1 n2) c h w -> (n1 h) (n2 w) c", 
        c=c,
        n1=n, 
        n2=n, 
    )
    return X_grid


def _add_spacing(img, img_size):
    """ Add whitespace between images on a grid """
    # Assuming channels added from `imgs_to_grid`, and square imgs
    h, w, c = img.shape
    idx = jnp.arange(img_size, h, img_size)
    # NaNs not registered by colormaps?
    img_  = jnp.insert(img, idx, jnp.nan, axis=0)
    img_  = jnp.insert(img_, idx, jnp.nan, axis=1)
    return img_


def samples_onto_ax(_X, fig, ax, vs, cmap):
    """ Drop a sample _X onto an ax by gridding it first """
    _, c, img_size, _ = _X.shape
    im = ax.imshow(
        _add_spacing(imgs_to_grid(_X), img_size), 
        # **vs, # 'vs' is dict of imshow vmin and vmax
        cmap=cmap
    )
    ax.axis("off")
    # If only one channel, use colorbar
    if c == 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
    else:
        pass


def plot_metrics(train_losses, valid_losses, step, exp_dir):
    if step != 0:
        fig, ax = plt.subplots(1, 1, figsize=(8., 4.))
        ax.loglog(train_losses)
        ax.loglog(valid_losses)
        plt.savefig(os.path.join(exp_dir, "loss.png"))
        plt.close()