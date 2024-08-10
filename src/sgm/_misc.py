import os
import pickle
import cloudpickle
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import einops
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def count_params(model: eqx.Module) -> int:
    return np.log10(
        sum(
            x.size for x in jax.tree_util.tree_leaves(model) 
            if eqx.is_array(x)
        )
    )


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


def plot_metrics(train_losses, valid_losses, dets, step, exp_dir):
    if step != 0:
        fig, ax = plt.subplots(1, 1, figsize=(8., 4.))
        ax.loglog(train_losses)
        ax.loglog(valid_losses)
        plt.savefig(os.path.join(exp_dir, "loss.png"))
        plt.close()
    
    if dets is not None:
        plt.figure()
        plt.semilogy(dets)
        plt.savefig(os.path.join(exp_dir, "dets.png"))
        plt.close()


def save_opt_state(opt, opt_state, i, filename="state.obj"):
    """ Save an optimiser and its state for a model, to train later """
    state = {"opt" : opt, "opt_state" : opt_state, "step" : i}
    f = open(filename, 'wb')
    cloudpickle.dump(state, f)
    f.close()


def load_opt_state(filename="state.obj"):
    f = open(filename, 'rb')
    state = cloudpickle.load(f)
    f.close()
    return state