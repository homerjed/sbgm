import os
import jax
import jax.numpy as jnp 
import jax.random as jr
import optax
from einops import rearrange 
import matplotlib.pyplot as plt

import sbgm
from data import grfs

# Load (data) and save (model, samples & optimiser state) directories 
datasets_path = "../datasets/"
root_dir = "./"

n_devices = len(jax.devices())

key = jr.key(0)

data_key, model_key, train_key = jr.split(key, 3)

"""
    Configuration
"""

# Data
dataset_name          = "grfs" 
n_pix                 = 64
img_shape             = (1, n_pix, n_pix)
context_shape         = (1, n_pix, n_pix)
parameter_dim         = 2 # Power spectrum amplitude and index

# Model
is_biggan             = False
dim_mults             = [1, 1, 1]
hidden_size           = 128
heads                 = 4
dim_head              = 64
dropout_rate          = 0.3
num_res_blocks        = 2
attn_resolutions      = [8, 16, 32]
final_activation      = None

embed_dim             = 64 
patch_size            = 4 
depth                 = 4 
n_heads               = 4 

# SDE
t1                    = 8.
t0                    = 0.
dt                    = 0.1
beta_integral         = lambda t: t 
weight_fn             = lambda t: 1. - jnp.exp(-beta_integral(t)) 

# Sampling
use_ema               = False
sample_size           = 5 # Squared for a grid
exact_log_prob            = False
ode_sample            = True # Sample the ODE during training
eu_sample             = True # Euler-Maruyama sample the SDE during training

# Optimisation hyperparameters
start_step            = 0
n_steps               = 1_000_000
batch_size            = 10 * n_devices
sample_and_save_every = 1_000
lr                    = 1e-4
opt                   = optax.adamw
opt_kwargs            = {} 

"""
    Dataset
"""

# Dataset object of training data and loaders
dataset = grfs(data_key, n_pix=n_pix, n_fields=10000)

n_plot = 8 # Grid side length in images
cmap = "coolwarm"

# Grab a batch to plot from the dataloader
X, Q, A = next(dataset.train_dataloader.loop(n_plot ** 2))

fig, axs = plt.subplots(1, 2, dpi=150)
ax = axs[0]
ax.imshow(
    rearrange(Q, "(h w) c x y -> (h x) (w y) c", h=n_plot, w=n_plot, x=n_pix, y=n_pix), 
    cmap=cmap
)
ax = axs[1]
ax.imshow(
    rearrange(X, "(h w) c x y -> (h x) (w y) c", h=n_plot, w=n_plot, x=n_pix, y=n_pix), 
    cmap=cmap
)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
# plt.savefig(os.path.join(results_dir, "grfs_data.png"))
plt.close()

"""
    Model and sharding 
"""

# Multiple GPU training if you are so inclined
sharding, replicated_sharding = sbgm.shard.get_shardings()

n_channels, img_size, img_size = img_shape

# Diffusion model 
# model = sbgm.models.UNet(
#     dim=hidden_size,
#     channels=n_channels,
#     dim_mults=dim_mults,
#     attn_heads=heads,
#     attn_dim_head=dim_head,
#     dropout=dropout_rate,
#     learned_sinusoidal_cond=True,
#     random_fourier_features=True,
#     a_dim=parameter_dim,          # Number of parameters in power spectrum model
#     q_channels=context_shape[0],  # Number of channels in conditioning map
#     key=model_key
# )
model = sbgm.models.DiT(
    img_size=img_size,
    channels=n_channels,
    embed_dim=embed_dim,
    patch_size=patch_size,
    depth=depth,
    n_heads=n_heads,
    q_dim=context_shape[0], # Number of channels in conditioning map
    a_dim=parameter_dim,    # Number of parameters in power spectrum model
    key=model_key
)

"""
    SDE
"""

# Stochastic differential equation (SDE)
sde = sbgm.sde.VPSDE(
    beta_integral_fn=beta_integral,
    dt=dt,
    t0=t0, 
    t1=t1,
    weight_fn=weight_fn
)

def diffuse(x, t, eps):
    mu, std = sde.marginal_prob(x, t)
    return mu + std * eps
    
n_side = 10
n_t = 100

imgs = [X[0]]
for i, t in enumerate(jnp.linspace(t0, t1, n_t)):

    eps = jr.normal(jr.fold_in(key, i), img_shape) * 0.05 # Illustration purposes!

    Xt = diffuse(imgs[i], t, eps)

    imgs.append(Xt)

fig, axs = plt.subplots(1, 2, dpi=150)
ax = axs[0]
ax.imshow(imgs[0].transpose(1, 2, 0), cmap=cmap)
ax = axs[1]
ax.imshow(
    rearrange(imgs[1:], "(h w) c x y -> (h x) (w y) c", h=10, w=10, x=n_pix, y=n_pix), 
    cmap=cmap,
    vmin=imgs[0].min(),
    vmax=imgs[0].max()
)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
# plt.savefig(os.path.join(results_dir, "grfs_diffusion.png"))
plt.close()

"""
    Train
"""

# Fit model to dataset
model = sbgm.train.train(
    train_key,
    model,
    sde,
    dataset,
    opt=opt(lr, **opt_kwargs),
    n_steps=n_steps,
    batch_size=batch_size,
    sample_size=sample_size,
    eu_sample=eu_sample,
    ode_sample=ode_sample,
    reload_opt_state=False,
    plot_train_data=True,
    sharding=sharding,
    replicated_sharding=replicated_sharding,
    save_dir=root_dir,
    cmap=cmap
)

"""
    Sample p(x|y)
"""

key, key_Q, sample_key = jr.split(key, 3)
sample_keys = jr.split(sample_key, sample_size ** 2)

# Sample random labels or use parameter prior for labels
Q, A = dataset.label_fn(key_Q, sample_size ** 2)

# EU sampling
sample_fn = sbgm.sample.get_eu_sample_fn(model, sde, dataset.data_shape)
eu_samples = jax.vmap(sample_fn)(sample_keys, Q, A)

# ODE sampling
sample_fn = sbgm.sample.get_ode_sample_fn(model, sde, dataset.data_shape)
ode_samples = jax.vmap(sample_fn)(sample_keys, Q, A)

fig, axs = plt.subplots(1, 2, dpi=150)
ax = axs[0]
ax.imshow(
    rearrange(ode_samples, "(h w) c x y -> (h x) (w y) c", h=10, w=10, x=n_pix, y=n_pix), 
    cmap=cmap
)
ax = axs[1]
ax.imshow(
    rearrange(ode_samples, "(h w) c x y -> (h x) (w y) c", h=10, w=10, x=n_pix, y=n_pix), 
    cmap=cmap
)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
# plt.savefig(os.path.join(results_dir, "grfs_samples.png"))
plt.close()

"""
    Log-likelihood
"""
key, key_L = jr.split(key)

log_likelihood_fn = sbgm.ode.get_log_likelihood_fn(
    model, sde, dataset.data_shape, exact_log_prob=True
)
L_X = log_likelihood_fn(X[0], Q[0], A[0], key_L)

L_X