from typing import Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
import einops
import equinox as eqx


class AdaLayerNorm(eqx.Module):
    net: eqx.Module
    ln: eqx.nn.LayerNorm
    data_shape: Sequence[int]

    def __init__(self, data_shape, condition_dim, *, key):
        """
            Adaptive layer norm; generate scale and shift parameters from conditioning context.
        """
        data_dim = jnp.prod(jnp.asarray(data_shape))
        self.net = eqx.nn.Linear(condition_dim, data_dim * 2, key=key)
        # Don't use bias or scale since these will be learnable through the conditioning context
        self.ln = eqx.nn.LayerNorm(data_shape, use_bias=False, use_weight=False)
        self.data_shape = data_shape

    def __call__(self, x, conditioning):
        # Compute scale and shift parameters from conditioning context
        scale_and_shift = jax.nn.gelu(self.net(conditioning))
        scale, shift = jnp.split(scale_and_shift, 2)
        scale = scale.reshape(self.data_shape)
        shift = shift.reshape(self.data_shape)
        # Apply layer norm
        x = self.ln(x)
        # Apply scale and shift (same scale, shift to all elements)
        x = x * (1. + scale) + shift
        return x


class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    num_patches: int
    hidden_size: int
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self, 
        num_patches, 
        hidden_size, 
        mix_patch_size, 
        mix_hidden_size, 
        context_dim,
        *, 
        key
    ):
        tkey, ckey = jr.split(key)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, 
            num_patches, 
            mix_patch_size, 
            depth=1, 
            key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, 
            hidden_size, 
            mix_hidden_size, 
            depth=1, 
            key=ckey
        )
        # Possible adaNorm instead of layernorm?
        # self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        # self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))
        key1, key2 = jr.split(key)
        self.norm1 = AdaLayerNorm(
            (hidden_size, num_patches), context_dim, key=key1
        )
        self.norm2 = AdaLayerNorm(
            (num_patches, hidden_size), context_dim, key=key2
        )
        self.hidden_size = hidden_size
        self.num_patches = num_patches 

    def __call__(self, y, q):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y, q))
        y = einops.rearrange(y, "c p -> p c")
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y, q))
        y = einops.rearrange(y, "p c -> c p")
        return y


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Build sinusoidal embeddings (from Fairseq)."""

    # Convert scalar timesteps to an array
    if jnp.isscalar(timesteps):
        timesteps = jnp.array([timesteps], dtype=dtype)

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list[MixerBlock]
    norm: eqx.nn.LayerNorm
    t1: float

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        context_dim,
        *,
        key
    ):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            # input_size + 1, # Time is tiled along with time as channels
            input_size, # Time is tiled along with time as channels
            hidden_size, 
            patch_size, 
            stride=patch_size, 
            key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, 
            input_size, 
            patch_size, 
            stride=patch_size, 
            key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, 
                hidden_size, 
                mix_patch_size, 
                mix_hidden_size, 
                context_dim=context_dim + 4,
                key=bkey
            ) 
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1

    def __call__(self, t, y, q, *, key=None):
        _, height, width = y.shape
        # t = t / self.t1
        t, q = jnp.atleast_1d(t), jnp.atleast_1d(q)
        # _t = einops.repeat(
        #     t, "1 -> 1 h w", h=height, w=width
        # )
        # y = jnp.concatenate([y, _t])
        t = get_timestep_embedding(t, embedding_dim=4)[0]
        q = jnp.concatenate([q, t])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(
            y, "c h w -> c (h w)"
        )
        for block in self.blocks:
            y = block(y, q)
        y = self.norm(y)
        y = einops.rearrange(
            y, "c (h w) -> c h w", h=patch_height, w=patch_width
        )
        return self.conv_out(y)


"""
class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key
    ):
        tkey, ckey = jr.split(key, 2)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, num_patches, mix_patch_size, depth=1, key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
        )
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, y):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = einops.rearrange(y, "c p -> p c")
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = einops.rearrange(y, "p c -> c p")
        return y


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list
    norm: eqx.nn.LayerNorm
    t1: float

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        *,
        key,
    ):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            input_size + 1, hidden_size, patch_size, stride=patch_size, key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, input_size, patch_size, stride=patch_size, key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bkey
            )
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1

    def __call__(self, t, y):
        t = t / self.t1
        _, height, width = y.shape
        t = einops.repeat(t, "-> 1 h w", h=height, w=width)
        y = jnp.concatenate([y, t])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(y, "c h w -> c (h w)")
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = einops.rearrange(y, "c (h w) -> c h w", h=patch_height, w=patch_width)
        return self.conv_out(y)

"""