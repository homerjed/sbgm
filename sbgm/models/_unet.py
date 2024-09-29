import math
from collections.abc import Callable
from typing import Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array, jaxtyped
from beartype import beartype as typechecker
from einops import rearrange


class SinusoidalPosEmb(eqx.Module):
    emb: Array

    def __init__(self, dim: int):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: Array) -> Array:
        emb = x * self.emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class LinearTimeSelfAttention(eqx.Module):
    group_norm: eqx.nn.GroupNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        key: Key,
        heads: int = 4,
        dim_head: int = 32
    ):
        keys = jax.random.split(key, 2)
        self.group_norm = eqx.nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=keys[0])
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=keys[1])

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, 
            "(qkv heads c) h w -> qkv heads c (h w)", 
            heads=self.heads, 
            qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn, hen -> hde", k, v)
        out = jnp.einsum("hde, hdn -> hen", context, q)
        out = rearrange(
            out, 
            "heads c (h w) -> (heads c) h w", 
            heads=self.heads, 
            h=h, 
            w=w
        )
        return self.to_out(out)


def upsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H, 1, W, 1])
    y = jnp.tile(y, [1, 1, factor, 1, factor])
    return jnp.reshape(y, [C, H * factor, W * factor])


def downsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H // factor, factor, W // factor, factor])
    return jnp.mean(y, axis=[2, 4])


def exact_zip(*args):
    _len = len(args[0])
    for arg in args:
        assert len(arg) == _len
    return zip(*args)


def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jr.split(key)


class Residual(eqx.Module):
    fn: LinearTimeSelfAttention

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(eqx.Module):
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: list[Union[Callable, eqx.nn.Linear]]
    scaling: Union[
        None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d
    ]
    block1_groupnorm: eqx.nn.GroupNorm
    block1_conv: eqx.nn.Conv2d
    block2_layers: list[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    attn: Optional[Residual]
    a_dim: Optional[int]

    def __init__(
        self,
        dim_in,
        dim_out,
        is_biggan,
        up,
        down,
        time_emb_dim,
        dropout_rate,
        is_attn,
        heads,
        dim_head,
        a_dim=None,
        *,
        key,
    ):
        keys = jr.split(key, 7)
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [
            jax.nn.silu,
            eqx.nn.Linear(
                time_emb_dim + a_dim if a_dim is not None else time_emb_dim, 
                dim_out, 
                key=keys[0]
            ),
        ]
        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)
        self.block1_conv = eqx.nn.Conv2d(dim_in, dim_out, 3, padding=1, key=keys[1])
        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
            eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=keys[2]),
        ]

        assert not self.up or not self.down

        if is_biggan:
            if self.up:
                self.scaling = upsample_2d
            elif self.down:
                self.scaling = downsample_2d
            else:
                self.scaling = None
        else:
            if self.up:
                self.scaling = eqx.nn.ConvTranspose2d(
                    dim_in,
                    dim_in,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=keys[3],
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=keys[4],
                )
            else:
                self.scaling = None
        # For DDPM Yang use their own custom layer called NIN, which is
        # equivalent to a 1x1 conv
        self.res_conv = eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=keys[5])

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(
                    dim_out,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys[6],
                )
            )
        else:
            self.attn = None
        self.a_dim = a_dim

    def __call__(
        self, 
        x: Array, 
        t: Union[float, Array], 
        a: Array = None, 
        *, 
        key: Key
    ) -> Array:
        C, _, _ = x.shape
        # In DDPM, each set of resblocks ends with an up/down sampling. In
        # biggan there is a final resblock after the up/downsampling. In this
        # code, the biggan approach is taken for both.
        # norm -> nonlinearity -> up/downsample -> conv follows Yang
        # https://github.dev/yang-song/score_sde/blob/main/models/layerspp.py
        h = jax.nn.silu(self.block1_groupnorm(x))
        if self.up or self.down:
            h = self.scaling(h) # pyright: ignore
            x = self.scaling(x) # pyright: ignore
        h = self.block1_conv(h)

        for layer in self.mlp_layers:
            # Add parameter conditioning here
            if isinstance(layer, eqx.nn.Linear):
                if a is not None and self.a_dim is not None:
                    _input = jnp.concatenate([t, a]) 
                else:
                    _input = t
            else:
                _input = t
            t = layer(_input)
        h = h + t[..., jnp.newaxis, jnp.newaxis]
        for layer in self.block2_layers:
            # Precisely 1 dropout layer in block2_layers which requires a key.
            if isinstance(layer, eqx.nn.Dropout):
                h = layer(h, key=key)
            else:
                h = layer(h)

        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)

        out = (h + x) / jnp.sqrt(2.)
        if self.attn is not None:
            out = self.attn(out)
        return out


class UNet(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv2d
    down_res_blocks: list[list[ResnetBlock]]
    mid_block1: ResnetBlock
    mid_block2: ResnetBlock
    ups_res_blocks: list[list[ResnetBlock]]
    final_conv_layers: list[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]
    final_activation: Optional[Callable]
    q_dim: int
    a_dim: int

    def __init__(
        self,
        data_shape: tuple[int, int, int],
        is_biggan: bool,
        dim_mults: list[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: list[int],
        final_activation: Optional[Callable] = jax.nn.tanh,
        q_dim: Optional[int] = None, # Number of channels in conditioning map
        a_dim: Optional[int] = None, # Number of parameters in conditioning 
        *,
        key: jr.PRNGKey,
    ):
        """
            UNet score network. 
            
            This model supports optional conditioning through `q_dim` and `a_dim`, 
            and can be adjusted to different input shapes and configurations.

            Parameters:
            -----------
            `data_shape` : `tuple[int, int, int]`
                Shape of the input data as `(height, width, channels)`.
            
            `is_biggan` : `bool`
                Whether the model is based on the BigGAN architecture.
            
            `dim_mults` : `list[int]`
                List of integers representing the dimension multipliers for each level in the UNet.
            
            `hidden_size` : `int`
                Size of the hidden layers in the MLPs and other parts of the network.
            
            `heads` : `int`
                Number of heads used in the attention mechanism (if any).
            
            `dim_head` : `int`
                Dimension of each head in the attention mechanism.
            
            `dropout_rate` : `float`
                The dropout rate used in various parts of the network.
            
            `num_res_blocks` : `int`
                Number of residual blocks used at each stage in the UNet.
            
            `attn_resolutions` : `list[int]`
                List of resolutions at which attention is applied in the network.
            
            `final_activation` : `Optional[Callable]`, default: `jax.nn.tanh`
                The final activation function to be applied to the output.
            
            `q_dim` : `Optional[int]`, default: `None`
                The number of channels in the conditioning map. 
                Must be same shape as `x` in `__call__`.
            
            `a_dim` : `Optional[int]`, default: `None`
                The number of parameters in the conditioning.
            
            `key` : `jr.PRNGKey`
                JAX random key used for initialization.
        """

        keys = jr.split(key, 7)

        data_channels, in_height, in_width = data_shape

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(exact_zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = eqx.nn.MLP(
            hidden_size + a_dim if a_dim is not None else hidden_size,
            hidden_size,
            width_size=4 * hidden_size,
            depth=1,
            activation=jax.nn.silu,
            key=keys[0],
        )
        self.first_conv = eqx.nn.Conv2d(
            data_channels + q_dim if q_dim is not None else data_channels, 
            hidden_size, 
            kernel_size=3, 
            padding=1, 
            key=keys[1]
        )

        h, w = in_height, in_width
        self.down_res_blocks = []
        num_keys = len(in_out) * num_res_blocks - 1
        keys_resblock = jr.split(keys[2], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = [
                ResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                )
            ]
            i += 1
            for _ in range(num_res_blocks - 2):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=True,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(res_blocks)
        assert i == num_keys

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=True,
            heads=heads,
            dim_head=dim_head,
            a_dim=a_dim,
            key=keys[3],
        )
        self.mid_block2 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=False,
            heads=heads,
            dim_head=dim_head,
            a_dim=a_dim,
            key=keys[4],
        )

        self.ups_res_blocks = []
        num_keys = len(in_out) * (num_res_blocks + 1) - 1
        keys_resblock = jr.split(keys[5], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False

            res_blocks = []
            for _ in range(num_res_blocks - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out * 2,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
            res_blocks.append(
                ResnetBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_in,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                )
            )
            i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_in,
                        dim_out=dim_in,
                        is_biggan=is_biggan,
                        up=True,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                    )
                )
                i += 1
                h, w = h * 2, w * 2

            self.ups_res_blocks.append(res_blocks)
        assert i == num_keys

        self.final_conv_layers = [
            eqx.nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            jax.nn.silu,
            eqx.nn.Conv2d(hidden_size, data_channels, 1, key=keys[6]),
        ]
        self.final_activation = final_activation

        self.q_dim = q_dim 
        self.a_dim = a_dim 

    def __call__(
        self, 
        t: Union[float, Array], 
        y: Array, 
        q: Optional[Array] = None, 
        a: Optional[Array] = None, 
        *, 
        key: Optional[Key] = None
    ) -> Array:
        t = self.time_pos_emb(t)

        if self.a_dim is not None and a is not None:
            _input = jnp.concatenate([t, a])
        else:
            _input = t
        t = self.mlp(_input) 

        # Stack d_g, d_m on channel axis
        if self.q_dim is not None and q is not None:
            _input = jnp.concatenate([y, q])
        else:
            _input = y
        h = self.first_conv(_input)

        # Downsampling blocks
        hs = [h]
        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                h = res_block(h, t, key=subkey)
                hs.append(h)

        # Middle blocks
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, t, a=a, key=subkey) 
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, t, a=a, key=subkey) 

        # Upsampling blocks
        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                if res_block.up:
                    h = res_block(h, t, key=subkey)
                else:
                    h = res_block(jnp.concatenate((h, hs.pop()), axis=0), t, key=subkey)

        assert len(hs) == 0

        for layer in self.final_conv_layers:
            h = layer(h)
        return self.final_activation(h) if self.final_activation is not None else h