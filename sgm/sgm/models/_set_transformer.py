from typing import List
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn


class MultiheadAttentionBlock(eqx.Module):
    ln_x: nn.LayerNorm
    ln_y: nn.LayerNorm
    ln: nn.LayerNorm
    xx_attention: nn.MultiheadAttention
    xy_attention: nn.MultiheadAttention
    linear_0: nn.Linear
    linear_1: nn.Linear

    def __init__(self, num_heads, data_dim, hidden_dim, *, key):
        self.ln_x = nn.LayerNorm(data_dim)
        self.ln_y = nn.LayerNorm(data_dim)
        self.ln = nn.LayerNorm(data_dim)

        key, key_xx, key_xy = jr.split(key, 3)
        self.xx_attention = nn.MultiheadAttention(
            num_heads=num_heads, query_size=data_dim, key=key_xx
        )
        self.xy_attention = nn.MultiheadAttention(
            num_heads=num_heads, query_size=data_dim, key=key_xy
        )
        key_0, key_1 = jr.split(key)
        self.linear_0 = nn.Linear(data_dim, hidden_dim, key=key_0)
        self.linear_1 = nn.Linear(hidden_dim, data_dim, key=key_1)

    def _map(layer, input):
        return jax.vmap(layer)(input)

    def __call__(self, x, y):
        if x is y:
            _x = jax.vmap(self.ln_x)(x)
            x_a = self.xx_attention(_x, _x, _x)
        else:
            _x = jax.vmap(self.ln_x)(x)
            _y = jax.vmap(self.ln_y)(y)
            x_a = self.xy_attention(_x, _y, _y)

        x = x + x_a

        _x = jax.vmap(self.ln)(x)
        _x = jax.vmap(self.linear_0)(_x)
        _x = jax.nn.gelu(_x)
        _x = jax.vmap(self.linear_1)(_x)

        x = x + _x
        return x


class SetTransformer(eqx.Module):
    projection: nn.Linear
    attention_blocks: List[MultiheadAttentionBlock]
    ln: nn.LayerNorm
    out: nn.Linear

    def __init__(
        self,
        data_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        n_heads,
        *,
        key
    ):
        self.projection = nn.Linear(data_dim, embedding_dim, key=key)
        self.attention_blocks = [
            MultiheadAttentionBlock(
                n_heads,
                embedding_dim, #data_dim,
                hidden_dim,
                key=_key # embedding_dim not data_dim?
            )
            for _key in jr.split(key, n_layers)
        ]
        self.ln = nn.LayerNorm(embedding_dim)
        self.w = jnp.zeros((data_dim, embedding_dim)) 
        self.b = jnp.zeros((data_dim,)) 
        self.out = nn.Linear(embedding_dim, data_dim, key=key)

    def __call__(self, x):
        # Map set points to embedding dim
        x = jax.vmap(self.projection)(x)
        # Attention
        for a in self.attention_blocks:
            x = a(x, x)
        # Regressor
        print("x", x.shape)
        x = jax.vmap(self.ln)(x)
        x = jax.vmap(self.out)(x)
        return x


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    data_dim = 99

    attention = nn.MultiheadAttention(
        num_heads=4, 
        query_size=data_dim, 
        key=key
    )
    x = jnp.ones((3, data_dim))
    a = attention(x, x, x)
    print(a.shape)

    attention_block = MultiheadAttentionBlock(
        num_heads=4, 
        data_dim=data_dim, 
        hidden_dim=32, 
        key=key
    )
    print(attention_block(x, x).shape)

    transformer = SetTransformer(
        data_dim,
        embedding_dim=5,
        hidden_dim=8,
        n_layers=2,
        n_heads=4,
        key=key
    )
    print(transformer(x).shape)