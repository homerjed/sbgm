import jax
import jax.numpy as jnp
import jax.random as jr

from sbgm.models import UNet, ResidualNetwork, Mixer2d, DiT


def test_resnet():

    key = jr.key(0)

    x = jnp.ones((5,))
    t = jnp.ones(())

    a = jnp.ones((3,))
    q = None

    net = ResidualNetwork(
        x.size, 
        width_size=32, 
        depth=2,
        q_dim=None,
        a_dim=a.size,
        dropout_p=0.1,
        activation=jax.nn.tanh,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    a = None
    q = jnp.ones((5,)) 

    net = ResidualNetwork(
        x.size, 
        width_size=32, 
        depth=2,
        dropout_p=0.1,
        activation=jax.nn.tanh,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    a = jnp.ones((3,))
    q = jnp.ones((5,)) 

    net = ResidualNetwork(
        x.size, 
        width_size=32, 
        depth=2,
        q_dim=q.size,
        a_dim=a.size,
        dropout_p=0.1,
        activation=jax.nn.tanh,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.isfinite(out)

    net = ResidualNetwork(
        x.size, 
        width_size=32, 
        depth=2,
        dropout_p=0.1,
        activation=jax.nn.tanh,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


def test_mixer():

    key = jr.key(0)

    x = jnp.ones((1, 32, 32))
    t = jnp.ones(())

    q = jnp.ones((1, 32, 32))
    a = jnp.ones((5,))

    q_dim = 1
    a_dim = 5

    net = Mixer2d(
        x.shape,
        patch_size=2,
        hidden_size=256,
        mix_patch_size=4,
        mix_hidden_size=256,
        num_blocks=3,
        t1=1.0,
        embedding_dim=8,
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q = None
    a = jnp.ones((5,))

    q_dim = None
    a_dim = 5

    net = Mixer2d(
        x.shape,
        patch_size=2,
        mix_patch_size=4,
        hidden_size=256,
        mix_hidden_size=256,
        num_blocks=3,
        t1=1.0,
        embedding_dim=8,
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q = jnp.ones((1, 32, 32)) 
    a = None

    q_dim = 1 
    a_dim = None

    net = Mixer2d(
        x.shape,
        patch_size=2,
        mix_patch_size=4,
        hidden_size=256,
        mix_hidden_size=256,
        num_blocks=3,
        t1=1.0,
        embedding_dim=8,
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q = None
    a = None

    q_dim = None 
    a_dim = None

    net = Mixer2d(
        x.shape,
        patch_size=2,
        mix_patch_size=4,
        hidden_size=256,
        mix_hidden_size=256,
        num_blocks=3,
        t1=1.0,
        embedding_dim=8,
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )

    out = net(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


def test_unet():

    key = jr.key(0)

    hidden_size = 32
    img_size = 32
    n_channels = 1
    embed_dim = 32
    patch_size = 8 
    n_heads = 2
    depth = 2

    x = jnp.ones((1, 32, 32))
    t = jnp.ones(())

    q_dim = 1
    a_dim = 2

    dit = DiT(
        img_size=img_size,
        channels=n_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        n_heads=n_heads,
        q_dim=q_dim, # Number of channels in conditioning map
        a_dim=a_dim,  
        key=key
    )

    q = jnp.ones((1, 32, 32))
    a = jnp.ones((2,))

    out = dit(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q_dim = None
    a_dim = None

    dit = DiT(
        img_size=img_size,
        channels=n_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        n_heads=n_heads,
        q_dim=q_dim, # Number of channels in conditioning map
        a_dim=a_dim,  
        key=key
    )
    
    out = dit(t, x, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q_dim = 1
    a_dim = None

    dit = DiT(
        img_size=img_size,
        channels=n_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        n_heads=n_heads,
        q_dim=q_dim, # Number of channels in conditioning map
        a_dim=a_dim,  
        key=key
    )
    
    q = jnp.ones((1, 32, 32))

    out = dit(t, x, q=q, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q_dim = None
    a_dim = 2 

    dit = DiT(
        img_size=img_size,
        channels=n_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        n_heads=n_heads,
        q_dim=q_dim, # Number of channels in conditioning map
        a_dim=a_dim,  
        key=key
    )
    
    a = jnp.ones((2,))
    out = dit(t, x, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


def test_unet():

    key = jr.key(0)

    hidden_size = 32
    n_channels = 1
    dim_mults = (1, 1)
    heads = 2
    dim_head = 32
    dropout_rate = 0.0

    x = jnp.ones((1, 32, 32))
    t = jnp.ones(())

    q_dim = 1
    a_dim = 2

    unet = UNet(
        dim=hidden_size,
        channels=n_channels,
        dim_mults=dim_mults,
        attn_heads=heads,
        attn_dim_head=dim_head,
        dropout=dropout_rate,
        learned_sinusoidal_cond=True,
        random_fourier_features=True,
        a_dim=a_dim, 
        q_channels=q_dim,
        key=key
    )

    q = jnp.ones((1, 32, 32))
    a = jnp.ones((2,))

    out = unet(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q_dim = None
    a_dim = None

    unet = UNet(
        dim=hidden_size,
        channels=n_channels,
        dim_mults=dim_mults,
        attn_heads=heads,
        attn_dim_head=dim_head,
        dropout=dropout_rate,
        learned_sinusoidal_cond=True,
        random_fourier_features=True,
        a_dim=a_dim, 
        q_channels=q_dim,
        key=key
    )
    
    out = unet(t, x, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q_dim = 1
    a_dim = None

    unet = UNet(
        dim=hidden_size,
        channels=n_channels,
        dim_mults=dim_mults,
        attn_heads=heads,
        attn_dim_head=dim_head,
        dropout=dropout_rate,
        learned_sinusoidal_cond=True,
        random_fourier_features=True,
        a_dim=a_dim, 
        q_channels=q_dim,
        key=key
    )
    
    q = jnp.ones((1, 32, 32))

    out = unet(t, x, q=q, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))

    q_dim = None
    a_dim = 2 

    unet = UNet(
        dim=hidden_size,
        channels=n_channels,
        dim_mults=dim_mults,
        attn_heads=heads,
        attn_dim_head=dim_head,
        dropout=dropout_rate,
        learned_sinusoidal_cond=True,
        random_fourier_features=True,
        a_dim=a_dim, 
        q_channels=q_dim,
        key=key
    )
    
    a = jnp.ones((2,))
    out = unet(t, x, a=a, key=key)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))