import jax
import jax.numpy as jnp
import jax.random as jr

from sbgm.models import UNet, ResidualNetwork, Mixer2d


def test_resnet():

    key = jr.key(0)

    x = jnp.ones((5,))
    t = jnp.ones((1,))

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


def test_mixer():

    key = jr.key(0)

    x = jnp.ones((1, 32, 32))
    t = jnp.ones((1,))

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



def test_unet():

    key = jr.key(0)

    x = jnp.ones((1, 32, 32))
    t = jnp.ones((1,))

    q_dim = 1
    a_dim = 2

    unet = UNet(
        x.shape,
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=32,
        heads=2,
        dim_head=32,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )

    q = jnp.ones((1, 32, 32))
    a = jnp.ones((2,))

    out = unet(t, x, q=q, a=a, key=key)
    assert out.shape == x.shape

    q_dim = None
    a_dim = None

    unet = UNet(
        x.shape,
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=32,
        heads=2,
        dim_head=32,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )
    
    out = unet(t, x, key=key)
    assert out.shape == x.shape

    q_dim = 1
    a_dim = None

    unet = UNet(
        x.shape,
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=32,
        heads=2,
        dim_head=32,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )
    
    q = jnp.ones((1, 32, 32))

    out = unet(t, x, q=q, key=key)
    assert out.shape == x.shape

    q_dim = None
    a_dim = 2 

    unet = UNet(
        x.shape,
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=32,
        heads=2,
        dim_head=32,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
        q_dim=q_dim,
        a_dim=a_dim,
        key=key
    )
    
    a = jnp.ones((2,))
    out = unet(t, x, a=a, key=key)
    assert out.shape == x.shape