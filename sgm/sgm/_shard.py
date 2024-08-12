from typing import Tuple, Optional
import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx
from jaxtyping import Array


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
    batch: Tuple[Array, ...], 
    sharding: Optional[NamedSharding] = None
) -> Tuple[Array, ...]:
    if sharding:
        batch = eqx.filter_shard(batch, sharding)
    return batch