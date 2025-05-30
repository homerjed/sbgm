from typing import Tuple, Optional
import jax


def get_shardings() -> Tuple[
    Optional[jax.sharding.NamedSharding], Optional[jax.sharding.NamedSharding]
]:
    """
        Create sharding strategies based on the number of available JAX devices.

        If multiple devices are available, this function creates a 1D device mesh and returns:
        - a data-parallel sharding along the mesh dimension `'x'`
        - a fully replicated sharding across devices (for sharding a model across the devices)

        If only one device is available, both sharding strategies are returned as `None`.

        Returns
        -------
        sharding : Optional[jax.sharding.NamedSharding]
            Sharding strategy that partitions data across devices along axis `'x'`,
            or `None` if only one device is available.
        replicated : Optional[jax.sharding.NamedSharding]
            Fully replicated sharding strategy (no partitioning),
            or `None` if only one device is available.

        Notes
        -----
        - This function prints out the number and identities of local devices.
    """
    devices = jax.devices()
    n_devices = len(devices)

    print(f"Running on {n_devices} local devices: \n\t{devices}")

    if n_devices > 1:

        mesh = jax.sharding.Mesh(devices, "x")

        replicated = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        )

    else:
        sharding = replicated = None

    return sharding, replicated