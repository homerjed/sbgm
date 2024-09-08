import jax.random as jr

import sbgm
import data 
import configs 


def main():
    """
        Fit a score-based diffusion model.
    """

    datasets_path = "/project/ls-gruen/users/jed.homer/"
    root_dir = "./"

    config = [
        configs.mnist_config(),
        configs.grfs_config(),
        configs.flowers_config(),
        configs.cifar10_config(),
        configs.quijote_config()
    ][-1]

    key = jr.key(config.seed)
    data_key, model_key, train_key = jr.split(key, 3)

    dataset = data.get_dataset(
        datasets_path, data_key, config
    )
    sharding = sbgm.shard.get_sharding()
    reload_opt_state = False # Restart training or not
        
    # Diffusion model 
    model = sbgm.models.get_model(
        model_key, 
        config.model.model_type, 
        dataset.data_shape, 
        dataset.context_shape, 
        dataset.parameter_dim,
        config
    )

    # Stochastic differential equation (SDE)
    sde = sbgm.sde.get_sde(config.sde)

    # Fit model to dataset
    model = sbgm.train.train(
        train_key,
        model,
        sde,
        dataset,
        config,
        reload_opt_state=reload_opt_state,
        sharding=sharding,
        save_dir=root_dir
    )


if __name__ == "__main__":
    main()