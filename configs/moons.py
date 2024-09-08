import jax 
import ml_collections


def moons_config():
    config = ml_collections.ConfigDict()

    config.seed                = 0

    # Data
    config.dataset_name        = "moons" 

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.model_type       = "mlp"
    model.model.width_size = 128
    model.depth            = 2
    model.activation       = jax.nn.tanh
    model.dropout_p        = 0.1

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                = "VP"
    sde.t1                 = 4.
    sde.t0                 = 0.
    sde.dt                 = 0.02
    sde.beta_integral      = lambda t: t ** 2.

    # Sampling
    config.use_ema         = True
    config.sample_size     = 64 
    config.exact_logp      = True
    config.ode_sample      = True
    config.eu_sample       = True

    # Optimisation hyperparameters
    config.start_step      = 0
    config.n_steps         = 200_000
    config.batch_size      = 256 
    config.print_every     = 5_000
    config.lr              = 1e-4
    config.opt             = "adabelief" 

    # Other
    config.cmap            = "PiYG" 

    return config