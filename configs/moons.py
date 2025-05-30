import jax 
import jax.numpy as jnp
import ml_collections


def moons_config():
    config = ml_collections.ConfigDict()

    config.seed                  = 0

    # Data
    config.dataset_name          = "moons" 

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.model_type             = "mlp"
    model.width_size             = 128
    model.depth                  = 5
    model.activation             = jax.nn.tanh
    model.dropout_p              = 0.1

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                      = "VP"
    sde.t0                       = 0.
    sde.t1                       = 4.
    sde.dt                       = 0.1
    sde.beta_integral            = lambda t: t 
    sde.weight_fn                = lambda t: 1. - jnp.exp(-sde.beta_integral(t)) 

    # Sampling
    config.use_ema               = True
    config.sample_size           = 64 # Squared in sampling
    config.exact_log_prob        = True
    config.ode_sample            = False 
    config.eu_sample             = False 

    # Optimisation hyperparameters
    config.start_step            = 0
    config.n_steps               = 100_000
    config.batch_size            = 1024 
    config.accumulate_gradients  = False
    config.n_minibatches         = 1
    config.sample_and_save_every = 5_000
    config.lr                    = 3e-4
    config.opt                   = "adabelief" 
    config.opt_kwargs            = {}

    # Other
    config.cmap                  = "PiYG" 

    return config