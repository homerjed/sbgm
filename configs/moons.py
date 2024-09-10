import jax 
import ml_collections


def moons_config():
    config = ml_collections.ConfigDict()

    config.seed            = 0

    # Data
    config.dataset_name    = "moons" 

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.model_type       = "mlp"
    model.width_size       = 128
    model.depth            = 5
    model.activation       = jax.nn.tanh
    model.dropout_p        = 0.1

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                = "VP"
    sde.t0                 = 0.
    sde.t1                 = 4.
    sde.dt                 = 0.1
    sde.beta_integral      = lambda t: t 
    sde.N                  = 1000

    # Sampling
    config.use_ema         = True
    config.sample_size     = 64 
    config.exact_logp      = True
    config.ode_sample      = False 
    config.eu_sample       = False 

    # Optimisation hyperparameters
    config.start_step      = 0
    config.n_steps         = 50_000
    config.batch_size      = 512 
    config.print_every     = 5_000
    config.lr              = 1e-4
    config.opt             = "adabelief" 
    config.opt_kwargs      = {}

    # Other
    config.cmap            = "PiYG" 

    return config