import ml_collections
import jax.numpy as jnp


def quijote_config():
    config = ml_collections.ConfigDict()

    config.seed                  = 0

    # Data
    config.dataset_name          = "quijote" 
    config.n_pix                 = 64

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.model_type             = "Mixer"
    model.patch_size             = 2
    model.hidden_size            = 1024
    model.mix_patch_size         = 512
    model.mix_hidden_size        = 1024
    model.num_blocks             = 5
    model.t1                     = 10.
    model.final_activation       = None #"tanh"

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                      = "VP"
    sde.t1                       = model.t1
    sde.t0                       = 0. 
    sde.dt                       = 0.1
    sde.beta_integral            = lambda t: t 
    sde.weight_fn                = lambda t: 1. - jnp.exp(-sde.beta_integral(t)) 

    # Sampling
    config.use_ema               = False
    config.sample_size           = 5
    config.exact_log_prob        = False
    config.ode_sample            = True
    config.eu_sample             = True

    # Optimisation hyperparameters
    config.start_step            = 0
    config.n_steps               = 1_000_000
    config.lr                    = 1e-4
    config.batch_size            = 32
    config.accumulate_gradients  = False
    config.n_minibatches         = 1
    config.sample_and_save_every = 5_000
    config.opt                   = "adabelief"
    config.opt_kwargs            = {} 
    config.num_workers           = 8

    # Other
    config.cmap                  = "gist_stern" 

    return config 