import ml_collections


def mnist_config():
    config = ml_collections.ConfigDict()

    config.seed            = 0

    # Data
    config.dataset_name    = "mnist" 

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.model_type       = "UNet"
    model.is_biggan        = False
    model.dim_mults        = [1, 1, 1]
    model.hidden_size      = 32
    model.heads            = 4
    model.dim_head         = 64
    model.dropout_rate     = 0.3
    model.num_res_blocks   = 2
    model.attn_resolutions = [8, 16, 32]
    model.final_activation = None

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                = "VP"
    sde.t1                 = 1.
    sde.t0                 = 1e-5
    sde.dt                 = 0.1
    sde.beta_integral      = lambda t: t 

    # Sampling
    config.sample_size     = 8
    config.exact_logp      = False
    config.ode_sample      = True
    config.eu_sample       = True
    config.use_ema         = False

    # Optimisation hyperparameters
    config.start_step      = 0
    config.n_steps         = 1_000_000
    config.lr              = 1e-4
    config.batch_size      = 256 
    config.sample_and_save_every     = 1_000
    config.opt             = "adabelief"
    config.opt_kwargs      = {}
    config.num_workers     = 8

    # Other
    config.cmap            = "gray_r" 

    return config