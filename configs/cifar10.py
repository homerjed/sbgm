import ml_collections


def cifar10_config():
    config = ml_collections.ConfigDict()

    config.seed            = 0

    # Data
    config.dataset_name    = "cifar10" 

    # Model
    # model_type          = "Mixer"
    # model_args          = dict(
    #     patch_size=1,
    #     hidden_size=1024,
    #     mix_patch_size=128,
    #     mix_hidden_size=128,
    #     num_blocks=4
    # )
    # model_args          = dict(
    #     patch_size=4,
    #     hidden_size=512,
    #     mix_patch_size=512,
    #     mix_hidden_size=512,
    #     num_blocks=4
    # )
    config.model = model = ml_collections.ConfigDict()
    model.model_type       = "UNet"
    model.is_biggan        = False
    model.dim_mults        = [1, 1, 1]
    model.hidden_size      = 128
    model.heads            = 4
    model.dim_head         = 64
    model.dropout_rate     = 0.3
    model.num_res_blocks   = 2
    model.attn_resolutions = [8, 16, 32]
    model.final_activation = None

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                = "VP"
    sde.t1                 = 8.
    sde.t0                 = 1e-5 
    sde.dt                 = 0.1
    sde.N                  = 1000
    sde.beta_integral      = lambda t: t 

    # Sampling
    config.use_ema         = False
    config.sample_size     = 5
    config.exact_logp      = False
    config.ode_sample      = True
    config.eu_sample       = True

    # Optimisation hyperparameters
    config.start_step      = 0
    config.n_steps         = 1_000_000
    config.lr              = 1e-4
    config.batch_size      = 512 #256 # 256 with UNet
    config.print_every     = 1_000
    config.opt             = "adabelief"
    config.opt_kwargs      = {} 
    config.num_workers     = 8

    # Other
    config.cmap                = None

    return config