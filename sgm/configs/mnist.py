class MNISTConfig:
    seed                = 0
    # Data
    dataset_name        = "mnist" 
    # Model
    model_type: str     = "UNet"
    model_args          = dict(
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=32,
        heads=4, 
        dim_head=64,
        dropout_rate=0.3,
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
        final_activation=None
    )
    # model_type          = "Mixer"
    # model_args          = dict(
    #     patch_size=2,
    #     hidden_size=512,
    #     mix_patch_size=512,
    #     mix_hidden_size=512,
    #     num_blocks=4
    # )
    # SDE
    sde                 = "VP"
    t1                  = 1.
    t0                  = 1e-5
    dt                  = 0.1
    beta_integral       = lambda t: t 
    N                   = 1000
    # Sampling
    sample_size         = 8
    exact_logp          = False
    ode_sample          = True
    eu_sample           = True
    use_ema             = False
    # Optimisation hyperparameters
    start_step          = 0
    n_steps             = 1_000_000
    lr                  = 1e-4
    batch_size          = 256 
    print_every         = 1_000
    opt                 = "adabelief"
    opt_kwargs          = {}
    num_workers         = 8
    # Other
    cmap                = "gray_r" 