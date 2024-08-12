class CIFAR10Config:
    seed                = 0
    # Data
    dataset_name        = "cifar10" 
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
    model_type: str     = "UNet"
    model_args          = dict(
        is_biggan=False,
        dim_mults=[1, 2, 4],
        hidden_size=128,
        heads=4, 
        dim_head=64,
        dropout_rate=0.3,
        num_res_blocks=2,
        attn_resolutions=[8, 16, 32],
    )
    use_ema             = False
    # SDE
    sde                 = "VP"
    t1                  = 8.
    t0                  = 1e-5 
    dt                  = 0.1
    N                   = 1000
    beta_integral       = lambda t: t 
    # Sampling
    sample_size         = 5
    exact_logp          = False
    ode_sample          = True
    eu_sample           = True
    # Optimisation hyperparameters
    start_step          = 0
    n_steps             = 1_000_000
    lr                  = 1e-4
    batch_size          = 512 #256 # 256 with UNet
    print_every         = 1_000
    opt                 = "adabelief"
    # Other
    cmap                = None