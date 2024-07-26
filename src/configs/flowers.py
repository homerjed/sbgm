class FlowersConfig:
    seed                = 0
    # Data
    dataset_name        = "flowers" 
    n_pix               = 64 
    # Model
    model_type: str     = "UNet"
    model_args          = dict(
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=256,
        heads=2, 
        dim_head=64,
        dropout_rate=0.3,
        num_res_blocks=2,
        attn_resolutions=[8, 32, 64]
    )
    use_ema             = False
    # SDE
    t1                  = 8.
    t0                  = 1e-5 
    dt                  = 0.1
    beta_integral       = lambda t: t 
    # sde: SDE            = VPSDE(beta_integral, dt=dt, t0=t0, t1=t1)
    # Sampling
    sample_size         = 5
    exact_logp          = False
    ode_sample          = True
    eu_sample           = True
    # Optimisation hyperparameters
    start_step          = 0
    n_steps             = 1_000_000
    lr                  = 1e-4
    batch_size          = 64 #128 #256
    print_every         = 1_000
    opt                 = "adabelief"
    # Other
    cmap                = None