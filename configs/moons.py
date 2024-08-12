import jax 

class MoonsConfig:
    seed                = 0
    # Data
    dataset_name        = "moons" 
    # Model
    model_type: str     = "mlp"
    model_args          = dict(
        width_size=128,
        depth=2,
        activation=jax.nn.tanh,
        dropout_p=0.1
    )
    use_ema             = True
    # SDE
    t1                  = 4.
    t0                  = 0.
    dt                  = 0.02
    beta_integral       = lambda t: t ** 2.
    # sde: SDE            = VPSDE(beta_integral, dt=dt, t0=t0, t1=t1)
    # Sampling
    sample_size         = 64 # This gets squared...
    exact_logp          = True
    ode_sample          = True
    eu_sample           = True
    # Optimisation hyperparameters
    start_step          = 0
    n_steps             = 200_000
    lr                  = 1e-4
    batch_size          = 256 
    print_every         = 5_000
    opt                 = "adabelief" 
    # Other
    cmap                = "PiYG" 