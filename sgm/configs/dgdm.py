import jax.random as jr
import optax

img_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_with_sde_lib/imgs/"
exp_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_with_sde_lib/exps/"


class DgdmConfig:
    key                 = jr.PRNGKey(0)
    # Data
    dataset_name        = "dgdm" 
    n_pix               = 100 
    # Model
    model_type: str     = "UNetImageXY"
    model_args          = dict(
        is_biggan=False,
        dim_mults=[1, 1, 1],
        hidden_size=256,
        heads=2, 
        dim_head=64,
        dropout_rate=0.3,
        num_res_blocks=2,
        attn_resolutions=[25, 50, 100] # Assuming n_pix=100
    )
    use_ema             = False
    # SDE
    beta_integral       = lambda t: t 
    t1                  = 8.
    t0                  = 1e-5 
    dt                  = 0.1
    # Sampling
    sample_size         = 5
    exact_logp          = False
    ode_sample          = True
    eu_sample           = True
    # Optimisation hyperparameters
    start_step          = 0
    n_steps             = 1_000_000
    lr                  = 3e-5
    batch_size          = 32 #64 #256 
    print_every         = 1_000
    opt                 = optax.adabelief(lr)
    # Other
    cmap                = "cividis" 
    img_dir             = img_dir
    exp_dir             = exp_dir 