import jax.random as jr
import optax

img_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/sgm_with_sde_lib/imgs/"
exp_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/sgm_with_sde_lib/exps/"


class CIFAR10Config:
    key                 = jr.PRNGKey(0)
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
    batch_size          = 512 #256 # 256 with UNet
    print_every         = 1_000
    opt                 = optax.adabelief(lr)
    # Other
    cmap                = None
    img_dir             = img_dir
    exp_dir             = exp_dir 