<h1 align='center'>sbgm</h1>
<h2 align='center'>Score-Based Diffusion Models in JAX</h2>

Implementation and extension of [Score-Based Generative Modeling through Stochastic Differential Equations (Song++20)](https://arxiv.org/abs/2011.13456) and [Maximum Likelihood Training of Score-Based Diffusion Models (Song++21)](https://arxiv.org/abs/2101.09258) in `jax` and `equinox`. 

This repository provides a lightweight library of models, sampling and likelihood routines. Suitable for likelihood-free or emulation based approaches. Tested and typed code to ensure reliable and benchmarkable training and inference.

> [!WARNING]
> :building_construction: Note this repository is under construction, expect changes. :building_construction:

### Score-based diffusion models

Diffusion models are deep hierarchical models for data that use neural networks to model the reverse of a diffusion process that adds a sequence of noise perturbations to the data. 

Modern cutting-edge diffusion models (see citations) express both the forward and reverse diffusion processes as a Stochastic Differential Equation (SDE).

-----

<p align="center">
  <img src="https://github.com/homerjed/sbgm/blob/main/assets/sde_ode.png" />
</p>

*A diagram showing how to map data to a noise distribution (the prior) with an SDE, and reverse this SDE for generative modeling. One can also reverse the associated probability flow ODE, which yields a deterministic process that samples from the same distribution as the SDE. Both the reverse-time SDE and probability flow ODE can be obtained by estimating the score.* 
<!-- $\nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}_t)$ -->

-----

For any SDE of the form 

$$
\text{d}\boldsymbol{x} = f(\boldsymbol{x}, t)\text{d}t + g(t)\text{d}\boldsymbol{w},
$$

the reverse of the SDE from noise to data is given by 

$$
\text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t) - g(t)^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t + g(t)\text{d}\boldsymbol{w}.
$$

For every SDE there exists an associated ordinary differential equation (ODE)

$$
\text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t)\text{d}t - \frac{1}{2}g(t)^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t,
$$

where the trajectories of the SDE and ODE have the same marginal PDFs $p_t(\boldsymbol{x})$.

The Stein score of the marginal probability distributions over $t$ is approximated with a neural network $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\approx s_{\theta}(\boldsymbol{x}(t), t)$. The parameters of the neural network are fit by minimising the score-matching loss.

### Computing log-likelihoods with diffusion models

For each SDE there exists a deterministic ODE with marginal likelihoods $p_t(\boldsymbol{x})$ that match the SDE for all time $t$

$$
\text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t)\text{d}t - \frac{1}{2}g(t)^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t = f'(\boldsymbol{x}(t), t)\text{d}t.
$$

The continuous normalizing flow formalism allows the ODE to be expressed as

$$
\frac{\partial}{\partial t} \log p(\boldsymbol{x}(t)) = \nabla_{\boldsymbol{x}} \cdot f'(\boldsymbol{x}(t), t),
$$

which gives the log-likelihood of a datapoint $\boldsymbol{x}$ as 

$$
\log p(\boldsymbol{x}(0)) = \log p(\boldsymbol{x}(T)) + \int_{t=0}^{t=T}\text{d}t \; \nabla_{\boldsymbol{x}}\cdot f'(\boldsymbol{x}, t).
$$

Note that maximum-likelihood training is prohibitively expensive for SDE based diffusion models.

### Usage

Install via

```
pip install sbgm
```

and for the [examples](https://github.com/homerjed/sbgm/tree/main/examples), run 

```
pip install .[examples] 
```

To fit a diffusion model to the `cifar10` image dataset, try something like

```python
import sbgm
import data
import configs

datasets_path = "./"
root_dir = "./"

config = configs.cifar10_config()

key = jr.key(config.seed)
data_key, model_key, train_key = jr.split(key, 3)

dataset = data.cifar10(datasets_path, data_key)

sharding = sbgm.shard.get_sharding()
    
# Diffusion model 
model = sbgm.models.get_model(
    model_key, 
    config.model.model_type, 
    dataset.data_shape, 
    dataset.context_shape, 
    dataset.parameter_dim,
    config
)

# Stochastic differential equation (SDE)
sde = sbgm.sde.get_sde(config.sde)

# Fit model to dataset
model = sbgm.train.train(
    train_key,
    model,
    sde,
    dataset,
    config,
    sharding=sharding,
    save_dir=root_dir
)
```

### Features

* Parallelised exact and approximate log-likelihood calculations,
* UNet and transformer score network implementations,
* VP, SubVP and VE SDEs (neural network $\beta(t)$ and $\sigma(t)$ functions are on the list!),
* Multi-modal conditioning (basically just optional parameter and image conditioning methods),
* Checkpointing for optimiser and model,
* Multi-device training and sampling.

### Samples

> [!NOTE]
> I haven't optimised any training/architecture hyperparameters or trained long enough here, you could do a lot better. 

<h4 align='left'>Flowers</h4>

Euler-Marayama sampling
![Flowers Euler-Marayama sampling](assets/flowers_eu.png?raw=true)

ODE sampling
![Flowers ODE sampling](assets/flowers_ode.png?raw=true)

<h4 align='left'>CIFAR10</h4>

Euler-Marayama sampling
![CIFAR10 Euler-marayama sampling](assets/cifar10_eu.png?raw=true)

ODE sampling
![CIFAR10 ODE sampling](assets/cifar10_ode.png?raw=true)

<!-- ![alt text](assets/flowers_ode.png?raw=true) -->

### SDEs 
![alt text](assets/sdes.png?raw=true)

### Citations
```bibtex
@misc{song2021scorebasedgenerativemodelingstochastic,
      title={Score-Based Generative Modeling through Stochastic Differential Equations}, 
      author={Yang Song and Jascha Sohl-Dickstein and Diederik P. Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
      year={2021},
      eprint={2011.13456},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2011.13456}, 
}
```

```bibtex
@misc{song2021maximumlikelihoodtrainingscorebased,
      title={Maximum Likelihood Training of Score-Based Diffusion Models}, 
      author={Yang Song and Conor Durkan and Iain Murray and Stefano Ermon},
      year={2021},
      eprint={2101.09258},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2101.09258}, 
}
```