---
title: 'SBGM: Score-Based Generative Models in JAX.'
tags:
  - Python
  - Machine learning 
  - Generative models 
  - Diffusion models 
  - Simulation based inference
  - Emulators
authors:
  - name: Jed Homer
    orcid: 0009-0002-0985-1437
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Ludwig-Maximilians-Universität München, Faculty for Physics, University Observatory, München, Deustchland.
   index: 1
   ror: 00hx57361
date: 1 October 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Diffusion models [@diffusion; @ddpm; @sde] have emerged as the dominant paradigm for generative modelling. The advantages of accurate density estimation and high-quality samples of normalising flows [@flows; @ffjord], VAEs [@vaes] and GANs [@gans] are subsumed into this method. Significant limitations exist on implicit and neural network based likelihood models with respect to modelling normalised probability distributions and sampling speed. Score-matching diffusion models are more efficient than previous generative model algorithms for these tasks. The diffusion process is agnostic to the data representation meaning different types of data such as audio, point-clouds, videos and images can be modelled. The use of generative models, such as diffusion models, remains somewhat unexplored given the amount of research into these methods in the machine learning community. In order to bridge the gap, trusted software is needed to allow research in the natural sciences using generative models.

# Statement of need

Diffusion-based generative models [@diffusion; @ddpm] are a method for density estimation and sampling from high-dimensional distributions. A sub-class of these models, score-based diffusion generatives models (SBGMs, [@sde]), permit exact-likelihood estimation via a change-of-variables associated with the forward diffusion process [@sde_ml]. Diffusion models allow fitting generative models to high-dimensional data in a more efficient way than normalising flows since only one neural network model parameterises the diffusion process as opposed to a sequence of neural networks in typical normalising flow architectures. Whilst existing diffusion models [@ddpm, @vdms] allow for sampling, they are limited to innaccurate variational inference approaches for density estimation which limits their use for Bayesian inference. This code provides density estimation with diffusion models using GPU enabled ODE solvers in `jax` [@jax] and `diffrax` [@kidger].

<!-- problems in cosmology, need for SBI -->

The software we present, `sbgm`, is designed to be used by researchers in machine learning and the natural sciences for fitting diffusion models with a suite of custom architectures for their tasks. These models can be fit easily with multi-accelerator training and inference within the code. Typical use cases for these kinds of generative models are emulator approaches [@emulating], simulation-based inference (e.g. likelihood-free inference, @sbi:2019), field-level inference [@field_level_inference] and general inverse problems [@inverse_problem_medical; @Remy; @Feng2023; @Feng2024] (e.g. image inpainting [@sde] and denoising [@ambientdiffusion; @blinddiffusion]). This code allows for seemless integration of diffusion models to these applications by providing data-generating models with easy conditioning of the data on parameters, classifying variables or other data such as images. Furthermore, the implementation in `equinox` [@equinox] guarantees safe integration of `sbgm` with any other sampling libraries (e.g. BlackJAX @blackjax) or `jax` [@jax] based codes.

<!-- Other domains... audio etc -->

![A diagram showing how to map data to a noise distribution (the prior) with an SDE, and reverse this SDE for generative modeling. One can also reverse the associated probability flow ODE, which yields a deterministic process that samples from the same distribution as the SDE. Both the reverse-time SDE and probability flow ODE can be obtained by estimating the score.\label{fig:sde_ode}](sde_ode.png)

# Diffusion  

<!-- What is diffusion -->
Diffusion in the context of generative modelling describes the process of adding small amounts of noise sequentially to samples of data $\boldsymbol{x}$ [@diffusion]. A generative model for the data arises from training a neural network to reverse this process by subtracting the noise added to the data.

<!-- What is a diffusion model -->
Score-based diffusion models model the forward diffusion process with Stochastic Differential Equations (SDEs, @sde:2021) of the form

$$
\text{d}\boldsymbol{x} = f(\boldsymbol{x}, t)\text{d}t + g(t)\text{d}\boldsymbol{w},
$$

where $f(\boldsymbol{x}, t)$ is a vector-valued function called the drift coefficient, $g(t)$ is the diffusion coefficient and $\text{d}\boldsymbol{w}$ is a sample of noise $\text{d}\boldsymbol{w}\sim \mathcal{G}[\text{d}\boldsymbol{w}|\mathbf{0}, \mathbf{I}]$. The solution of a SDE is a collection of continuous random variables describing a path parameterised by a 'time' variable $t$. The diffusion path begins at $t=0$ and ends at $T=0$ where the resulting distribution is then a multivariate Gaussian with mean zero and covariance $\mathbf{I}$.

The SDE itself is formulated by design and existing options include the variance exploding (VE), variance preserving (VP) and sub-variance preserving (SubVP). These equations describe how the mean and covariances of the distributions of noise added to the data evolve with time.

The reverse of the SDE, mapping from multivariate Gaussian samples $\boldsymbol{x}(T)$ to samples of data $\boldsymbol{x}(0)$, is of the form

$$
\text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t) - g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t + g(t)\text{d}\boldsymbol{w},
$$

where the score function $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$ is substituted with a neural network $\boldsymbol{s}_{\theta}(\boldsymbol{x}(t), t)$ for the sampling process. This network predicts the noise added to the image at time $t$ with the forward diffusion process, in accordance with the SDE, and removes it. With a data-dimensional sample of Gaussian noise from the prior $p_T(\boldsymbol{x})$ (see \ref{fig:sde_ode}) one can reverse the diffusion process to generate data.

The score-based diffusion model for the data is fit by optimising the parameters of the network $\theta$ via stochastic gradient descent of the score-matching loss [@sde] 

$$
    \mathcal{L}(\theta) = \mathbf{E}_{t\sim\mathcal{U}(0, T)}\mathbf{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\mathbf{E}_{\boldsymbol{x}(t)\sim p(\boldsymbol{x}(t)|\boldsymbol{x})}[\lambda(t)||\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}(t)|\boldsymbol{x}(0)) - \boldsymbol{s}_{\theta}(\boldsymbol{x}(t),t)||_2^2]
$$

where $\lambda(t)$ is an arbitrary scalar weighting function, chosen to weight certain times - usually near $t=0$ where the data has only a small amount of noise added. Here, $p_t(\boldsymbol{x}(t)|\boldsymbol{x}(0))$ is the transition kernel for Gaussian diffusion paths. This is defined depending on the form of the SDE [@sde] and for the common variance-preserving (VP) SDE the kernel is written as 

$$
    p(\boldsymbol{x}(t)|\boldsymbol{x}(0)) = \mathcal{G}[\boldsymbol{x}(t)|\mu_t \cdot \boldsymbol{x}(0), \sigma^2_t \cdot \mathbf{I}]
$$

where $\mathcal{G}[\cdot]$ is a Gaussian distribution, $\mu_t=\exp(-\int_0^t\text{d}s \; \beta(s))$ and $\sigma^2_t = 1 - \mu_t$. $\beta(t)$ is typically chosen to be a simple linear function of $t$.

In Figure \ref{fig:sde_ode} the forward and reverse diffusion processes are shown for a samples from a Gaussian mixture with their corresponding SDE and ODE paths.

The reverse SDE may be solved with Euler-Murayama sampling [@sde] (or other annealed Langevin sampling methods) which is featured in the code. 

# Likelihood 

However, many of the applications of generative models depend on being able to calculate the likelihood of data. In @sde it is shown that any SDE may be converted into an ordinary differential equation (ODE) without changing the distributions, defined by the SDE, from which the noise is sampled from in the diffusion process (denoted $p_t(x)$ and shown in grey in Figure \ref{fig:sde_ode}). This ODE is known as the probability flow ODE [@sde; @sde_ml] and is written

$$
    \text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t) - g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t = f'(\boldsymbol{x}, t)\text{d}t.
$$

This ODE can be solved with an initial-value problem that maps a prior sample from a multivariate Gaussian to the data distribution. This inherits the formalism of continuous normalising flows [@neuralodes; @ffjord] without the expensive ODE simulations used to train these flows - this allows for a likelihood estimate based on diffusion models [@sde_ml].

![A diagram showing a log-likelihood calculation over the support of a Gaussian mixture model with eight components. Data is drawn (shown in red) from this mixture to train the diffusion model that gives the likelihood in gray. The log-likelihood is calculated using the ODE and a trained diffusion model. \label{fig:8gauss}](8gauss.png){ width=50% } 

The likelihood estimate under a score-based diffusion model is estimated by solving the change-of-variables equation for continuous normalising flows. 

$$
\frac{\partial}{\partial t} \log p(\boldsymbol{x}(t)) = \nabla_{\boldsymbol{x}} \cdot f(\boldsymbol{x}(t), t),
$$

which gives the log-likelihood of a single datapoint $\boldsymbol{x}(0)$ as 

$$
\log p(\boldsymbol{x}(0)) = \log p(\boldsymbol{x}(T)) + \int_{t=0}^{t=T}\text{d}t \; \nabla_{\boldsymbol{x}}\cdot f(\boldsymbol{x}, t).
$$

The code implements these calculations also for the Hutchinson trace estimation method [@ffjord] that reduces the computational expense of the estimate. Figure \ref{fig:8gauss} shows an example of a data-likelihood calculation using a trained diffusion model with the ODE associated from an SDE. It is possible to train score-based diffusion models such that the score-matching loss bounds the Kullback-Leibler divergence for each data point $\boldsymbol{x}$ against the unknown data distribution. This is shown in [@sde_ml] via a choice of $\lambda(t)$ termed the 'likelihood weighting'. It is also implemented in the code such that the score-matching bounds the KL divergence between the model and unknown data distribution per datapoint.

# Conditioning diffusion models

As shown in @batziolis it is possible to fit score-based diffusion models to a conditional distribution $p(\boldsymbol{x}|\boldsymbol{\pi}, \boldsymbol{y})$ where in typical inverse problems $\boldsymbol{y}$ would be an image and $\boldsymbol{\pi}$ a set of parameters in a physical model for the data. The code is implemented such that all training, sampling and density estimation is possible with these inputs. This allows for diffusion models to be used many different kinds of inverse problems.

<!--  Controllable generation Yang Song? -->

<!-- # Citations -->

<!-- For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Implementations and future work

Diffusion models are defined in `sbgm` via a score-network model $\boldsymbol{s}_{\theta}$ and an SDE. All the availble SDEs in the literature of score-based diffusion models are available. We provide implementations for UNet [@unet] and Residual Network [@resnet] models which are state-of-the-art for diffusion tasks. The code is compatible with any model written in the `equinox` [@equinox] framework. We are extending the code to provide transformer-based [@dits] and latent diffusion models [@ldms]. 

Our implementation allows for the organisation of projects based on save/load configuration files, model and optimiser checkpointing and utility functions for plotting and saving metrics and sampled data.

# GPU Support

`sbgm` offers easy GPU support including the use of multiple GPU devices for training and sampling within the code.

# Acknowledgements

We thank the developers of the packages `jax` [@jax], `optax` [@optax], `equinox` [@equinox] and `diffrax` [@kidger] for their work and for making their code available to the community.

# References