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
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Ludwig-Maximilians-Universität München, Faculty for Physics, University Observatory, Scheinerstrasse 1, München, Deustchland.
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

<!--  
    What the code does
    - Large / parallel training of big diffusion models on multiple accelerators
    - Speeding up MCMC, LFI, field-level, inverse problems
-->

<!-- The forces on stars, galaies, and dark matter under external gravitational -->
<!-- fields lead to the dynamical evolution of structures in the universe. The orbits -->
<!-- of these bodies are therefore key to understanding the formation, history, and -->
<!-- future state of galaxies. The field of "galactic dynamics," which aims to model -->
<!-- the gravitating components of galaxies to study their structure and evolution, -->
<!-- is now well-established, commonly taught, and frequently used in astronomy. -->
<!-- Aside from toy problems and demonstrations, the majority of problems require -->
<!-- efficient numerical tools, many of which require the same base code (e.g., for -->
<!-- performing numerical orbit integration). -->

# Statement of need

<!--  
    - Diffusion models are theoretically complex generative models. 
      Need fast sampling and likleihood methods built on GPU-parallel
      ODE solvers (diffrax). Subclass of energy-based generative models.

    - Given this dataset, the goal of generative modeling is to fit a model 
      to the data distribution such that we can synthesize new data points 
      at will by sampling from the distribution.

    - Significant limitations of implicit and likelihood-based ML models
      e.g. modelling normalised probability distributions, likelihood calculations
      and sampling speed. Score matching avoids this. Diffusion scales to large
      datasets of high dimension better than other approaches.

    - Score-based models have achieved SOTA results on many tasks and applications
      e.g. LDMs, ...

    - Given the new avenues of research fast and large generative models offer,
      a code that carefully implements them is valuable.

    - Memory efficiency compared to normalising flows for the same tasks (one network conditioned on 't' compared to many sub-flows + faster than CNFs)

    - implemented in JAX, equinox and diffrax

    - likelihood weighting (maximum likelihood training of SBGMs)
-->

Diffusion-based generative models are a method for density estimation and sampling from high-dimensional distributions. A sub-class of these models, score-based diffusion generatives models (SBGMs), permit exact-likelihood estimation via a change-of-variables associated with the forward diffusion process. Diffusion models allow fitting generative models to high-dimensional data in a more efficient way than normalising flows since only one neural network model parameterises the diffusion process as opposed to a stack of networks in typical normalising flow architectures.

<!-- problems in cosmology, need for SBI -->

The software we present, `sbgm`, is designed to be used by machine learning and physics researchers for fitting diffusion models with a suite of custom architectures for their tasks. These models can be fit easily with multi-accelerator training and inference within the code. Typical use cases for these kinds of generative models are emulator approaches, simulation-based inference (likelihood-free inference), field-level infrence and general inverse problems. This code allows for seemless integration of diffusion models to these applications by allowing for easy conditioning of data on parameters, classes or other data such as images.


<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`). -->

<!-- `Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->


![A diagram showing how to map data to a noise distribution (the prior) with an SDE, and reverse this SDE for generative modeling. One can also reverse the associated probability flow ODE, which yields a deterministic process that samples from the same distribution as the SDE. Both the reverse-time SDE and probability flow ODE can be obtained by estimating the score.\label{fig:sde_ode}](sde_ode.png)

# Mathematics

<!-- What is diffusion -->
Diffusion models model the reverse of a forward diffusion process on samples of data $\boldsymbol{x}$ by adding a sequence of noisy perturbations. 

<!-- What is a diffusion model -->
Score-based diffusion models model the forward diffusion process with Stochastic Differential Equations (SDEs) of the form

$$
\text{d}\boldsymbol{x} = f(\boldsymbol{x}, t)\text{d}t + g(t)\text{d}\boldsymbol{w},
$$

where $f(\boldsymbol{x}, t)$ is a vector-valued function called the drift coefficient, $g(t)$ 
is the diffusion coefficient and $\text{d}\boldsymbol{w}$ is a sample of infinitesimal noise.
The solution of a SDE is a collection of continuous random variables describing a path parameterised
by a 'time' variable $t$.

The SDE itself is formulated by design and existing options include the variance exploding (VE), 
variance preserving (VP) and sub-variance preserving (SubVP). These equations describe how the mean 
and covariances of the distributions of noise added to the data evolve with time.


the reverse of the SDE, mapping from multivariate Gaussian samples to data, is of the form

$$
\text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t) - g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t + g(t)\text{d}\boldsymbol{w},
$$

where the score function $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$ is substituted with a neural network $\boldsymbol{s}_{\theta}(\boldsymbol{x}(t), t)$ for the sampling process. This network predicts the noise added to the image at time $t$ with the forward diffusion process, in accordance with the SDE, and removes it. This defines the sampling chain for a diffusion model.

The parameters of the network $\theta$ are fit via stochastic gradient descent of the score-matching loss 

$$
    \mathbb{E}_{t\sim\mathcal{U}(0, T)}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[\lambda(t)||\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) - \boldsymbol{s}_{\theta}(\boldsymbol{x},t)||_2^2]
$$

where $\lambda(t)$ is an arbitrary scalar weighting function, chosen to weight certain times - usually near $t=0$ where the data has only a small amount of noise added.

In Figure \autoref{fig:sde_ode} the forward and reverse diffusion processes are shown for a toy problem with their corresponding SDE and ODE paths.

The reverse SDE may be solved with Euler-Murayama sampling (or other annealed Langevin sampling methods) which is featured in the code. 

However, many of the applications of generative models depend on being able to calculate the likelihood of data. In [1] it is shown that any SDE may be converted into an ordinary differential equation (ODE) without changing the distributions, defined by the SDE, from which the noise is sampled from in the diffusion process. This ODE is known as the probability flow ODE and is written

$$
    \text{d}\boldsymbol{x} = [f(\boldsymbol{x}, t) - g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})]\text{d}t.
$$

This ODE can be solved with an initial-value problem that maps a prior 
sample from a multivariate Gaussian to the data distribution. This inherits the formalism of continuous normalising flows without the expensive ODE simulations used to train these flows. The likelihood estimate under a score-based diffusion model is estimated by solving the change-of-variables equation for continuous normalising flows. The code implements these calculations also for the Hutchinson trace estimation method that reduces the computational expense of the estimate. 

<!--  Controllable generation Yang Song? -->


<!-- Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$ -->

<!-- Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We thank the developers of these packages for their work and for making their code available to the community.

# References