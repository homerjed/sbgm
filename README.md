<h1 align='center'>sgm</h1>
<h2 align='center'>Score-based Diffusion models in JAX</h2>

Implementation and extension of [Score-Based Generative Modeling through Stochastic Differential Equations (Song++20)](https://arxiv.org/abs/2011.13456) and [Maximum Likelihood Training of Score-Based Diffusion Models (Song++21)](https://arxiv.org/abs/2101.09258) in `jax` and `equinox`. 

To do:
* NN beta schedule
* Diffusion transformer

<h3 align='left'>Flowers</h3>

Euler-Marayama sampling
![alt text](figs/flowers_eu.png?raw=true)

ODE sampling
![alt text](figs/flowers_ode.png?raw=true)

<h3 align='left'>CIFAR10</h3>

Euler-Marayama sampling
![alt text](figs/cifar10_eu.png?raw=true)

ODE sampling
![alt text](figs/cifar10_ode.png?raw=true)

<!-- <p align="center">
  <img src="figs/flowers_eu.png" width="350" title="hover text">
</p> -->