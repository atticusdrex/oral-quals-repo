# Multi-Fidelity and Gaussian Process Regression Toolbox 

This repository contains the computational artifact developed for the paper Hyperkriging: Non-Markovian Multi-Fidelity Regression. It includes two main Python packages:

**Gaussian Process Regression Package**

The Gaussian Process (GP) package is built using jax, allowing high-performance, array-based computations on both CPUs and GPUs. This package supports custom user-defined kernels, training, online prediction and kernel parameter optimization using the ADAM optimization algorithm. 

**Multi-Fidelity Toolbox Package**

This package is designed to implement and compare several surrogate modeling techniques for approximating expensive high-fidelity functions using cheaper, lower-fidelity approximations. Multi-fidelity regression combines sparse, expensive high-fidelity data with abundant low-fidelity data to construct a surrogate model that is both accurate and computationally efficient. This repository implements several methods:

* Kriging: Single-fidelity Gaussian Process Regression.
* AR1: First-order autoregressive method.
* NARGP: Nonlinear autoregressive Gaussian Process.
* Hyperkriging: A novel method that incorporates all low-fidelity function evaluations as features to predict the high-fidelity output.

