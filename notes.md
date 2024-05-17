# Notes on Diffusion Models

Given observed samples $x$ from a distribution of interest, the goal of a **generative model** is to learn to model its true data distribution $p(x)$.

## Types of Generative Models

- **GANs**: model the sampling procedure of a complex distribution (in a adversarial manner).
- **Likelihood-based** models seek to assign a high likelihood to the observed data samples. There are autoregressive models, normalizing flows, and Variational Autoencoders (VAEs).
- **Energy-based** models learn an arbitrarily flexible energy function which is then normalized.
- **Score-based** models learn the score function of the energy-based model and then integrate it to get the energy function.

**Diffusion models** work as both **likelihood-based** and **score-based** models.


## Mathematical Formulation

Let $x$ be a set of obeserved samples and $z$ be the unseen latent variable, which generates $x$. We can imagine that both are modeled by joint distribution $p(x, z)$. Two ways to manipulate this joint distribution to recover the likelihood $p(x)$ are:

a. **Marginalization**:

$$
p(x) = \int p(x, z) dz
$$

b. **Apply the Chain rule of probability**:

$$
p(x) = \frac{p(x, z)}{p(z | x)}
$$

But this means we must either integrate out all latent $z$, which is intractable in most cases, or need to have access to a ground truth latent encoder $p(z | x)$.

This however gives us a proxy objective to optimize the log likelihoood of the observed data **Evidence Lower Bound**:

$$
\mathbb{E}_{q_\Phi(z|x)} \left[ \log \frac{p(x,z)}{q_\Phi(z|x)} \right] \leq \log p(x)
$$

where $q_\Phi(z|x)$ is the approximate posterior distribution of the latent variable $z$ given $x$, parameterized by $\Phi$.

## Variational Autoencoders (VAEs)

Directly maximize the ELBO, by optimizing for the best $q_\Phi(z|x)$. Without delving too deep this approach can be generalized to Hierarchical VAEs, which can model multiple levels of latent variables.