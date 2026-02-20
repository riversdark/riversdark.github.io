+++
title = "Eight Schools, or the importance of model reparameterisation"
author = ["olivier"]
tags = ["reparam"]
categories = ["Bayesian", "NumPyro"]
date = 2022-06-22
draft = false
showtoc = true
tocopen = true
math = true
+++

This is an introduction to NumPyro, using the Eight Schools model as example.

Here we demonstrate the effects of model reparameterisation.
Reparameterisation is especially important in hierarchical models, where
the joint density tend to have high curvatures.

<a id="code-snippet--import"></a>
```python
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive

rng_key = random.PRNGKey(0)
```

Here we are using the classic eight schools dataset from Gelman et
al. We have collected the test score statistics for eight schools,
including the mean and standard error. The goal, is to determine whether
some schools have done better than others. Note that since we are
working with the mean and standard error of eight different schools, we
are actually modeling the statistical analysis resutls of some other
people: this is essentially a meta analysis problem.

<a id="code-snippet--data"></a>
```python
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

Visualize the data.

<a id="code-snippet--data-vis"></a>
```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_theme()
sns.set_palette("Set2")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(-50, 70, 1000)
for mean, sd in zip(y, sigma):
    ax.plot(x, norm.pdf(x, mean, sd))

plt.title('Test Score from Eight Schools')
```

{{< figure src="/ox-hugo/e9d3417662a95b7aac6ceef284da58183a443c88.png" >}}


## The baseline model {#the-baseline-model}

The model we are building is a standard hierarchical model. We assume
that the observed school means represent their true mean \\(\theta\_n\\), but
corrupted with some Normal noise, and the true means are themselves
drawn from another district level distribution, with mean \\(\mu\\) and
standard deviation \\(\tau\\), which are also modeled with suitable
distributions. Essentially it's Gaussian all the way up, and we have
three different levels to consider: student, school, and the whole
district level population.

\[
\begin{align*}
y_n &\sim \text{N} (\theta_n, \sigma_n) \\
\theta_n &\sim \text{N} (\mu, \tau) \\
\mu &\sim \text{N} (0, 5) \\
\tau &\sim \text{HalfCauchy} (5).
\end{align*}
\]

In NumPyro models are coded as functions.

<a id="code-snippet--baseline-model"></a>
```python
def es_0(J, sigma):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))

    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma))
```

Note that the code and the mathematical model are almost identical,
except that they go in different directions. In the mathematical model,
we start with the observed data, and reason backward to determine how
they might be generated. In the code we start with the hyperparameters,
and move forward to generate the observed data.

`J` and `sigma` are data we used to build our model, but they are not
part of the model, in the sense that they are not assigned any
probability distribution. `y`, on the other hand, is the central
variable of the model, and is named `obs` in the model.

`numpyro.plate` is used to denote that the variables inside the plate
are conditionally independent. Probability distributions are the
building blocks of Bayesian models, and NumPyro has a lot of them. In
NunPyro, probability distributions are wrappers of JAX random number
generators, and they are translated into sampling statements using the
`numpyro.sample` primitive.

However `numpyro.sample` is used to define the model, not to draw
samples from the distribution. To actually draw samples from a
distribution, we use `numpyro.infer.Predictive`. In numpyro each model
defines a joint distribution, and since it's a probability distribution,
we can draw samples from it. And since we haven't conditioned on any
data, the samples we draw are from the prior distribution.

Sampling directly from the prior distribution, and inspect the samples,
is a good way to check if the model is correctly defined.

```python
es_prior_predictive_0 = Predictive(es_0, num_samples=1000)
es_prior_samples_0 = es_prior_predictive_0(rng_key, J, sigma)

def print_stats(name, samples):
    print(f"Variable: {name}")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {jnp.mean(samples, axis=0)}")
    print(f"  Variance: {jnp.var(samples, axis=0)}\n")

# Print statistics for each variable
print_stats('mu', es_prior_samples_0['mu'])
print_stats('tau', es_prior_samples_0['tau'])
print_stats('theta', es_prior_samples_0['theta'])
print_stats('obs', es_prior_samples_0['obs'])
```

```text
Variable: mu
  Shape: (1000,)
  Mean: 0.07919252663850784
  Variance: 25.190040588378906

Variable: tau
  Shape: (1000,)
  Mean: 18.472793579101562
  Variance: 4237.69921875

Variable: theta
  Shape: (1000, 8)
  Mean: [ 0.9881359   0.5466191  -3.216042    2.8044345  -1.2074522   2.4413567
  4.238887    0.55094534]
  Variance: [ 5404.2183  2688.3088  4958.9756  2792.266   2839.972   6843.413
 23103.627   1778.8481]

Variable: obs
  Shape: (1000, 8)
  Mean: [ 0.3910051  0.649485  -3.0550027  3.0970583 -1.112252   2.3252409
  4.4084034  1.3089797]
  Variance: [ 5565.5737  2790.3027  5192.405   2918.936   2898.9036  6917.7285
 23114.23    2121.8855]
```

When the samples of some variables are observed, we can condition on
these observations, and infer the conditional distributions of the other
variables. This process is called inference, and is commonly done using
MCMC methods. The conditioning is done using the
`numpyro.handlers.condition` primitive, by feeding it a data dict and
the model.

Since we have a GPU available, we will also configure the MCMC sampler
to use vectorized chains.

<a id="code-snippet--baseline-inf"></a>
```python
from numpyro.handlers import condition

mcmc_args = {
    'num_warmup': 1000,
    'num_samples': 5000,
    'num_chains': 8,
    'progress_bar': True,
    'chain_method': 'vectorized'
}

es_conditioned_0 = condition(es_0, data={'obs': y})
es_nuts_0 = NUTS(es_conditioned_0)
es_mcmc_0 = MCMC(es_nuts_0, **mcmc_args)

es_mcmc_0.run(rng_key, J, sigma)
```

```text
sample: 100% 6000/6000 [00:18<00:00, 328.12it/s]
```

The NUTS sampler is a variant of the Hamiltonian Monte Carlo (HMC)
sampler, which is a powerful tool for sampling from complex probability
distributions. HMC also comes with its own set of diagnostics, which can
be used to check the convergence of the Markov Chain. The most important
ones are the effective sample size and the Gelman-Rubin statistic, which
is a measure of the convergence of the Markov Chain.

```python
es_mcmc_0.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.46      3.30      4.55     -0.72     10.12   2801.92      1.00
       tau      4.11      3.21      3.26      0.49      8.25   1534.54      1.01
  theta[0]      6.61      5.83      6.05     -2.52     15.59   6268.80      1.00
  theta[1]      5.04      4.86      5.00     -2.70     13.15   6283.30      1.00
  theta[2]      3.90      5.57      4.24     -4.87     12.66   6632.62      1.00
  theta[3]      4.90      4.98      4.91     -3.12     13.02   6666.61      1.00
  theta[4]      3.53      4.85      3.89     -4.25     11.55   4699.78      1.00
  theta[5]      4.04      5.04      4.38     -3.89     12.33   5403.89      1.00
  theta[6]      6.63      5.25      6.09     -1.70     15.17   5692.19      1.00
  theta[7]      4.98      5.59      4.99     -4.08     13.58   7813.93      1.00

Number of divergences: 1169
```

The effective sample size for \\(\tau\\) is low, and judging from the
`r_hat` value, the Markov Chain might not have converged. Also, the
large number of divergences is a sign that the model might not be well
specified. Here we are looking at a prominent problem in hierarchical
modeling, known as Radford's funnel, where the posterior distribution
has a very sharp peak at the center, and a long tail, and the curvature
of the distribution is very high. This seriously hinders the performance
of HMC, and the divergences are a sign that the sampler is having
trouble exploring the space.

However, the issue can be readily rectified using a non-centered
parameterization.


## Manual reparameterisation {#manual-reparameterisation}

The remedy we are proposing is quite simple: replacing

$$
\theta_n \sim \text{N} (\mu, \tau)
$$

with

\[
\begin{align*}
\theta_n &= \mu + \tau \theta_0 \\
\theta_0 &\sim \text{N} (0, 1).
\end{align*}
\]

In essence, instead of drawing from a Normal distribution whose
parameters are themselves variables in the model, we draw from the unit
Normal distribution, and transform it to get the variable we want. By
doing so we untangled the sampling process of \\(\theta\\) from that of
\\(\mu\\) and \\(\tau\\).

<a id="code-snippet--manual"></a>
```python
def es_1(J, sigma):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))

    with numpyro.plate('J', J):
        theta_0 = numpyro.sample('theta_0', dist.Normal(0, 1))
        theta = numpyro.deterministic('theta', mu + theta_0 * tau)
        numpyro.sample('obs', dist.Normal(theta, sigma))
```

Here we use another primitive, `numpyro.deterministic`, to register the
transformed variable, so that its values can be stored and used later.

```python
es_prior_predictive_1 = Predictive(es_1, num_samples=1000)
es_prior_samples_1 = es_prior_predictive_1(rng_key, J, sigma)

print_stats('theta_0', es_prior_samples_1['theta_0'])
```

```text
Variable: theta_0
  Shape: (1000, 8)
  Mean: [ 0.03691418  0.02472821  0.01554041  0.03054582 -0.01188057  0.00954548
 -0.01233995  0.03313057]
  Variance: [1.0163321 1.0030503 0.9524028 0.9750142 1.0098913 0.9325964 1.0606602
 1.0366122]
```

we can see the mean and variance are indeed quite close to that of the unit normal.

Condition on the observed data and do inference.

<a id="code-snippet--manual-inf"></a>
```python
es_conditioned_1 = condition(es_1, data={'obs': y})
es_nuts_1 = NUTS(es_conditioned_1)
es_mcmc_1 = MCMC(es_nuts_1, **mcmc_args)

es_mcmc_1.run(rng_key, J, sigma)
es_mcmc_1.print_summary(exclude_deterministic=False)
```

```text
sample: 100% 6000/6000 [00:10<00:00, 546.18it/s]
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.37      3.32      4.39     -1.12      9.79  36259.41      1.00
       tau      3.61      3.20      2.78      0.00      7.86  29127.09      1.00
  theta[0]      6.21      5.57      5.66     -2.33     14.76  37206.26      1.00
  theta[1]      4.93      4.69      4.85     -2.45     12.63  45631.36      1.00
  theta[2]      3.89      5.30      4.13     -4.26     12.39  37622.30      1.00
  theta[3]      4.77      4.79      4.72     -2.97     12.43  41713.54      1.00
  theta[4]      3.60      4.68      3.85     -3.90     11.11  43218.27      1.00
  theta[5]      4.01      4.85      4.20     -3.50     11.99  42052.70      1.00
  theta[6]      6.29      5.07      5.78     -1.80     14.32  41045.35      1.00
  theta[7]      4.87      5.34      4.76     -3.54     13.29  38723.13      1.00
theta_0[0]      0.32      0.99      0.34     -1.33      1.92  42168.81      1.00
theta_0[1]      0.10      0.93      0.11     -1.45      1.60  47772.91      1.00
theta_0[2]     -0.08      0.97     -0.09     -1.67      1.50  46935.15      1.00
theta_0[3]      0.07      0.94      0.07     -1.49      1.61  46038.02      1.00
theta_0[4]     -0.16      0.93     -0.16     -1.69      1.39  46670.00      1.00
theta_0[5]     -0.07      0.94     -0.07     -1.56      1.55  44819.96      1.00
theta_0[6]      0.36      0.96      0.38     -1.30      1.88  41323.50      1.00
theta_0[7]      0.08      0.99      0.08     -1.59      1.64  44932.38      1.00

Number of divergences: 4

```

This looks much better, both the effective sample size and the `r_hat`
have massively improved for the hyperparameter `tau`, and the number of
divergences is also much lower. However, the fact that there are still
divergences tells us that reparameterisation might improve the topology
of the posterior parameter space, but there is no guarantee that it will
completely eliminate the problem. When doing Bayesian inference,
especially with models of complex dependency relationships as in
hierarchical models, good techniques are never a sufficient replacement
for good thinking.


## Using numpyro's reparameterisation handler {#using-numpyro-s-reparameterisation-handler}

Since this reparameterisation is so widely used, it has already been
implemented in NumPyro. And since reparameterisation in general is so
important in probabilistic modelling, NumPyro has implemented a wide
suite of them.

In probabilistic modeling, although it's always a good practice to
separate modeling from inference, it's not always easy to do so. As we
have seen, how we formulate the model can have a significant impact on
the inference performance. When building the model, not only do we need
to configure the variable transformations, but we also need to inform
the inference engine how to handle these transformed variables. This is
where the `numpyro.handlers.reparam` handler comes in.

<a id="code-snippet--handler"></a>
```python
from numpyro.handlers import reparam
from numpyro.infer.reparam import TransformReparam
from numpyro.distributions.transforms import AffineTransform
from numpyro.distributions import TransformedDistribution

def es_2(J, sigma):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))

    with numpyro.plate('J', J):
        with reparam(config={'theta': TransformReparam()}):
            theta = numpyro.sample(
                'theta',
                TransformedDistribution(dist.Normal(0., 1.), AffineTransform(mu, tau)))
        numpyro.sample('obs', dist.Normal(theta, sigma))
```

The process of reparameterisation goes as follows:

1.  Start with a standard Normal distribution `dist.Normal(0., 1.)`,
2.  Transform it using the affine transformation `AffineTransform(mu, tau)`,
3.  Denote the result as a `TransformedDistribution`,
4.  Register the transformed variable `theta` using `numpyro.sample`,
5.  Inform the inference engine of the reparameterisation using `reparam`.

Proceed with prior predictive sampling.

```python
es_prior_predictive_2 = Predictive(es_2, num_samples=1000)
es_prior_samples_2 = es_prior_predictive_2(rng_key, J, sigma)

print_stats('theta', es_prior_samples_2['theta'])
```

```text
Variable: theta
  Shape: (1000, 8)
  Mean: [ 0.9881359   0.5466191  -3.216042    2.8044345  -1.2074522   2.4413567
  4.238887    0.55094534]
  Variance: [ 5404.2183  2688.3088  4958.9756  2792.266   2839.972   6843.413
 23103.627   1778.8481]
```

Condition on the observed data and do inference.

<a id="code-snippet--handler-inf"></a>
```python
es_conditioned_2 = condition(es_2, data={'obs': y})
es_nuts_2 = NUTS(es_conditioned_2)
es_mcmc_2 = MCMC(es_nuts_2, **mcmc_args)

es_mcmc_2.run(rng_key, J, sigma)
es_mcmc_2.print_summary()
```

```text
sample: 100% 6000/6000 [00:10<00:00, 550.07it/s]

                   mean       std    median      5.0%     95.0%     n_eff     r_hat
           mu      4.37      3.32      4.39     -1.12      9.79  36259.41      1.00
          tau      3.61      3.20      2.78      0.00      7.86  29127.09      1.00
theta_base[0]      0.32      0.99      0.34     -1.33      1.92  42168.81      1.00
theta_base[1]      0.10      0.93      0.11     -1.45      1.60  47772.91      1.00
theta_base[2]     -0.08      0.97     -0.09     -1.67      1.50  46935.15      1.00
theta_base[3]      0.07      0.94      0.07     -1.49      1.61  46038.02      1.00
theta_base[4]     -0.16      0.93     -0.16     -1.69      1.39  46670.00      1.00
theta_base[5]     -0.07      0.94     -0.07     -1.56      1.55  44819.96      1.00
theta_base[6]      0.36      0.96      0.38     -1.30      1.88  41323.50      1.00
theta_base[7]      0.08      0.99      0.08     -1.59      1.64  44932.38      1.00

Number of divergences: 4
```

The model reparameterised using handlers, as we can see, performs just like the manually reparameterised one, with the same mean and variance estimation, same effective number of samples, and the same number of divergences.

The results are consistent with the manual reparameterisation. It might
seem uncessarily complicated to use the reparameterisation handler in
this simple example, but in more complex models, especially those with
many layers of dependencies, the reparameterisation handler can greatly
facilitate the model building process.
