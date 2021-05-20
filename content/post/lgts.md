+++
title = "Bayesian modeling of global trend, local variation, seasonality, and heterogeneity in time series"
date = 2021-05-18
tags = ["ts"]
categories = ["Stats"]
draft = true
+++


In this post we'll implement the time series models from the RGLT R package, which can be used to model global trend, local variation, and seasonalities, key features of flexible time series data modeling. 

<!--more-->
The post also covers several different features under this general framework, notably:

-   modeling over-dispersed observations with Student-t distribution
-   explicit and extensive modeling of the variation to capture data heteroscedasticity
-   adding regression components when necessary

Some other general but important features include:

-   extensive usage of visualisation for model criticism
-   prior and posterior simulation for model inspection
-   and a unified prediction framework.

But most importantly, apart from the specific features and techniques, the post gives a general procedure for Bayesian time series model building and criticism, which should be useful when building any kind of model for any kind of data set.

Different options for modeling trends and seasonalities have been considered in this post, to understand their varying capabilities and characteristics, and to prepare an arsenal for building bespoke models for future data sets with different characteristics.

The models are fitted using `numpyro`, a Bayesian inference package based on `JAX`.

<a id="code-snippet--lgts-import"></a>
```python
import numpyro
numpyro.set_host_device_count(4)
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.diagnostics import autocorrelation, hpdi
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import deterministic, sample
from numpyro.handlers import seed, condition, plate

import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns
sns.set_theme(palette='Set2')
colors = sns.color_palette()

import jax.numpy as jnp
from jax import random
from jax import lax, nn
from jax.config import config; config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=2)
rng = random.PRNGKey(123)

from rdatasets import data, descr, summary
%config InlineBackend.figure_format = 'retina'

print("Packages loaded!")
```

```text
Packages loaded!
```

We are going to use two data sets, the BJ sales data and the monthly airline passenger data set. For the BJ sales data set we have an extra lead indicator, which can be used to build a regression component to the model.

<a id="code-snippet--lgts-data"></a>
```python
bj = data("BJsales")['value']
air_passengers = data("AirPassengers")['value']
bj_lead = jnp.array(
    [10.01,10.07,10.32, 9.75,10.33,10.13,10.36,10.32,10.13,10.16,10.58,10.62, 10.86,11.20,10.74,10.56,10.48,10.77,11.33,10.96,11.16,11.70,11.39,11.42, 11.94,11.24,11.59,10.96,11.40,11.02,11.01,11.23,11.33,10.83,10.84,11.14, 10.38,10.90,11.05,11.11,11.01,11.22,11.21,11.91,11.69,10.93,10.99,11.01, 10.84,10.76,10.77,10.88,10.49,10.50,11.00,10.98,10.61,10.48,10.53,11.07, 10.61,10.86,10.34,10.78,10.80,10.33,10.44,10.50,10.75,10.40,10.40,10.34, 10.55,10.46,10.82,10.91,10.87,10.67,11.11,10.88,11.28,11.27,11.44,11.52, 12.10,11.83,12.62,12.41,12.43,12.73,13.01,12.74,12.73,12.76,12.92,12.64, 12.79,13.05,12.69,13.01,12.90,13.12,12.47,12.47,12.94,13.10,12.91,13.39, 13.13,13.34,13.34,13.14,13.49,13.87,13.39,13.59,13.27,13.70,13.20,13.32, 13.15,13.30,12.94,13.29,13.26,13.08,13.24,13.31,13.52,13.02,13.25,13.12, 13.26,13.11,13.30,13.06,13.32,13.10,13.27,13.64,13.58,13.87,13.53,13.41, 13.25,13.50,13.58,13.51,13.77,13.40])
```

These data sets have different characteristics and as such will demand different modeling considerations.


## Modeling global trend and local variation {#modeling-global-trend-and-local-variation}

The data used for this model is the [BJsales](https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BJsales.html) data set, which contains 150 sales observations. The data are bounded by zero, as can be expected for sales data, and range between 200 and 300, and there is very little variation between neighbouring points.

<a id="code-snippet--lgt-data-process"></a>
```python
lgt_y = jnp.array(bj)

plt.plot(bj, '.')
plt.title('BJ sales data');
```

{{< figure src="/ox-hugo/c3db535d3ece37a5f1c66d722f9eeccb6e0ae206.png" >}}

The first model we are building can be described as

\begin{align\*}
y\_{t} &\sim \text{T} (\nu,\mu\_{t}, \sigma\_{t}) \\\\\\
\sigma\_{t} &= \kappa \mu\_{t}^{ \tau} + \xi   \\\\\\
\mu\_{t} &= \text{G}\_{t} + \text{L}\_{t} \\\\\\
\text{G}\_{t} &= g\_{t-1} + \gamma g\_{t-1}^{ \rho } \\\\\\
\text{L}\_{t} &= \lambda l\_{t-1}  \\\\\\
g\_{t} &= \alpha y\_{t} + ( 1- \alpha ) \text{G}\_{t} \\\\\\
l\_{t} &= \beta  \left( g\_{t}-g\_{t-1} \right) + \left( 1- \beta  \right) l\_{t-1}\\\\\\
\end{align\*}

To model the outcome we'll use the Student's T distribution, which has three parameters: the degree of freedom \\(\nu\\), the expected value \\(\mu\\), and the standard deviation \\(\sigma\\).

The degree of freedom determines how heavy tailed the distribution will be: when the degree of freedom is one, the T distribution becomes Cauchy distribution, which is so heavy tailed that the mean and standard deviation can not be properly defined; when the degree of freedom approches infinity the distribution approaches Normal, with very narrow tails. Using the T distribution, we can account for heavy tail data when the data are indeed heavy tailed, but it's also flexible enough for when they are not. Since à priori we don't have much information on the degree of freedom, we'll put a wide uniform prior on it.

The expected value is where we'll put most of our modeling focus on. In this first model we'll consider modeling it as the sum of a global trend \\(\text{G}\\) and a local variation \\(\text{L}\\), with the global trend being modeled as a smoothed exponential function of the hidden level \\(g\\), and the local variation as a dampened baseline variation \\(l\\). Furthermore, both the hidden level \\(g\\) and the variation \\(l\\) follow a smoothed autoregressive process. For hidden level \\(g\\), it's a weighted average between the outcome (with local trend removed), and the previous global trend. For the baseline variation \\(l\\) it's a weighted average between the increment in hidden levels, and the previous baseline variation. \\(\alpha\\) and \\(\beta\\) are the corresponding smoothing weights.

The standard deviation is modeled with another smoothed exponential process of the expected value, so this is a heterogeneous model, and the variance grows as the expected value grows.

Because both the standard deviation \\(\sigma\\) and the expected value \\(\mu\\) follow a smoothed exponential process, we have to guarantee the expected value and the hidden level \\(g\\) be always positive. This assumption is reasonable when the data is also positive, as is the case here.

It's important to note that the models we are building might be too complex for the data we are using. Here by exploratory analysis we can see that the BJ sales data have a global trend, and there are clearly local variations, but there doesn't seem to be many sudden changes in the series, and as such the smoothed exponential process for the standard deviation might not be strictly necessar. Still, we use a more complex model to account for the **potential** heterogeneities in the data.

Now we can chose some proper priors for the model parameters and code the model in NumPyro. Of course, chosing priors is an integral part of Bayesian modeling and it is no the easy part, we need to consider our prior knowledge carefully and gradually update that in model criticism. The priors chosen for this model implementation take into account both the model structure, and the general knowledge we have gained from exploratory data analysis. In Bayesian analysis each model should be a bespoke model for the specific data set we are modeling.

Because this is a time series model, we use the first observation to initialise the model. Specifically, we'll use it to indicate the starting whereabouts of the hidden level \\(g\\).

<a id="code-snippet--lgt-model"></a>
```python
def lgt(y, future=3):
    N = y.shape[0]
    nu = sample("nu", dist.Uniform(1, 50))

    xi = sample("xi", dist.HalfNormal(2))
    tau = sample("tau", dist.Beta(1, 4))
    kappa = sample("kappa", dist.HalfNormal(2))

    gamma = sample("gamma", dist.HalfNormal(2))
    rho = sample("rho", dist.Beta(1, 4))

    lbda = sample("lbda", dist.Beta(2, 2))
    alpha = sample("alpha", dist.Beta(2, 2))
    beta = sample("beta", dist.Beta(2, 2))

    g_init = sample("g_init", dist.Normal(y[0], 5))
    l_init = sample("l_init", dist.Normal(0, 10))

    def transition_fn(carry, t):
        g, l = carry
        G = deterministic("G", g + gamma * g ** rho)
        L = deterministic("L", lbda * l)
        mu = deterministic("mu", jnp.clip(G + L, 0))
        sigma = deterministic("sigma", xi + kappa * mu ** tau)

        yt = sample("y", dist.StudentT(nu, mu, sigma))

        g_new = deterministic("g", jnp.clip(alpha * yt + (1 - alpha) * G, 0))
        l_new = deterministic("l", beta * (g_new - g) + (1 - beta) * l)

        return (g_new, l_new), yt

    with condition(data={'y':y}):
        _, ys = scan(transition_fn, (g_init, l_init),  jnp.arange(0, N+future))

    return ys

print("Observed data:", lgt_y[:4])

with seed(rng_seed=3):
    print("\nPrior sampling:")
    print(lgt(lgt_y[:1], future=3))
    print("\nFixed data sampling:")
    print(lgt(lgt_y[:4], future=0))
```

```text
Observed data: [200.1 199.5 199.4 198.9]

Prior sampling:
[200.1  200.04 200.08 201.52]

Fixed data sampling:
[200.1 199.5 199.4 198.9]
```

With the model specified we can draw prior samples from it. We can generate prior samples without the observed data, because our model is completely generative. We also write a simple helper function `check_prior` to collect model variables and print out summary statistics. The model variables are separated into two groups: global variables, which determine the behaviour of the whole model; and local variables which are specific to each observed outcome.

<a id="code-snippet--lgt-prior"></a>
```python
lgt_prior = Predictive(lgt, num_samples=500)(rng, lgt_y[:1], 139)

def check_prior(prior):
    """Print out prior shapes,
    collect local and global variables into lists."""
    lvs = []
    gvs = []
    y_length = prior['y'].shape[-1]
    for k, v in prior.items():
        if v.ndim == 1:
            gvs.append(k)
        elif v.shape[1] == y_length:
            lvs.append(k)
        else:
            gvs.append(k)

    print("Global variables:")
    for k in gvs:
        print(k, prior[k].shape, end='; ')
    print("\n\nLocal variables:")
    for k in lvs:
        print(k, prior[k].shape, end='; ')

    return lvs, gvs

lgt_lvs, lgt_gvs = check_prior(lgt_prior)
```

```text
Global variables:
alpha (500,); beta (500,); g_init (500,); gamma (500,); kappa (500,); l_init (500,); lbda (500,); nu (500,); rho (500,); tau (500,); xi (500,);

Local variables:
G (500, 140); L (500, 140); g (500, 140); l (500, 140); mu (500, 140); sigma (500, 140); y (500, 140);
```

It's important to check the model prior predictions to make sure the model indeed has the desired properties we intended for. We can use a parallel plot in which each sample will be plotted as one single line across different coordinates. While doing so, we'll also differentiate the normal samples, that is that intended samples of our model, from samples with extremely large outcomes, and also from samples with numerical overflows. The samples with numerical overflow will be in red, while those with extreme outcomes will be in blue.

Here we first write another helper function `concat_arrays` to concatenate one or two dimensional arrays stored in a dict, as is the case with our prior and posterior samples. And if the variable is multivariate, we also break down its dimensions into separate variables. This is to facilitate later visualisation.

<a id="code-snippet--concate-array"></a>
```python
def concat_arrays(dct, vs):
    nvs = []
    ds = []
    for v in vs:
        d = dct[v]
        if d.ndim == 2:
            vplus = [v+str(i) for i in range(d.shape[-1])]
            nvs += vplus
            ds.append(d)
        else:
            nvs.append(v)
            ds.append(d[:, None])

    ds = jnp.concatenate(ds, axis=1)

    return nvs, ds


nvs, ds = concat_arrays(lgt_prior, ['gamma', 'rho'])
print(len(nvs), ds.shape)  # 1+1=2, (500, 2)

nvs, ds = concat_arrays(lgt_prior, ['g', 'l'])
print(len(nvs), ds.shape)  # 140+140=280, (500, 280)

nvs, ds = concat_arrays(lgt_prior, ['gamma', 'l', 'rho'])
print(len(nvs), ds.shape)  # 1+140+1=142, (500, 142)
```

```text
2 (500, 2)
280 (500, 280)
142 (500, 142)
```

And plot the global parameters of the model

<a id="code-snippet--plot-parallel-global-variables"></a>
```python
def compare_global_params(dct, vs, v_ref='y', val_ref=10000):
    # TODO add custom labels to different groups

    nans = jnp.any(jnp.isnan(dct[v_ref]), axis=1)
    extremes = jnp.any(dct[v_ref] > val_ref, axis=1)
    print('Number of samples with numerical overflows: ', nans.sum())
    print('Number of samples with extreme values : ', extremes.sum())

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    nvs, ds = concat_arrays(dct, vs)

    for i in range(ds.shape[0]):
        if not nans[i]:
            if not extremes[i]:
                ax.plot(ds[i], color=colors[0], alpha=0.1, lw=1)

    for i in range(ds.shape[0]):
        if not nans[i]:
            if extremes[i]:
                ax.plot(ds[i], color='r', alpha=0.2, lw=1)

    for i in range(ds.shape[0]):
        if nans[i]:
            ax.plot(ds[i], color='k', alpha=0.2, lw=1)

    ax.set_xticks(range(len(nvs)))
    ax.set_xticklabels(nvs, rotation=30)
    plt.tight_layout();

compare_global_params(lgt_prior, lgt_gvs)
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  48
```

{{< figure src="/ox-hugo/f36cdeaba76e91047066dda57566f14836fcc102.png" >}}

Because different variables have different scales, to get a better idea of some certain variables we can limit the list of variables to be plotted:

```python
compare_global_params(lgt_prior, ['rho', 'tau', 'gamma', 'kappa', 'lbda'])
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  48
```

{{< figure src="/ox-hugo/06995b099e8370d091883d06fff45bedb3e0919c.png" >}}

Even though we have consciously limited the prior distribution for \\(\rho\\) to favour smaller values, when its values are still relatively large, we'are likely to see samples with extreme outcomes. This is understandable because \\(\rho\\) decides how fast the exponential grows.

We can also plot the distribution of one single variable of interest:

```python
sns.histplot(lgt_prior['rho'], bins=30, stat='probability');
```

{{< figure src="/ox-hugo/8129883de10c3328bc042ec5ea6b84706f0ca6ef.png" >}}

This corresponds to the samples at coordinate `rho` in the previous plot. A priori the expected value of \\(\rho\\) is 0.2, but as we can see, that are many samples will buch bigger values. We can try to change the priors for these parameters to limit their behavoir, but the problem here doesn't seem very serious so we'll leave them be for now.

We now plot the model outcome, and other local variables to check the implications of the model priors.

<a id="code-snippet--plot-local-variables"></a>
```python
def plot_locals(dct, vs, v_ref='y', val_ref=5000, **plt_kws):

    extremes = jnp.any(dct[v_ref] > val_ref, axis=1)
    print('Number of samples with extreme values : ', extremes.sum())

    fig, axes = plt.subplots(**plt_kws)
    axes = axes.flatten()
    for i, v in enumerate(vs):
        data = dct[v]
        for k in range(data.shape[0]):
            if not extremes[k]:
                axes[i].plot(data[k], color=colors[0], alpha=0.2)
        axes[i].set_title(v)
    plt.tight_layout();

plot_locals(lgt_prior, lgt_lvs, nrows=4, ncols=2, figsize=(8, 8))
```

```text
Number of samples with extreme values :  72
```

{{< figure src="/ox-hugo/8c6ac64f9324fbd3fb1307e72336395e78b258b7.png" >}}

We can see that the outcome generally follow the smoothed exponential process, with some cases the outcome growing faster than other cases. Also notice that the local variation is roughly one to two orders of magnitudes smaller than the global trend. This is intended because we want to to able to properly identify them.

We can also generate prior samples conditioned on the observed data. Since time series models are Markovian, i.e. path dependent, the observed data will serve to limit some of the model parameters that depend on the outcome to a region in the ambient space that is compatible with the observed data.

<a id="code-snippet--lgt-prior-with-observation"></a>
```python
lgt_prior_fixed = Predictive(lgt, num_samples=500)(rng, lgt_y[:140], 0)

plot_locals(lgt_prior_fixed, lgt_lvs, v_ref='mu', val_ref=500,
            nrows=4, ncols=2, figsize=(8,8))
```

```text
Number of samples with extreme values :  6
```

{{< figure src="/ox-hugo/1f2ab2d1f33367ba5cd843a25af49a8f5771e0fe.png" >}}

This might be useful for understanding the general interactions between the model and the data, and also for comparison with the posterior, to see how much we have learned.

To test that our inference engine and the model works properly, We can then chose one of our prior samples, feed it to our inference engine to see if we can recover the parameters used to generate them. It's important to realise that we are not always able to recover the parameters, because as we have see in the prior check, there are some highly unregular samples.

First chose one sample

<a id="code-snippet--lgt-fake-data"></a>
```python
n = 14
y = lgt_prior['y'][n]

plt.plot(y)
plt.title('Fake outcome no. {} from prior'.format(n));
```

{{< figure src="/ox-hugo/88d78e652367e4e961c7a9703a8302562cf72a7f.png" >}}

The sample seems ordinary enough, not one of the extreme ones. We then run inference on it and compare the posterior samples with the real parameter value that we used to generate the outcome.

<a id="code-snippet--lgt-fake-inference"></a>
```python
kernel = NUTS(lgt)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=1)
mcmc.run(rng, y, future=0)
mcmc.print_summary()
```

```text
sample: 100%|██████████| 3000/3000 [00:19<00:00, 156.16it/s, 15 steps of size 9.66e-02. acc. prob=0.94]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.36      0.06      0.35      0.26      0.46   1301.22      1.00
      beta      0.44      0.21      0.41      0.11      0.77   1823.17      1.00
    g_init    200.33      4.98    200.46    192.50    208.45   1660.66      1.00
     gamma      1.80      1.16      1.65      0.01      3.32   1831.46      1.00
     kappa      2.67      1.10      2.55      0.91      4.28    932.95      1.00
    l_init      0.33      9.47      0.53    -15.43     15.54   1545.74      1.00
      lbda      0.45      0.22      0.45      0.07      0.78   1519.01      1.00
        nu     30.52     11.69     31.23     13.31     49.64   1847.32      1.00
       rho      0.15      0.11      0.13      0.00      0.31   1351.82      1.00
       tau      0.63      0.07      0.62      0.52      0.75    744.34      1.00
        xi      1.60      1.21      1.36      0.00      3.34   1670.03      1.00

Number of divergences: 0
```

Let's visualise the parameter posterior.

<a id="code-snippet--lgt-fake-parameter-recovery"></a>
```python
def check_inference(mcmc, prior, vs, n):
    refs = []
    for v in vs:
        val = prior[v][n]
        if val.ndim == 0:
            refs.append(val.item())
        else:
            refs += list(val)

    d = az.from_numpyro(mcmc)
    az.plot_posterior(d, var_names=vs, ref_val=refs,
                      point_estimate=None, hdi_prob='hide', figsize=(8, 8));

check_inference(mcmc, lgt_prior, lgt_gvs, n=14)
```

{{< figure src="/ox-hugo/6d0ffbb1dab11ffdba5cc5c8663177d84d1cc85f.png" >}}

The recovered parameters are well in the range of possibilities, but there are large uncertainties around them. This is to be expected: when building complex models, there is  only so much we can learn from the limited data. We move on to the inference on real data.

<a id="code-snippet--lgt-inference"></a>
```python
lgt_kernel = NUTS(lgt, target_accept_prob=0.95)
lgt_mcmc = MCMC(lgt_kernel,
                num_warmup=2000, num_samples=2000, num_chains=4, progress_bar=False)
lgt_mcmc.run(rng, lgt_y[:-10], future=0)
lgt_mcmc.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.84      0.07      0.84      0.72      0.96   4341.04      1.00
      beta      0.59      0.14      0.58      0.36      0.82   5120.06      1.00
    g_init    201.00      1.99    200.98    197.75    204.22   4283.96      1.00
     gamma      0.12      0.08      0.11      0.00      0.23   6171.05      1.00
     kappa      0.43      0.31      0.37      0.00      0.89   4396.55      1.00
    l_init     -1.68      2.96     -1.55     -6.40      3.02   3892.23      1.00
      lbda      0.70      0.13      0.70      0.49      0.93   5001.75      1.00
        nu     27.32     12.94     26.81      9.90     49.96   6098.62      1.00
       rho      0.11      0.10      0.08      0.00      0.25   5451.25      1.00
       tau      0.10      0.10      0.07      0.00      0.23   6201.72      1.00
        xi      0.65      0.38      0.66      0.00      1.17   4259.09      1.00

Number of divergences: 2
```

Now we can make predictions for future periods.

<a id="code-snippet--lgt-posterior-prediction"></a>
```python
lgt_sample = lgt_mcmc.get_samples()
lgt_post = Predictive(lgt, posterior_samples=lgt_sample)(rng, lgt_y[:140], 10)

print("posterior samples:")
for k, v in lgt_sample.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in lgt_post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (8000, 140); L (8000, 140); alpha (8000,); beta (8000,); g (8000, 140); g_init (8000,); gamma (8000,); kappa (8000,); l (8000, 140); l_init (8000,); lbda (8000,); mu (8000, 140); nu (8000,); rho (8000,); sigma (8000, 140); tau (8000,); xi (8000,);

posterior Prediction samples:
G (8000, 150); L (8000, 150); g (8000, 150); l (8000, 150); mu (8000, 150); sigma (8000, 150); y (8000, 150);
```

And plot the predictions. We'll plot the mean, along with 5th, 25th, 75th, and 95th prediction distribution percentiles.

<a id="code-snippet--plot-posterior-prediction"></a>
```python
def plot_post(y, post, vs, nrow=3, ncol=2, size=(8, 8)):
    fig, axes = plt.subplots(nrow, ncol, figsize=size)
    axes = axes.flatten()
    for i, v in enumerate(vs):
        if v == 'y':
            axes[i].plot(y, color=colors[0], lw=1);

        mean = post[v].mean(0)
        p5 = jnp.percentile(post[v], 5, axis=0)
        p25 = jnp.percentile(post[v], 25, axis=0)
        p75 = jnp.percentile(post[v], 75, axis=0)
        p95 = jnp.percentile(post[v], 95, axis=0)
        sd = post[v].std(0)
        axes[i].fill_between(range(len(mean)), p5, p95,
                         color=colors[4], alpha=0.2)
        axes[i].fill_between(range(len(mean)), p25, p75,
                         color=colors[1], alpha=0.4)
        axes[i].plot(mean, color=colors[2], lw=1)
        axes[i].set_title(v)
    plt.tight_layout();

plot_post(lgt_y, lgt_post, lgt_lvs, nrow=4, ncol=2)
```

{{< figure src="/ox-hugo/8959b3a3ab2988d641f2e88933e2176095cd9a81.png" >}}

We can see that the prediction is not that far off: we get the basic trend right, and the 50% credible interval does cover the observed data. But still, our model prediction is basically a linear extrapolation, the model prediction is almost entirely determined by the global trend, and the local variation in our model contributed little, if anything at all, in modifying the global behaviour. This motivates us to add some new information of locality to our model, and this is why we'll add a new regression component in the next part.

Besides, the posterior of \\(\sigma\\) is very small and almost constant, this indicates that the smoothhed exponential model for the standard deviation is entirely redandent, we might as well just use a simple homogeneous formulation.

Apart from visualising the posterior prediction, there are also many different scores we can check. The first one is the [symmetric mean absolute percentage error](https://en.wikipedia.org/wiki/Symmetric%5Fmean%5Fabsolute%5Fpercentage%5Ferror), which is based on the percentage error of the mean prediction. Notice that the "mean" here refers to **the mean of the point estimate** across all the observations, and for the point estimate itself, we can use the mean, the median, or any other point estimate of the posterior.

<a id="code-snippet--score-smape"></a>
```python
def eval_smape(pred, truth):
    pred = jnp.mean(pred, 0)
    return 200 * jnp.mean(jnp.abs(pred - truth) / (pred + truth)).item()
```

And the mean absolute error. This time we will use the median of the posterior prediction to calculate the point estimate.

<a id="code-snippet--score-mae"></a>
```python
def eval_mae(pred, truth):
    pred = jnp.median(pred, 0)
    return jnp.mean(jnp.abs(pred - truth)).item()
```

And the root mean squared error, using sample mean as point estimate.

<a id="code-snippet--score-rmse"></a>
```python
def eval_rmse(pred, truth):
    pred = jnp.mean(pred, 0)
    return jnp.sqrt(jnp.mean(jnp.square(pred - truth))).item()
```

And finally, the [continuous ranked probability score](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf). The previous scores all use point estimates of the posterior, CRPS takes the whole posterior into consideration.

<a id="code-snippet--score-crps"></a>
```python
def eval_crps(pred, truth):
    # ref: https://github.com/pyro-ppl/pyro/pull/2045
    num_samples = pred.shape[0]
    pred = jnp.sort(pred, axis=0)
    diff = pred[1:] - pred[:-1]
    weight = jnp.arange(1, num_samples) * jnp.arange(num_samples - 1, 0, -1)
    weight = weight.reshape(weight.shape + (1,) * truth.ndim)
    crps_empirical = jnp.mean(jnp.abs(pred - truth), 0) \
        - (diff * weight).sum(axis=0) / num_samples ** 2
    return jnp.mean(crps_empirical).item()
```

Let's see how well we have done.

<a id="code-snippet--lgt-score"></a>
```python
def check_scores(post, y, n_test):
    pred = post['y'][:, -n_test:]
    ytest = y[-n_test:]
    print("sMAPE: {:.2f}%, MAE: {:.2f}, RMSE: {:.2f}, CPRS: {:.2f}".format(
        eval_smape(pred, ytest),
        eval_mae(pred, ytest),
        eval_rmse(pred, ytest),
        eval_crps(pred, ytest)))

check_scores(lgt_post, lgt_y, 10)
```

```text
sMAPE: 0.67%, MAE: 1.75, RMSE: 2.04, CPRS: 1.40
```

Of course these scores are only useful when compared with performance from other models. We can come back and compare them to other models in the later parts of the post.


## Adding a regression component to the model {#adding-a-regression-component-to-the-model}

As we've seen before, the model prediction from the previous model is far from satisfactory. Luckily with the `BJsales` data set we have another accompanying lead indicator which we can integrate into our model as a regression component. Also, since the smoothed exponential process for the standard deviation doesn't contributed much, we'll remove this model component, and directly put a prior on \\(\sigma\\).

The lead indicator

```python
plt.plot(bj_lead, '.')
plt.title('BJ sales lead indicator');
```

{{< figure src="/ox-hugo/f9da05cf923db895899fafebb3cbee5a30b47e16.png" >}}

we can see that the indicator variable follows a similar progression pattern as that of the sales data, so it should offer us useful information to improve the model prediction.

We'll use the lead indicator, lagged 3 and 4 periods, as predictors. The total length of the data, since we have to remove the first 4 data points for lack of predictors, becomes 146. We'll use the last 10 periord for prediction as before. The regression component of the model will also have an intercept.

<a id="code-snippet--lgtr-data-processing"></a>
```python
x0 = jnp.ones([146])
x1 = bj_lead[:-4]
x2 = bj_lead[1:-3]
lgtr_x = jnp.stack((x0, x1, x2), axis=-1)

lgtr_y = jnp.array(bj[4:])
print('Predictors shape:', lgtr_x.shape, '\nOutcome shape:', lgtr_y.shape)
```

```text
Predictors shape: (146, 3)
Outcome shape: (146,)
```

We change the expected value of the model so that it now consists of three components: a global trend, a local variation, and a local regression component.

Also, with increasing components for the expected value, it's getting more difficult to assign a proper prior to the hidden level \\(g\\). For this reason we give the starting level a wider Cauchy prior, and limit its value to be positive.

\begin{align\*}
y\_{t} &\sim \text{T} (\nu,\mu\_{t}, \sigma) \\\\\\
\mu\_{t} &= \text{G}\_{t} + \text{L}\_{t} + \text{R}\_{t} \\\\\\
\text{G}\_{t} &= g\_{t-1} + \gamma g\_{t-1}^{ \rho } \\\\\\
\text{L}\_{t} &= \lambda l\_{t-1}  \\\\\\
\text{R}\_{t} &= \text{x}\_{t} \eta \\\\\\
g\_{t} &= \alpha y\_{t}  + ( 1- \alpha ) \text{G}\_{t} \\\\\\
l\_{t} &= \beta  \left( g\_{t}-g\_{t-1} \right) + \left( 1- \beta  \right) l\_{t-1}\\\\\\
\end{align\*}

<a id="code-snippet--lgtr-model"></a>
```python
def lgtr(x, y):
    nu = sample("nu", dist.Uniform(2, 20))
    sigma = sample("sigma", dist.HalfNormal(2))

    gamma = sample("gamma", dist.HalfNormal(2))
    rho = sample("rho", dist.Beta(1, 4))

    lbda = sample("lbda", dist.Beta(2, 2))
    alpha = sample("alpha", dist.Beta(2, 2))
    beta = sample("beta", dist.Beta(2, 2))

    with plate("D", size=x.shape[1]):
        eta = sample("eta", dist.Normal(0, 5))

    g_init = sample("g_init", dist.TruncatedCauchy(0, y[0], 10))
    l_init = sample("l_init", dist.Normal(0, 10))

    def transition_fn(carry, xt):
        g, l = carry
        G = deterministic("G", g + gamma * g ** rho)
        L = deterministic("L", lbda * l)
        R = deterministic("R", jnp.dot(xt, eta))
        # LR = deterministic("LR", L+R)
        mu = deterministic("mu", jnp.clip(G + L + R, 0))

        yt = sample("y", dist.StudentT(nu, mu, sigma))

        g_new = deterministic("g", jnp.clip(alpha * yt + (1 - alpha) * G, 0))
        l_new = deterministic("l", beta * (g_new - g) + (1 - beta) * l)

        return (g_new, l_new), yt

    with condition(data={'y':y}):
        _, ys = scan(transition_fn, (g_init, l_init),  x)

    return ys

print("Observed data:", lgtr_y[:4])

with seed(rng_seed=3):
    print("\nPrior sampling:")
    print(lgtr(lgtr_x[:4], lgtr_y[:1]))
    print("\nFixed data sampling:")
    print(lgtr(lgtr_x[:4], lgtr_y[:4]))
```

```text
Observed data: [199.  200.2 198.6 200. ]

Prior sampling:
[199.   194.42 196.31 197.46]

Fixed data sampling:
[199.  200.2 198.6 200. ]
```

Sample from the prior

<a id="code-snippet--lgtr-prior"></a>
```python
lgtr_prior = Predictive(lgtr, num_samples=500)(rng, lgtr_x[:-10], lgtr_y[:1])

lgtr_lvs, lgtr_gvs = check_prior(lgtr_prior)
```

```text
Global variables:
alpha (500,); beta (500,); eta (500, 3); g_init (500,); gamma (500,); l_init (500,); lbda (500,); nu (500,); rho (500,); sigma (500,);

Local variables:
G (500, 136); L (500, 136); R (500, 136); g (500, 136); l (500, 136); mu (500, 136); y (500, 136);
```

First let's take a look at our trancated Cauchy prior for the initial hidden level

<a id="code-snippet--lgtr-g-init"></a>
```python
sns.histplot(lgtr_prior['g_init']);
```

{{< figure src="/ox-hugo/d0ee0e56d2580fb2d1bf34a48a60738d88494a4e.png" >}}

We can see the bulk of the samples are around 200, but the samples have quite a wide range, from zero to a few thousands, so it should be flexible enough for our model.

And check for global parameters

<a id="code-snippet--lgtr-parallel-global-variables"></a>
```python
compare_global_params(lgtr_prior, ['eta', 'gamma', 'l_init', 'sigma', ])
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  106
```

{{< figure src="/ox-hugo/68557b6266a38902d69dc2513af05743afeca6cd.png" >}}

We don't have numerical overflows, this is good. We do have more samples with very large outcomes, but this is to be expected, because we are introducing more model components to the model, and this expands our modeling space and consequently the outcome range. What matters here, is that there are no clear patterns to the extreme outcomes, which implies that the extreme outcomes are the natural result of expanding the model space, not that of some malfunctional model component.

And the prior prediction for local variables

<a id="code-snippet--lgtr-prior-local-visualisation"></a>
```python
plot_locals(lgtr_prior, lgtr_lvs, val_ref=10000,
            nrows=4, ncols=2, figsize=(8, 8))
```

```text
Number of samples with extreme values :  106
```

{{< figure src="/ox-hugo/3de7c02a5b719615cacaa0d1a58956ff7dd198ec.png" >}}

Again, we can clearly see the effect of the smoothed exponential model. We can compare it with prior predictions conditioning on the data

<a id="code-snippet--lgtr-prior-with-observation"></a>
```python
lgtr_prior_fixed = Predictive(lgtr, num_samples=500)(rng, lgtr_x[:-10], lgtr_y[:-10])

plot_locals(lgtr_prior_fixed, lgtr_lvs, v_ref='mu', val_ref=500,
            nrows=4, ncols=2, figsize=(8, 8))
```

```text
Number of samples with extreme values :  26
```

{{< figure src="/ox-hugo/6d446eaefb01e8a2badfa0db3207e0691a107868.png" >}}

The introduction of the regression component makes the relationship between the hidden level \\(l\\) and the model outcome \\(y\\) less prominent. This implies that the current model has the **potential** to improve on the previous model, since the model now has more flexibility introduced by the R component. However, we should also be aware that if the regression part does not expalin the outcome very well, it can also potentially reduce the overall predictive power of the model, as it obscures the previously dominating relationship between the outcome and the hidden level.

Like before, now we chose one sample from the prior predictive samples and do parameter recovery.

<a id="code-snippet--lgtr-fake-data"></a>
```python
n = 14
y = lgtr_prior['y'][n]
plt.plot(y)
plt.title('Fake outcome no. {} from prior'.format(n));
```

{{< figure src="/ox-hugo/4397b97446043f1a25204fa0fe7f0928c36c67cb.png" >}}

And inference.

<a id="code-snippet--lgtr-fake-parameter-recovery"></a>
```python
kernel = NUTS(lgtr)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
mcmc.run(rng, lgtr_x[:-10], y)
mcmc.print_summary()
```

```text
sample: 100%|██████████| 2000/2000 [01:14<00:00, 26.71it/s, 1023 steps of size 5.82e-03. acc. prob=0.96]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.35      0.03      0.34      0.31      0.39    507.87      1.00
      beta      0.75      0.11      0.75      0.59      0.95    580.95      1.00
    eta[0]      1.95      1.90      2.38     -0.86      4.70    397.73      1.01
    eta[1]      1.85      0.24      1.88      1.46      2.23    504.25      1.00
    eta[2]      2.34      0.20      2.34      2.02      2.67    748.84      1.00
    g_init    206.85      6.44    206.45    197.58    216.77    417.51      1.00
     gamma      0.66      0.60      0.46      0.00      1.58    436.74      1.01
    l_init      0.56      9.91      0.80    -16.14     16.56    755.45      1.00
      lbda      0.41      0.15      0.38      0.19      0.65    443.65      1.00
        nu      2.83      0.55      2.73      2.00      3.57    665.90      1.00
       rho      0.04      0.05      0.03      0.00      0.10    678.45      1.00
     sigma      0.53      0.05      0.53      0.45      0.61    722.88      1.00

Number of divergences: 0
```

Let's visualise the parameter posterior.

```python
check_inference(mcmc, lgtr_prior, lgtr_gvs, n=14)
```

{{< figure src="/ox-hugo/c9102a44130b4df58098a176165561c75a56fd3c.png" >}}

We have been able to recover most of the parameters. The posterior for \\(\nu\\) is quite far off, but considering the model and the data used, this is also understandable: this specific sample hardly has any fluctuation at all, there is no way we can learn much about the true degree of freedom.

Next we do inference on the real data.

<a id="code-snippet--lgtr-inference"></a>
```python
lgtr_kernel = NUTS(lgtr, target_accept_prob=0.95)
lgtr_mcmc = MCMC(lgtr_kernel, num_warmup=2000, num_samples=2000,
                 num_chains=4, progress_bar=False)
lgtr_mcmc.run(rng, lgtr_x[:-10], lgtr_y[:-10])
lgtr_mcmc.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.97      0.02      0.97      0.95      1.00   4497.26      1.00
      beta      0.97      0.02      0.97      0.94      1.00   7304.41      1.00
    eta[0]     -1.69      1.33     -1.41     -3.72      0.26   3400.26      1.00
    eta[1]     -4.52      0.12     -4.52     -4.72     -4.33   4233.27      1.00
    eta[2]      4.51      0.11      4.51      4.31      4.69   4188.54      1.00
    g_init    198.93      0.55    198.93    198.04    199.81   5452.33      1.00
     gamma      1.08      0.93      0.84      0.00      2.43   3680.45      1.00
    l_init     -0.22      0.92     -0.23     -1.82      1.19   5528.16      1.00
      lbda      0.74      0.03      0.73      0.68      0.79   4904.06      1.00
        nu     13.62      4.08     13.96      7.75     19.99   6031.66      1.00
       rho      0.11      0.09      0.08      0.00      0.23   6672.39      1.00
     sigma      0.35      0.03      0.35      0.31      0.40   6528.68      1.00

Number of divergences: 4
```

Now we can make predictions for future periods. If we feed the `Predictive` function with more predictors than outcomes, the function will automatically make prediction for future periods.

<a id="code-snippet--lgtr-posterior-prediction"></a>
```python
lgtr_sample = lgtr_mcmc.get_samples()
lgtr_post = Predictive(lgtr, posterior_samples=lgtr_sample)(rng, lgtr_x, lgtr_y[:-10])

print("posterior samples:")
for k, v in lgtr_sample.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in lgtr_post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (8000, 136); L (8000, 136); R (8000, 136); alpha (8000,); beta (8000,); eta (8000, 3); g (8000, 136); g_init (8000,); gamma (8000,); l (8000, 136); l_init (8000,); lbda (8000,); mu (8000, 136); nu (8000,); rho (8000,); sigma (8000,);

posterior Prediction samples:
G (8000, 146); L (8000, 146); R (8000, 146); g (8000, 146); l (8000, 146); mu (8000, 146); y (8000, 146);
```

Let's look at the model prediction.

<a id="code-snippet--lgtr-posterior-visualisation"></a>
```python
plot_post(lgtr_y, lgtr_post, lgtr_lvs, nrow=4, ncol=2)
```

{{< figure src="/ox-hugo/afcca7a9cfa97acf04b861ed1d683420d5b50b74.png" >}}

This time the model prediction follows much closer to the local variations, we have made significant improvement to the model's predictive power. The reason for this improvement, as we can see from the above plot, is that we now have two model components, R and L, that contribute very meaningfully to the prediction. In the previous model, although the local variation component has been able to capture some local change, but the prediction for future periods is outright flat, which means it hardly alters the global trend at all. In this model, on the contrary, there is some uncertainty around the global trend G all along the time series, and this uncertainty permits the other model components, R and L, to make changes to the global trend when the data demands it, and to consequently change the overall mean and ultimately the outcome.

We can also check the posterior distribution of \\(\sigma\\):

```python
sns.histplot(lgtr_sample['sigma'], bins=30);
```

{{< figure src="/ox-hugo/7b1120a13915a0ad35e2e52a380df983f3111770.png" >}}

Indeed the standard deviation is very small, and with little variation. We can also check the scores.

<a id="code-snippet--lgtr-score"></a>
```python
check_scores(lgtr_post, lgtr_y, 10)
```

```text
sMAPE: 0.16%, MAE: 0.44, RMSE: 0.47, CPRS: 0.52
```

All the scores have significantly improved (that is, reduced), compared to the previous model.


## Using a different variation modeling approach {#using-a-different-variation-modeling-approach}

From the previous two models we can see that with the BJsales data set, the expected value does not affect the outcome variation very much, so the exponentially smoothed model is not very helpful.

Here we propose another model for the variation. In this model, the variation is a linear function of another hidden variable \\(w\\), which itself follows an autoregressive process as the smoothed average between its previous value and the current absolute error, just like \\(l\\) and \\(g\\).

<a id="code-snippet--lgtrs-data-processing"></a>
```python
lgtrs_x = lgtr_x
lgtrs_y = lgtr_y
print('Predictors shape:', lgtrs_x.shape, '\nOutcome shape:', lgtrs_y.shape)
```

```text
Predictors shape: (146, 3)
Outcome shape: (146,)
```

The full model can be written down as:

\begin{align\*}
y\_{t} &\sim \text{T} (\nu,\mu\_{t}, \sigma\_{t}) \\\\\\
\sigma\_{t} &= \kappa w\_{t-1} + \xi   \\\\\\
\mu\_{t} &= \text{G}\_{t} + \text{L}\_{t} + \text{R}\_{t} \\\\\\
\text{G}\_{t} &= g\_{t-1} + \gamma g\_{t-1}^{ \rho } \\\\\\
\text{L}\_{t} &= \lambda l\_{t-1}  \\\\\\
\text{R}\_{t} &= \text{x}\_{t} \eta \\\\\\
g\_{t} &= \alpha y\_{t}  + (1- \alpha) + \text{G}\_{t} \\\\\\
l\_{t} &= \beta  ( g\_{t}-g\_{t-1} ) + ( 1- \beta ) l\_{t-1}\\\\\\
w\_{t} &= \zeta \lvert y\_{t} - \mu\_{t} \rvert  + ( 1 - \zeta ) w\_{t-1} \\\\\\
\end{align\*}

And we update our model

<a id="code-snippet--lgtrs-model"></a>
```python
def lgtrs(x, y):
    nu = sample("nu", dist.Uniform(1, 50))

    xi = sample("xi", dist.HalfNormal(2))
    kappa = sample("kappa", dist.HalfNormal(2))

    gamma = sample("gamma", dist.HalfNormal(2))
    rho = sample("rho", dist.Beta(1, 4))

    lbda = sample("lbda", dist.Beta(2, 2))
    alpha = sample("alpha", dist.Beta(2, 2))
    beta = sample("beta", dist.Beta(2, 2))
    zeta = sample("zeta", dist.Beta(2, 2))

    with plate("D", size=x.shape[1]):
        eta = sample("eta", dist.Normal(0, 5))

    g_init = sample("g_init", dist.TruncatedCauchy(0, y[0], 10))
    l_init = sample("l_init", dist.Normal(0, 10))
    w_init = sample("w_init", dist.HalfNormal(5))

    def transition_fn(carry, xt):
        g, l, w = carry
        G = deterministic("G", g + gamma * g ** rho)
        L = deterministic("L", lbda * l)
        R = deterministic("R", jnp.dot(xt, eta))
        mu = deterministic("mu", jnp.clip(G + L + R, 0))
        sigma = deterministic("sigma", xi + kappa * w)

        yt = sample("y", dist.StudentT(nu, mu, sigma))

        g_new = deterministic("g", jnp.clip(alpha * yt + (1 - alpha) * G, 0))
        l_new = deterministic("l", beta * (g_new - g) + (1 - beta) * l)
        w_new = deterministic("w", zeta * jnp.abs(yt - mu) + (1 - zeta) * w)

        return (g_new, l_new, w_new), yt

    with condition(data={'y':y}):
        _, ys = scan(transition_fn, (g_init, l_init, w_init), x)

    return ys

print("Observed data:", lgtrs_y[:4])

with seed(rng_seed=5):
    print("\nPrior sampling:")
    print(lgtrs(lgtrs_x[:4], lgtrs_y[:1]))
    print("\nFixed data sampling:")
    print(lgtrs(lgtrs_x[:4], lgtrs_y[:4]))
```

```text
Observed data: [199.  200.2 198.6 200. ]

Prior sampling:
[199.   268.45 342.35 433.27]

Fixed data sampling:
[199.  200.2 198.6 200. ]
```

Sample from the prior model

<a id="code-snippet--lgtrs-prior"></a>
```python
lgtrs_prior = Predictive(lgtrs, num_samples=500)(rng, lgtrs_x[:-10], lgtrs_y[:1])

lgtrs_lvs, lgtrs_gvs = check_prior(lgtrs_prior)
```

```text
Global variables:
alpha (500,); beta (500,); eta (500, 3); g_init (500,); gamma (500,); kappa (500,); l_init (500,); lbda (500,); nu (500,); rho (500,); w_init (500,); xi (500,); zeta (500,);

Local variables:
G (500, 136); L (500, 136); R (500, 136); g (500, 136); l (500, 136); mu (500, 136); sigma (500, 136); w (500, 136); y (500, 136);
```

Check global parameters

<a id="code-snippet--lgtrs-parallel-global-variables"></a>
```python
compare_global_params(lgtrs_prior,
                      ['eta', 'kappa', 'l_init', 'w_init', 'xi'],
                      val_ref=50000)
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  247
```

{{< figure src="/ox-hugo/39580484afbdcaf30f123147d99e16dd1de2a007.png" >}}

And local variables

<a id="code-snippet--lgtrs-prior-local-visualisation"></a>
```python
plot_locals(lgtrs_prior, lgtrs_lvs, val_ref=50000, nrows=3, ncols=3, figsize=(8,8))
```

```text
Number of samples with extreme values :  247
```

{{< figure src="/ox-hugo/41d5f1b19907e3230cf6d3472461fb1675a9d0fb.png" >}}

The outcome range has greatly increased, we now have almost half the samples having outcomes greater than 50000. This is way bigger than the range of data we'd expect from the real world. If a store has regular sales between 200 and 300 per day, we can be sure that that store won't hold an inventory of 50000. If they do, well, statistical modeling of their sales figure should be the last of their concern.

Do we need to revise the model before we carry on doing inference? That depends. If we have some extra information readily available that we can use to make our priors more precise, that will certainly help. But this is above all a cost-benefit tradeoff. Does the improvement we bring to the model worth the effort we are putting in? It's possible that the information in the data likelihood will completely overwhelm the information in the prior, and if that is the case the extra effort spent in improving the priors wouldn't really worth it. Again, what's important is, à priori, not to limit our model to regions of the ambient space not covering the observed data. This doesn't seem to be the case here, so we'll move on.

Prior prediction conditioned on data

<a id="code-snippet--lgtrs-prior-with-observation"></a>
```python
lgtrs_prior_fixed = Predictive(lgtrs, num_samples=500)(rng, lgtrs_x[:-10], lgtrs_y[:-10])

plot_locals(lgtrs_prior_fixed, lgtrs_lvs, v_ref='mu', val_ref=500,
            nrows=3, ncols=3, figsize=(8, 8))
```

```text
Number of samples with extreme values :  18
```

{{< figure src="/ox-hugo/7e4c43e878381265437a4823c22ab695998bc079.png" >}}

Like before, now we chose one sample from the prior predictive samples and do parameter recovery on it.

<a id="code-snippet--lgtrs-fake-data"></a>
```python
n = 74
y = lgtrs_prior['y'][n]
plt.plot(y)
plt.title('Fake outcome no. {} from prior'.format(n));
```

{{< figure src="/ox-hugo/b15eefe116a08fed9e0d8ae61b8b381ebfbb5581.png" >}}

This does not look like our observed data at all. And inference.

<a id="code-snippet--lgtrs-fake-parameter-recovery"></a>
```python
kernel = NUTS(lgtrs)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
mcmc.run(rng, lgtrs_x[:-10], y)
mcmc.print_summary()
```

```text
sample: 100%|██████████| 2000/2000 [01:21<00:00, 24.64it/s, 255 steps of size 6.31e-03. acc. prob=0.84]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.62      0.10      0.62      0.45      0.77    383.38      1.00
      beta      0.56      0.21      0.58      0.22      0.89    397.28      1.00
    eta[0]      1.47      3.84      1.46     -4.80      7.56    587.22      1.00
    eta[1]      3.05      1.35      3.11      0.82      5.21    326.17      1.00
    eta[2]     -3.62      1.28     -3.69     -5.71     -1.56    339.03      1.00
    g_init    198.59     11.32    197.06    182.84    218.44     48.30      1.01
     gamma      1.32      1.06      1.07      0.01      2.84    654.79      1.00
     kappa      0.98      0.13      0.97      0.78      1.18    340.71      1.00
    l_init      3.02     11.07      2.55    -15.73     20.07    268.45      1.00
      lbda      0.43      0.21      0.41      0.07      0.76    388.16      1.00
        nu     30.39     12.30     31.86     12.85     49.92    821.47      1.00
       rho      0.10      0.08      0.08      0.00      0.21    714.88      1.00
    w_init      7.22      3.10      6.87      2.20     11.80    707.32      1.00
        xi      2.23      0.62      2.21      1.24      3.31    254.05      1.00
      zeta      0.66      0.10      0.65      0.49      0.83    375.06      1.00

Number of divergences: 0
```

Let's visualise the parameter posterior.

```python
check_inference(mcmc, lgtrs_prior, lgtrs_gvs, n=n)
```

{{< figure src="/ox-hugo/87e2a37dffef889d2b924412a30a1dc16a074573.png" >}}

Although the data look quite different from the observed, we are still able to recover the parameters reasonably well, this should give us some confidence in our model. Next we do inference on the real data.

<a id="code-snippet--lgtrs-inference"></a>
```python
lgtrs_kernel = NUTS(lgtrs, target_accept_prob=0.95)
lgtrs_mcmc = MCMC(lgtrs_kernel, num_warmup=2000, num_samples=2000,
                  num_chains=4, progress_bar=False)
lgtrs_mcmc.run(random.PRNGKey(1), lgtrs_x[:-10], lgtrs_y[:-10])
lgtrs_mcmc.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.98      0.02      0.98      0.95      1.00   2028.70      1.00
      beta      0.97      0.02      0.98      0.95      1.00   3382.26      1.00
    eta[0]     -1.70      1.36     -1.42     -3.60      0.32   1499.95      1.00
    eta[1]     -4.63      0.12     -4.64     -4.84     -4.44   1423.19      1.00
    eta[2]      4.62      0.12      4.63      4.43      4.81   1449.36      1.00
    g_init    198.98      0.90    198.97    197.59    200.37   2056.09      1.00
     gamma      1.09      0.93      0.84      0.00      2.38   1653.04      1.00
     kappa      0.29      0.14      0.28      0.05      0.50   1230.54      1.00
    l_init     -0.30      1.89     -0.28     -3.16      2.50   1734.64      1.00
      lbda      0.74      0.03      0.74      0.69      0.79   2036.51      1.00
        nu     30.10     11.89     30.43     13.27     49.94   3781.52      1.00
       rho      0.10      0.09      0.08      0.00      0.23   3477.43      1.00
    w_init      3.02      2.66      2.29      0.00      6.82   2451.80      1.00
        xi      0.27      0.04      0.27      0.20      0.35   1227.37      1.00
      zeta      0.70      0.17      0.73      0.44      0.98   2589.39      1.00

Number of divergences: 0
```

Prediction

<a id="code-snippet--lgtrs-posterior-prediction"></a>
```python
lgtrs_sample = lgtrs_mcmc.get_samples()
lgtrs_post = Predictive(lgtrs, posterior_samples=lgtrs_sample)(rng, lgtrs_x, lgtrs_y[:-10])

print("posterior samples:")
for k, v in lgtrs_sample.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in lgtrs_post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (8000, 136); L (8000, 136); R (8000, 136); alpha (8000,); beta (8000,); eta (8000, 3); g (8000, 136); g_init (8000,); gamma (8000,); kappa (8000,); l (8000, 136); l_init (8000,); lbda (8000,); mu (8000, 136); nu (8000,); rho (8000,); sigma (8000, 136); w (8000, 136); w_init (8000,); xi (8000,); zeta (8000,);

posterior Prediction samples:
G (8000, 146); L (8000, 146); R (8000, 146); g (8000, 146); l (8000, 146); mu (8000, 146); sigma (8000, 146); w (8000, 146); y (8000, 146);
```

and visualisation

<a id="code-snippet--lgtrs-posterior-visualisation"></a>
```python
plot_post(lgtrs_y, lgtrs_post, lgtrs_lvs, nrow=3, ncol=3)
```

{{< figure src="/ox-hugo/6eeff20b8101d2284ec5f1c066239988ebe77c0b.png" >}}

This time we indeed have a much more precise estimation of the standard deviation \\(\sigma\\), we can see that \\(\sigma\\) increases when the time series has a sudden change of global trend. But it is very small and for the prediction of future periods, the standard deviation returned to a flat prediction like in previous models.

Compare the scores.

<a id="code-snippet--lgtrs-score"></a>
```python
check_scores(lgtrs_post, lgtrs_y, 10)
```

```text
sMAPE: 0.14%, MAE: 0.37, RMSE: 0.41, CPRS: 0.48
```

The scores are slightly better than the previous model.

In the last three models, we have used three different methods to model the outcome variance, and two different methods for the hidden level. This flexibility is the most important feature of our Bayesian modeling approach: we take different components capable of capturing different features, and freely assemble them to fit the need of a specific data set we want to model.

In the previous models, to capture the local variation, we first used a Markov process, in which we modeled the local variation as the change in global trend; then, since we also have some predictors corresponding to each time period, we also added a regression component. But we didn't introduce seasonality to the model, because there doesn't seem to be any such feature present in this specific data set. However, if the data are indeed seasonal, we can also add new components to address this.

Still, seasonality can be represented in infinitely many different ways, and in our modeling framework, there remains the question of how seasonality should be incorporated into it. Should it be an additional component, like the regression component we have just added, to capture extra local variation, or should it be an enhancing factor, that impacts that outcome by changing the global trend? In the next section we first introduce a multiplicative seasonality, which affects the outcome by increasing or dereasing the global trend, and in the section that follows, we'll introduce an additive seasonality as an independent component.


## Adding multiplicative seasonality to the model {#adding-multiplicative-seasonality-to-the-model}

For this model we'll use the monthly air Passengers data, which contains 144 observations of passenger counts. We'll use the first 120 points for modeling and the last 24 for prediction.

Although the passenger counts are clearly integers, the outcome of the Student's T distribution is real number. Later we might try another heavy tailed integer valued distribution, such as Negative Binomial, to better fit the observation.

<a id="code-snippet--sgt-data-process"></a>
```python
sgt_y = jnp.array(air_passengers, dtype='float64')

plt.plot(air_passengers, '.')
plt.title('Monthly Air passengers');
```

{{< figure src="/ox-hugo/2396e9a5706fa0784abf8b64818fcf9a90cbc0e0.png" >}}

In this model we'll replace the local trend with a seasonal effect. Specifically, we'll start with a multiplicative model in this section and explore the additive one in the next.

This model assumes the number of periods for seasonality is already known and thus not infered from the data. This works with datasets with natural seasonality, which is the case here with monthly data.

\begin{align\*}
y\_{t} &\sim \text{T} (\nu,\mu\_{t}, \sigma\_{t}) \\\\\\
\sigma\_{t} &= \kappa \mu\_{t}^{ \tau} + \xi   \\\\\\
\mu\_{t} &= s\_{t}  \text{G}\_{t}  \\\\\\
\text{G}\_{t} &= g\_{t-1} + \gamma g\_{t-1}^{ \rho } \\\\\\
g\_{t} &= \alpha (y\_{t} / s\_{t}) + ( 1- \alpha ) \text{G}\_{t}\\\\\\
s\_{t+T} &= \beta (y\_{t} / \text{G}\_{t}) + ( 1- \beta ) s\_{t}  \\\\\\
\end{align\*}

Because this is a multiplicative model, when computing the global trend and seasonality, we have to remove from the outcome the respective effects of each other, that's why we have the divisions in the smoothed average process.

<a id="code-snippet--sgt-model"></a>
```python
def sgt(y, T, future=3):
    N = y.shape[0]
    nu = sample("nu", dist.Uniform(1, 50))

    xi = sample("xi", dist.HalfNormal(2))
    tau = sample("tau", dist.Beta(1, 4))
    kappa = sample("kappa", dist.HalfNormal(2))

    gamma = sample("gamma", dist.HalfNormal(2))
    rho = sample("rho", dist.Beta(1, 4))

    alpha = sample("alpha", dist.Beta(2, 2))
    beta = sample("beta", dist.Beta(2, 2))

    g_init = sample("g_init", dist.TruncatedCauchy(0, y[0], 10))

    with plate("T", size=T):
        s_init = sample("s_init", dist.HalfNormal(4))

    def transition_fn(carry, t):
        g, s = carry
        G = deterministic("G", g + gamma * g ** rho)
        mu = deterministic("mu", jnp.clip(G * s[0], 0))
        sigma = deterministic("sigma", xi + kappa * mu ** tau)

        yt = sample("y", dist.StudentT(nu, mu, sigma))

        g_new = deterministic("g", jnp.clip(alpha * (yt/s[0]) + (1-alpha) * G, 0.1))
        su = deterministic("su", jnp.clip(beta * (yt/G) + (1 - beta) * s[0], 0.1))
        s_new = deterministic("s", jnp.concatenate([s[1:], su[None]], axis=0))

        return (g_new, s_new), yt

    with condition(data={'y':y}):
        _, ys = scan(transition_fn, (g_init, s_init),  jnp.arange(0, N+future))

    return ys

print("Observed data:", sgt_y[:9])

with seed(rng_seed=3):
    print("\nPrior sampling:")
    print(sgt(sgt_y[:1], T=3, future=8))
    print("\nFixed data sampling:")
    print(sgt(sgt_y[:9], T=3, future=0))
```

```text
Observed data: [112. 118. 132. 129. 121. 135. 148. 148. 136.]

Prior sampling:
[112.   512.32 519.71 173.66 513.11 519.74 174.1  514.08 520.91]

Fixed data sampling:
[112. 118. 132. 129. 121. 135. 148. 148. 136.]
```

Sample from prior

<a id="code-snippet--sgt-prior"></a>
```python
sgt_prior = Predictive(sgt, num_samples=500)(rng, sgt_y[:1], T=12, future=119)

sgt_lvs, sgt_gvs = check_prior(sgt_prior)
sgt_lvs.remove('s')
```

```text
Global variables:
alpha (500,); beta (500,); g_init (500,); gamma (500,); kappa (500,); nu (500,); rho (500,); s_init (500, 12); tau (500,); xi (500,);

Local variables:
G (500, 120); g (500, 120); mu (500, 120); s (500, 120, 12); sigma (500, 120); su (500, 120); y (500, 120);
```

Check global parameters

<a id="code-snippet--sgt-parallel-global-variables"></a>
```python
compare_global_params(sgt_prior,
                      ['alpha', 'beta', 'gamma', 'kappa', 'rho', 'tau'],
                      val_ref=10000)
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  126
```

{{< figure src="/ox-hugo/48f99d576cdff8e6d86d841778cf58dc8444cec2.png" >}}

And local variables

<a id="code-snippet--sgt-prior-local-visualisation"></a>
```python
plot_locals(sgt_prior, sgt_lvs, val_ref=50000, nrows=3, ncols=2, figsize=(8,8))
```

```text
Number of samples with extreme values :  41
```

{{< figure src="/ox-hugo/848bad8f342d2359cad5aa7527b14c79ec80707f.png" >}}

Prior prediction conditioned on data

<a id="code-snippet--sgt-prior-with-observation"></a>
```python
sgt_prior_fixed = Predictive(sgt, num_samples=500)(rng, sgt_y[:-24], T=12, future=0)

plot_locals(sgt_prior_fixed, sgt_lvs, v_ref='sigma', val_ref=100,
            nrows=3, ncols=2, figsize=(8, 8))
```

```text
Number of samples with extreme values :  23
```

{{< figure src="/ox-hugo/1e2db5fd0747a569161f15675445251f124720be.png" >}}

Like before, now we chose one sample from the prior predictive samples and do parameter recovery on it.

<a id="code-snippet--sgt-fake-data"></a>
```python
n = 34
y = sgt_prior['y'][n]
plt.plot(y)
plt.title('Fake outcome no. {} from prior'.format(n));
```

{{< figure src="/ox-hugo/ff39eb04f0fd19d76877795e3e68acb47e13fa39.png" >}}

And inference.

<a id="code-snippet--sgt-fake-parameter-recovery"></a>
```python
kernel = NUTS(sgt)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
mcmc.run(rng, y, 12, 0)
mcmc.print_summary()
```

```text
sample: 100%|██████████| 2000/2000 [00:28<00:00, 69.51it/s, 127 steps of size 2.65e-02. acc. prob=0.95]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.08      0.02      0.07      0.05      0.10    382.73      1.00
      beta      0.25      0.09      0.25      0.08      0.38    164.82      1.01
    g_init     70.03      8.90     70.14     56.56     85.79    140.10      1.02
     gamma      0.21      0.02      0.21      0.17      0.25    227.87      1.00
     kappa      2.79      0.98      2.68      1.27      4.39    390.38      1.00
        nu     25.09     13.30     24.07      4.45     44.45    787.37      1.00
       rho      0.63      0.02      0.63      0.61      0.66    238.41      1.01
 s_init[0]      1.73      0.21      1.69      1.46      2.09    111.58      1.02
 s_init[1]      0.05      0.04      0.03      0.01      0.11    168.23      1.01
 s_init[2]      2.77      0.32      2.71      2.38      3.31    102.47      1.02
 s_init[3]      1.04      0.13      1.01      0.84      1.23    109.68      1.03
 s_init[4]      4.86      0.55      4.72      4.13      5.76    101.04      1.02
 s_init[5]      2.01      0.23      1.96      1.67      2.37    104.42      1.03
 s_init[6]      0.68      0.09      0.67      0.56      0.81    119.14      1.02
 s_init[7]      2.57      0.30      2.51      2.14      3.03    104.67      1.02
 s_init[8]      2.93      0.34      2.86      2.44      3.43    100.30      1.02
 s_init[9]      3.77      0.44      3.67      3.15      4.43     97.93      1.02
s_init[10]      7.47      0.85      7.29      6.29      8.76     98.90      1.02
s_init[11]      0.37      0.05      0.36      0.29      0.45    140.11      1.02
       tau      0.26      0.05      0.26      0.17      0.34    355.37      1.00
        xi      1.64      1.22      1.42      0.00      3.36    564.33      1.00

Number of divergences: 0
```

Let's visualise the parameter posterior.

```python
check_inference(mcmc, sgt_prior, sgt_gvs, n=n)
```

{{< figure src="/ox-hugo/08f20b311b16e3e19f0dbfe23afa1b417177934f.png" >}}

This time, let's also look at the prediction.

```python
samples = mcmc.get_samples()
post = Predictive(sgt, posterior_samples=samples)(rng, y, T=12, future=0)

print("posterior samples:")
for k, v in samples.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (1000, 120); alpha (1000,); beta (1000,); g (1000, 120); g_init (1000,); gamma (1000,); kappa (1000,); mu (1000, 120); nu (1000,); rho (1000,); s (1000, 120, 12); s_init (1000, 12); sigma (1000, 120); su (1000, 120); tau (1000,); xi (1000,);

posterior Prediction samples:
G (1000, 120); g (1000, 120); mu (1000, 120); s (1000, 120, 12); sigma (1000, 120); su (1000, 120); y (1000, 120);
```

And visualisation.

```python
plot_post(y, post, sgt_lvs)
```

{{< figure src="/ox-hugo/21b766000b52a6245fb26a596815a313aa01757d.png" >}}

This looks passable enough. Next we do inference on the real data.

<a id="code-snippet--sgt-inference"></a>
```python
sgt_kernel = NUTS(sgt)
sgt_mcmc = MCMC(sgt_kernel, num_warmup=2000, num_samples=2000,
                num_chains=4, progress_bar=False)
sgt_mcmc.run(rng, sgt_y[:-24], 12, 0)
sgt_mcmc.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.44      0.13      0.40      0.23      0.64    303.53      1.01
      beta      0.57      0.16      0.61      0.36      0.81    247.78      1.01
    g_init     38.82     13.05     35.86     22.09     55.70    781.41      1.00
     gamma      0.45      0.25      0.41      0.07      0.78   1777.05      1.00
     kappa      1.52      1.03      1.28      0.14      2.99   1653.59      1.00
        nu     26.60     13.23     26.51      8.56     50.00   2684.45      1.00
       rho      0.14      0.12      0.11      0.00      0.32   2381.43      1.00
 s_init[0]      3.09      0.83      3.07      1.77      4.45    901.72      1.00
 s_init[1]      3.24      0.86      3.23      1.85      4.67    900.04      1.00
 s_init[2]      3.61      0.96      3.58      2.11      5.21    901.65      1.00
 s_init[3]      3.45      0.91      3.43      1.97      4.92    912.12      1.00
 s_init[4]      3.24      0.86      3.22      1.87      4.66    890.69      1.00
 s_init[5]      3.58      0.96      3.55      2.08      5.18    873.56      1.00
 s_init[6]      3.89      1.04      3.87      2.31      5.67    871.61      1.00
 s_init[7]      3.84      1.02      3.81      2.22      5.51    871.98      1.00
 s_init[8]      3.47      0.92      3.45      2.00      4.98    892.56      1.00
 s_init[9]      3.00      0.80      2.98      1.73      4.32    886.32      1.00
s_init[10]      2.64      0.70      2.61      1.54      3.79    915.90      1.00
s_init[11]      3.04      0.81      3.01      1.68      4.30    913.26      1.00
       tau      0.34      0.14      0.33      0.09      0.55   1972.29      1.00
        xi      1.42      1.12      1.16      0.00      2.97   3589.84      1.00

Number of divergences: 0
```

Now we can make predictions for future periods.

<a id="code-snippet--sgt-posterior-prediction"></a>
```python
sgt_sample = sgt_mcmc.get_samples()
sgt_post = Predictive(sgt, posterior_samples=sgt_sample)(rng, sgt_y[:-24], T=12, future=24)

print("posterior samples:")
for k, v in sgt_sample.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in sgt_post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (8000, 120); alpha (8000,); beta (8000,); g (8000, 120); g_init (8000,); gamma (8000,); kappa (8000,); mu (8000, 120); nu (8000,); rho (8000,); s (8000, 120, 12); s_init (8000, 12); sigma (8000, 120); su (8000, 120); tau (8000,); xi (8000,);

posterior Prediction samples:
G (8000, 144); g (8000, 144); mu (8000, 144); s (8000, 144, 12); sigma (8000, 144); su (8000, 144); y (8000, 144);
```

Let's look at the model prediction.

<a id="code-snippet--sgt-posterior-visualisation"></a>
```python
plot_post(sgt_y, sgt_post, sgt_lvs)
```

{{< figure src="/ox-hugo/b078a41d1d102ff8c2a47e7f9eee0c9d24585b58.png" >}}

The model has been able to capture the growing global trend, and the seasonality present in it. However, looking at the prediction for future periods, it looks clear that the model has been systematically underestimating the outcome.

We have saw similar problem before, in the first LGT model. Back then, the model is also underestimating, and because we have another predictor variable, we have been able to use it to compensate for some extra local information. With this data set, however, we don't have any predictor available.

One way we can get around this, is to simply use lagged outcomes to construct a predictor matrix, and add a regression component to the model as before. However, there is another way to make use of local information, and that is using a moving window of outcomes. In a later model, we'll use this moving window information to compute the global trend. But first, we'll return to, as promised, an additive approach to modeling seasonality.

Finally, look at the scores.

<a id="code-snippet--sgt-score"></a>
```python
check_scores(sgt_post, sgt_y, 24)
```

```text
sMAPE: 6.74%, MAE: 29.26, RMSE: 33.17, CPRS: 19.79
```


## Using additive seasonality {#using-additive-seasonality}

Next we replace the multiplicate seasonality with an additive one. For the multiplicative seasonality, we assume it affects the outcome by enhancing or decreasing the global trend, an amplifying factor, so it naturally only takes positive values. But for the additive seasonality, we assume it adds or deducts from the global trend, so it can be both positive and negative.

<a id="code-snippet--sgta-data-process"></a>
```python
sgta_y = sgt_y
```

The model can be represented as:

\begin{align\*}
y\_{t} &\sim \text{T} (\nu,\mu\_{t}, \sigma\_{t}) \\\\\\
\sigma\_{t} &= \kappa \mu\_{t}^{ \tau} + \xi   \\\\\\
\mu\_{t} &= \text{G}\_{t} + s\_{t}  \\\\\\
\text{G}\_{t} &= g\_{t-1} + \gamma g\_{t-1}^{\rho} \\\\\\
g\_{t} &= \alpha (y\_{t} - s\_{t} ) + ( 1 - \alpha ) \text{G}\_{t} \\\\\\
s\_{t+T} &= \beta (y\_{t} - g\_{t}) + ( 1- \beta  ) s\_{t} \\\\\\
\end{align\*}

We'll put a relatively wide prior on the seasonality effect. And because updating the seasonality effect needs the outcome before, we need to keep at least T periods of outcomes in memory, when doing the update.

<a id="code-snippet--sgta-model"></a>
```python
def sgta(y, T, future=3):
    N = y.shape[0]
    nu = sample("nu", dist.Uniform(1, 50))

    xi = sample("xi", dist.HalfNormal(2))
    tau = sample("tau", dist.Beta(1, 3))
    kappa = sample("kappa", dist.HalfNormal(2))

    gamma = sample("gamma", dist.HalfNormal(2))
    rho = sample("rho", dist.Beta(1, 4))

    alpha = sample("alpha", dist.Beta(2, 2))
    beta = sample("beta", dist.Beta(2, 2))

    g_init = sample("g_init", dist.TruncatedCauchy(0, y[0], 10))

    with plate("T", size=T):
        s_init = sample("s_init", dist.Cauchy(0, 10))

    def transition_fn(carry, t):
        g, s = carry
        G = deterministic("G", g + gamma * g ** rho)
        mu = deterministic("mu", jnp.clip(G + s[0], 0))
        sigma = deterministic("sigma", xi + kappa * mu ** tau)

        yt = sample("y", dist.StudentT(nu, mu, sigma))

        g_new = deterministic("g", jnp.clip(alpha * (yt-s[0]) + (1-alpha) * g, 0.1))
        su = deterministic("su", beta * (yt-G) + (1-beta) * s[0])
        s_new = deterministic("s", jnp.concatenate([s[1:], su[None]], axis=0))

        return (g_new, s_new), yt

    with condition(data={'y':y}):
        _, ys = scan(transition_fn, (g_init, s_init),  jnp.arange(0, N+future))

    return ys

print("Observed data:", sgta_y[:9])

with seed(rng_seed=3):
    print("\nPrior sampling:")
    print(sgta(sgta_y[:1], T=3, future=8))
    print("\nFixed data sampling:")
    print(sgta(sgta_y[:9], T=3, future=0))
```

```text
Observed data: [112. 118. 132. 129. 121. 135. 148. 148. 136.]

Prior sampling:
[112.    99.91 117.08 112.89 100.13 116.52 112.9  100.4  116.98]

Fixed data sampling:
[112. 118. 132. 129. 121. 135. 148. 148. 136.]
```

Sample from prior

<a id="code-snippet--sgta-prior"></a>
```python
sgta_prior = Predictive(sgta, num_samples=500)(rng, sgta_y[:1], T=12, future=119)

sgta_lvs, sgta_gvs = check_prior(sgta_prior)
sgta_lvs.remove('s')
```

```text
Global variables:
alpha (500,); beta (500,); g_init (500,); gamma (500,); kappa (500,); nu (500,); rho (500,); s_init (500, 12); tau (500,); xi (500,);

Local variables:
G (500, 120); g (500, 120); mu (500, 120); s (500, 120, 12); sigma (500, 120); su (500, 120); y (500, 120);
```

Check global parameters

<a id="code-snippet--sgta-prior-global-variables"></a>
```python
compare_global_params(sgta_prior,
                      ['alpha', 'beta', 'gamma', 'kappa', 'rho', 'tau'],
                      val_ref=10000)
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  17
```

{{< figure src="/ox-hugo/b4f1c1828e896783627962117285acff72acc2d9.png" >}}

Nothing special. And local variables

<a id="code-snippet--sgta-prior-local-visualisation"></a>
```python
plot_locals(sgta_prior, sgta_lvs, val_ref=1000, nrows=3, ncols=2, figsize=(8,8))
```

```text
Number of samples with extreme values :  148
```

{{< figure src="/ox-hugo/2e500733e4aa215bd3345a6ef4dc8519f56bd621.png" >}}

Prior prediction conditioned on data

<a id="code-snippet--sgta-prior-with-observation"></a>
```python
sgta_prior_fixed = Predictive(sgta, num_samples=500)(rng, sgta_y[:-24], T=12, future=0)

plot_locals(sgta_prior_fixed, sgta_lvs, v_ref='mu', val_ref=500, nrows=3, ncols=2, figsize=(8, 8))
```

```text
Number of samples with extreme values :  115
```

{{< figure src="/ox-hugo/8664d2443e6489bef82df37592c157b7943a8d61.png" >}}

This prior predictive distribution, conditioned on data, looks a bit problematic because as we can see, the seasonality is still present in the global trend, which defeats our intention of separating the two. Later we may want to find better updating mechanisms for them.

Like before, now we chose one sample from the prior predictive samples and do parameter recovery on it.

<a id="code-snippet--sgta-fake-data"></a>
```python
n = 24
y = sgta_prior['y'][n]
plt.plot(y)
plt.title('Fake outcome no. {} from prior'.format(n));
```

{{< figure src="/ox-hugo/40812ed661209251a2c06b7fc062f8939210a0bd.png" >}}

And inference.

<a id="code-snippet--sgta-fake-parameter-recovery"></a>
```python
kernel = NUTS(sgta)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
mcmc.run(rng, y, 12, 0)
mcmc.print_summary()
```

```text
sample: 100%|██████████| 2000/2000 [00:21<00:00, 92.04it/s, 127 steps of size 7.50e-02. acc. prob=0.86]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.26      0.06      0.26      0.16      0.35    332.13      1.00
      beta      0.36      0.09      0.36      0.22      0.51    449.63      1.00
    g_init    110.09      3.95    110.05    104.09    116.60    112.53      1.00
     gamma      2.07      0.94      1.98      0.52      3.46    289.21      1.00
     kappa      0.96      0.69      0.85      0.00      1.98    284.37      1.00
        nu     24.17     12.60     22.73      5.61     44.76    436.27      1.00
       rho      0.14      0.10      0.12      0.00      0.28    217.01      1.00
 s_init[0]     -3.31      3.88     -3.48     -9.25      2.88    115.36      1.01
 s_init[1]      4.55      3.97      4.36     -1.60     11.18    108.99      1.00
 s_init[2]     12.40      4.18     12.19      5.87     19.49    108.51      1.00
 s_init[3]    181.05      4.37    180.92    174.20    188.20    117.60      1.01
 s_init[4]     -2.13      3.87     -2.19     -8.14      4.23    112.63      1.00
 s_init[5]     -7.41      3.88     -7.30    -13.94     -1.24    118.69      1.00
 s_init[6]    -31.78      4.12    -32.03    -39.49    -25.78    103.41      1.00
 s_init[7]      5.67      3.85      5.68     -0.59     11.82    110.47      1.00
 s_init[8]     -7.64      4.09     -7.62    -13.62     -0.62    104.28      1.00
 s_init[9]      0.07      3.61      0.01     -5.42      6.31     97.51      1.00
s_init[10]      0.58      3.73      0.70     -5.87      6.54    113.84      1.00
s_init[11]     56.40      4.17     56.48     49.78     62.74    108.84      1.00
       tau      0.10      0.09      0.07      0.00      0.22    445.90      1.00
        xi      1.62      0.88      1.68      0.05      2.79    234.60      1.00

Number of divergences: 0
```

Let's visualise the parameter posterior.

```python
check_inference(mcmc, sgta_prior, sgta_gvs, n=n)
```

{{< figure src="/ox-hugo/fc2751350ef86e4b564987343e38b3b1a732b4d9.png" >}}

Seems passable, though certainly not very precise, and there are a few initial seasonality values a bit far from true value. Let's also look at the prediction.

```python
samples = mcmc.get_samples()
post = Predictive(sgta, posterior_samples=samples)(rng, y, T=12, future=0)

print("posterior samples:")
for k, v in samples.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (1000, 120); alpha (1000,); beta (1000,); g (1000, 120); g_init (1000,); gamma (1000,); kappa (1000,); mu (1000, 120); nu (1000,); rho (1000,); s (1000, 120, 12); s_init (1000, 12); sigma (1000, 120); su (1000, 120); tau (1000,); xi (1000,);

posterior Prediction samples:
G (1000, 120); g (1000, 120); mu (1000, 120); s (1000, 120, 12); sigma (1000, 120); su (1000, 120); y (1000, 120);
```

And visualisation.

```python
plot_post(y, post, sgta_lvs)
```

{{< figure src="/ox-hugo/fba7ea76337c4a6c03525decf4d296afab21c779.png" >}}

The prediction looks alright but there is a visible increasing trend in the seasonality component, this again confirms the problem we've spotted in the prior predictive check, that in this model the global trend and seasonality are not completely separated.  Next we do inference on the real data.

Next we do inference on the real data.

<a id="code-snippet--sgta-inference"></a>
```python
sgta_kernel = NUTS(sgta, target_accept_prob=0.95)
sgta_mcmc = MCMC(sgta_kernel, num_warmup=2000, num_samples=2000, num_chains=4, progress_bar=False)
sgta_mcmc.run(rng, sgta_y[:-24], 12, 0)
sgta_mcmc.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.38      0.06      0.38      0.28      0.49   6867.45      1.00
      beta      0.86      0.06      0.87      0.77      0.96   9580.76      1.00
    g_init    115.91      5.67    115.70    106.96    125.53   5267.39      1.00
     gamma      2.07      1.08      1.95      0.33      3.72   6392.68      1.00
     kappa      1.08      0.82      0.86      0.06      2.19   5701.57      1.00
        nu     28.51     12.80     28.72     10.63     50.00   9145.00      1.00
       rho      0.18      0.12      0.17      0.00      0.35   4308.32      1.00
 s_init[0]     -5.47      6.80     -5.13    -16.71      5.47   6962.30      1.00
 s_init[1]     -1.95      6.44     -1.83    -12.70      8.41   6493.53      1.00
 s_init[2]      6.17      6.89      5.81     -5.22     17.21   6415.74      1.00
 s_init[3]      2.12      6.42      1.92     -8.62     12.27   6731.41      1.00
 s_init[4]     -2.94      6.57     -2.74    -14.27      7.06   6443.76      1.00
 s_init[5]      7.29      7.43      6.80     -4.71     19.03   6657.76      1.00
 s_init[6]     17.66      8.74     17.44      2.89     31.30   5920.77      1.00
 s_init[7]     15.45      8.41     15.13      1.62     28.82   5960.66      1.00
 s_init[8]      3.27      6.81      2.90     -7.49     14.58   6235.41      1.00
 s_init[9]     -9.43      7.79     -8.96    -21.87      3.38   6070.06      1.00
s_init[10]    -20.31      9.42    -20.42    -35.69     -4.91   5970.40      1.00
s_init[11]     -5.02      7.12     -4.68    -16.49      6.52   7123.06      1.00
       tau      0.45      0.15      0.44      0.22      0.70   4643.57      1.00
        xi      1.26      1.02      1.02      0.00      2.71   8965.06      1.00

Number of divergences: 0
```

And predictions.

<a id="code-snippet--sgta-posterior-prediction"></a>
```python
sgta_sample = sgta_mcmc.get_samples()
sgta_post = Predictive(sgta, posterior_samples=sgta_sample)(rng, sgta_y[:-24], T=12, future=24)

print("posterior samples:")
for k, v in sgta_sample.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in sgta_post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (8000, 120); alpha (8000,); beta (8000,); g (8000, 120); g_init (8000,); gamma (8000,); kappa (8000,); mu (8000, 120); nu (8000,); rho (8000,); s (8000, 120, 12); s_init (8000, 12); sigma (8000, 120); su (8000, 120); tau (8000,); xi (8000,);

posterior Prediction samples:
G (8000, 144); g (8000, 144); mu (8000, 144); s (8000, 144, 12); sigma (8000, 144); su (8000, 144); y (8000, 144);
```

Let's look at the model prediction.

<a id="code-snippet--sgta-posterior-visualisation"></a>
```python
plot_post(sgta_y, sgta_post, sgta_lvs, nrow=3, ncol=2)
```

{{< figure src="/ox-hugo/0c821589600e1197e73ab47a0a629a15cb54df82.png" >}}

Compared to the previous multiplicative seasonality, although the predictions for the individual components are quite different, but the final predictions for the outcome are quite similar. And as a model defect, we can see that both in this and the previous model, we are slightly underestimating the outcome.

Let's also check the scores.

<a id="code-snippet--sgta-score"></a>
```python
check_scores(sgta_post, sgta_y, 24)
```

```text
sMAPE: 7.25%, MAE: 32.15, RMSE: 37.05, CPRS: 21.50
```

The scores are also similar to the previous model.


## Calculating hidden level with moving average {#calculating-hidden-level-with-moving-average}

There are, of course, infinite ways to model the hidden level in the time series. Up to this point we have been using the current outcome to calculate a weighted average, but we can also use a moving window of past outcomes to do the calculation. In the previous models the outcome seems to be underestimated, so we also have to improve a little bit on this issue, by using more local information. Because the model is seasonal, the natural choice for window width is the seasonality.

<a id="code-snippet--sgtm-data-process"></a>
```python
sgtm_y = sgt_y
```

We'll continue to use the additive seasonality so the only change lies in how we update the hidden level:

\begin{align\*}
y\_{t} &\sim \text{T} (\nu,\mu\_{t}, \sigma\_{t}) \\\\\\
\sigma\_{t} &= \kappa \mu\_{t}^{ \tau} + \xi   \\\\\\
\mu\_{t} &= \text{G}\_{t} + s\_{t}  \\\\\\
\text{G}\_{t} &= g\_{t-1} + \gamma g\_{t-1}^{\rho} \\\\\\
g\_{t} &= \alpha ( \text{MA}\_{T} - s\_{t} )  + \left( 1- \alpha  \right) \text{G}\_{t} \\\\\\
s\_{t+T} &= \beta (y\_{t} - \text{G}\_{t}) + \left( 1- \beta  \right) s\_{t}  \\\\\\
\end{align\*}

The moving average is computed with the outcome of the previous T periods, but there are less than T previous outcomes for the fisrt T periods, so for these first T periods we'll calculate the moving average with all the outcomes available. When coding the model, it's easier to keep track of the moving window rather than the moving average, and that's how we are going to code our model.

<a id="code-snippet--sgtm-model"></a>
```python
def sgtm(y, T, future=3):
    N = y.shape[0]
    nu = sample("nu", dist.Uniform(1, 50))

    xi = sample("xi", dist.HalfNormal(2))
    tau = sample("tau", dist.Beta(1, 3))
    kappa = sample("kappa", dist.HalfNormal(2))

    gamma = sample("gamma", dist.HalfNormal(2))
    rho = sample("rho", dist.Beta(1, 4))

    alpha = sample("alpha", dist.Beta(2, 2))
    beta = sample("beta", dist.Beta(2, 2))

    g_init = sample("g_init", dist.TruncatedCauchy(0, y[0], 10))

    with plate("T", size=T):
        s_init = sample("s_init", dist.Cauchy(0, 10))

    def transition_fn(carry, t):
        g, s, mw = carry
        G = deterministic("G", g + gamma * g ** rho)
        mu = deterministic("mu", jnp.clip(G + s[0], 0))
        sigma = deterministic("sigma", xi + kappa * mu ** tau)

        yt = sample("y", dist.StudentT(nu, mu, sigma))

        mw_new = jnp.concatenate([mw[1:], yt[None]], axis=0)
        ma = jnp.where(t<T, mw_new.sum()/(t+1), mw_new.sum()/T)
        g_new = deterministic("g", jnp.clip(alpha * (ma-s[0]) + (1-alpha) * G, 0))

        su = deterministic("su", beta * (yt-G) + (1-beta) * s[0])
        s_new = deterministic("s", jnp.concatenate([s[1:], su[None]], axis=0))

        return (g_new, s_new, mw_new), yt

    with condition(data={'y':y}):
        _, ys = scan(transition_fn, (g_init, s_init, jnp.zeros(T)),  jnp.arange(0, N+future))

    return ys

print("Observed data:", sgtm_y[:9])

with seed(rng_seed=3):
    print("\nPrior sampling:")
    print(sgtm(sgtm_y[:1], T=3, future=8))
    print("\nFixed data sampling:")
    print(sgtm(sgtm_y[:9], T=3, future=0))
```

```text
Observed data: [112. 118. 132. 129. 121. 135. 148. 148. 136.]

Prior sampling:
[112.    99.92 120.55 110.81  97.84 121.03 111.06  97.94 121.42]

Fixed data sampling:
[112. 118. 132. 129. 121. 135. 148. 148. 136.]
```

Sample from prior

<a id="code-snippet--sgtm-prior"></a>
```python
sgtm_prior = Predictive(sgtm, num_samples=500)(rng, sgtm_y[:1], T=12, future=119)

sgtm_lvs, sgtm_gvs = check_prior(sgtm_prior)
sgtm_lvs.remove('s')
```

```text
Global variables:
alpha (500,); beta (500,); g_init (500,); gamma (500,); kappa (500,); nu (500,); rho (500,); s_init (500, 12); tau (500,); xi (500,);

Local variables:
G (500, 120); g (500, 120); mu (500, 120); s (500, 120, 12); sigma (500, 120); su (500, 120); y (500, 120);
```

Check global parameters

<a id="code-snippet--sgtm-prior-global-variables"></a>
```python
compare_global_params(sgtm_prior,
                      ['alpha', 'beta', 'gamma', 'kappa', 'rho', 'tau'],
                      val_ref=10000)
```

```text
Number of samples with numerical overflows:  0
Number of samples with extreme values :  9
```

{{< figure src="/ox-hugo/c757b436ac2a67db1c4eca4b038cd91c22044b3e.png" >}}

And local variables. We limit the range of outcome values to focus on the region most useful to our inference.

<a id="code-snippet--sgtm-prior-local-visualisation"></a>
```python
plot_locals(sgtm_prior, sgtm_lvs, val_ref=1000, nrows=3, ncols=2, figsize=(8,8))
```

```text
Number of samples with extreme values :  107
```

{{< figure src="/ox-hugo/5d4ec361e0ddc5a5114e731a10be7eaa3cf4b901.png" >}}

Prior prediction conditioned on data

<a id="code-snippet--sgtm-prior-with-observation"></a>
```python
sgtm_prior_fixed = Predictive(sgtm, num_samples=500)(rng, sgtm_y[:-24], T=12, future=0)

plot_locals(sgtm_prior_fixed, sgtm_lvs, v_ref='mu', val_ref=600, nrows=3, ncols=2, figsize=(8, 8))
```

```text
Number of samples with extreme values :  91
```

{{< figure src="/ox-hugo/a2c55cb92f0df675ff3c386bdcb16e2e60b8af0f.png" >}}

Like before, now we chose one sample from the prior predictive samples and do parameter recovery on it.

<a id="code-snippet--sgtm-fake-data"></a>
```python
n = 34
y = sgtm_prior['y'][n]
plt.plot(y)
plt.title('Fake outcome no. {} from prior'.format(n));
```

{{< figure src="/ox-hugo/6a50ccc0e1cc78ea248b0c568f6af7c0ebd7fafe.png" >}}

And inference.

<a id="code-snippet--sgtm-fake-parameter-recovery"></a>
```python
kernel = NUTS(sgtm)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
mcmc.run(rng, y, 12, 0)
mcmc.print_summary()
```

```text
sample: 100%|██████████| 2000/2000 [00:34<00:00, 58.56it/s, 127 steps of size 2.55e-02. acc. prob=0.97]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.18      0.11      0.16      0.05      0.31    263.80      1.00
      beta      0.23      0.09      0.22      0.08      0.36    687.63      1.00
    g_init     83.43     10.30     83.17     65.81    100.54    494.97      1.00
     gamma      0.34      0.18      0.30      0.13      0.55    244.95      1.00
     kappa      2.96      1.18      2.84      1.18      4.84    403.30      1.00
        nu     22.49     13.40     20.51      4.24     43.52    604.32      1.00
       rho      0.61      0.06      0.61      0.52      0.71    595.31      1.01
 s_init[0]      8.62      9.26      7.68     -5.30     23.76    585.44      1.00
 s_init[1]      3.04      8.06      2.44     -9.96     16.12    579.93      1.00
 s_init[2]      0.87      7.69      0.51    -12.67     12.53    873.94      1.00
 s_init[3]     11.40     10.85     10.16     -5.94     28.85    548.89      1.00
 s_init[4]    -30.98     12.08    -31.82    -51.70    -11.12    547.21      1.00
 s_init[5]      0.33      7.32      0.56    -10.54     13.89    884.46      1.01
 s_init[6]     -7.52      9.26     -6.19    -21.98      7.02    661.12      1.00
 s_init[7]      7.49      8.43      6.58     -4.04     23.48    663.13      1.00
 s_init[8]     -2.15      7.57     -2.14    -16.27      8.56    723.97      1.00
 s_init[9]     -8.66      9.88     -7.88    -24.22      7.78    514.07      1.00
s_init[10]    -85.44     10.69    -85.90   -103.53    -68.74    519.43      1.00
s_init[11]      0.32      8.95      0.57    -14.22     14.11    589.05      1.00
       tau      0.31      0.07      0.30      0.18      0.42    374.19      1.01
        xi      2.04      1.48      1.83      0.01      4.10    499.35      1.00

Number of divergences: 0
```

Let's visualise the parameter posterior.

```python
check_inference(mcmc, sgtm_prior, sgtm_gvs, n=n)
```

{{< figure src="/ox-hugo/dd4e08c68d5477f05ca99755c76ac4ba7454e615.png" >}}

And the prediction.

```python
samples = mcmc.get_samples()
post = Predictive(sgtm, posterior_samples=samples)(rng, y, T=12, future=0)

print("posterior samples:")
for k, v in samples.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (1000, 120); alpha (1000,); beta (1000,); g (1000, 120); g_init (1000,); gamma (1000,); kappa (1000,); mu (1000, 120); nu (1000,); rho (1000,); s (1000, 120, 12); s_init (1000, 12); sigma (1000, 120); su (1000, 120); tau (1000,); xi (1000,);

posterior Prediction samples:
G (1000, 120); g (1000, 120); mu (1000, 120); s (1000, 120, 12); sigma (1000, 120); su (1000, 120); y (1000, 120);
```

And visualisation.

```python
plot_post(y, post, sgtm_lvs)
```

{{< figure src="/ox-hugo/012f8cbea4077305e632b01cc8082098bf78ff70.png" >}}

This looks passable enough. Next we do inference on the real data.

<a id="code-snippet--sgtm-inference"></a>
```python
sgtm_kernel = NUTS(sgtm)
sgtm_mcmc = MCMC(sgtm_kernel, num_warmup=2000, num_samples=2000, num_chains=4, progress_bar=False)
sgtm_mcmc.run(rng, sgtm_y[:-24], 12, 0)
sgtm_mcmc.print_summary()
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
     alpha      0.33      0.07      0.33      0.22      0.44   7789.37      1.00
      beta      0.85      0.07      0.85      0.74      0.97   7396.40      1.00
    g_init    117.07      6.57    116.72    106.33    127.62   5193.36      1.00
     gamma      1.43      0.92      1.23      0.17      2.77   4927.42      1.00
     kappa      1.42      0.94      1.21      0.14      2.72   4920.42      1.00
        nu     29.27     12.57     29.73     11.40     50.00   8295.78      1.00
       rho      0.31      0.13      0.30      0.09      0.51   4515.13      1.00
 s_init[0]     -6.46      8.27     -5.87    -19.57      7.11   6327.26      1.00
 s_init[1]     -4.37      7.92     -3.89    -18.30      7.69   6996.69      1.00
 s_init[2]      2.54      7.65      2.31     -9.43     15.46   6435.26      1.00
 s_init[3]      0.63      7.66      0.56    -12.77     12.61   7391.19      1.00
 s_init[4]     -3.46      8.02     -3.12    -16.49      9.90   6913.22      1.00
 s_init[5]      4.65      8.39      4.01     -8.66     18.29   6349.97      1.00
 s_init[6]     16.64     11.18     15.60     -1.59     33.32   5049.45      1.00
 s_init[7]     21.50     12.06     21.01      0.72     39.55   5704.33      1.00
 s_init[8]     12.13     10.45     11.14     -5.10     27.90   5348.41      1.00
 s_init[9]     -0.26      8.15     -0.21    -14.14     12.25   6551.92      1.00
s_init[10]    -11.73     11.34    -10.59    -30.25      6.35   6139.05      1.00
s_init[11]     -5.73      9.18     -5.19    -21.16      9.06   7049.80      1.00
       tau      0.43      0.13      0.42      0.22      0.64   3996.46      1.00
        xi      1.43      1.13      1.17      0.00      3.00   8111.81      1.00

Number of divergences: 0
```

Now we can make predictions for future periods.

<a id="code-snippet--sgtm-posterior-prediction"></a>
```python
sgtm_sample = sgtm_mcmc.get_samples()
sgtm_post = Predictive(sgtm, posterior_samples=sgtm_sample)(rng, sgtm_y[:-24], T=12, future=24)

print("posterior samples:")
for k, v in sgtm_sample.items():
    print(k, v.shape, end='; ')

print("\n\nposterior Prediction samples:")
for k, v in sgtm_post.items():
    print(k, v.shape, end='; ')
```

```text
posterior samples:
G (8000, 120); alpha (8000,); beta (8000,); g (8000, 120); g_init (8000,); gamma (8000,); kappa (8000,); mu (8000, 120); nu (8000,); rho (8000,); s (8000, 120, 12); s_init (8000, 12); sigma (8000, 120); su (8000, 120); tau (8000,); xi (8000,);

posterior Prediction samples:
G (8000, 144); g (8000, 144); mu (8000, 144); s (8000, 144, 12); sigma (8000, 144); su (8000, 144); y (8000, 144);
```

Let's look at the model prediction.

<a id="code-snippet--sgtm-posterior-visualisation"></a>
```python
plot_post(sgtm_y, sgtm_post, sgtm_lvs, ncol=2)
```

{{< figure src="/ox-hugo/d0d70c10f5b23a86cf71bc57239b75d16a352176.png" >}}

Here the prediction is slightly better than before, the underestimation problem seems abatted, but we're spotting another problem. The global trend still has a very strong seasonality effect built in it. Upon some reflection it's not difficult to understand why: we're now using the moving average to calculate the global trend, and since the moving average uses all the outcome for one cycle, the seasonality is clearly present in it. To avoid this we might consider removing the seasonality in moving average calculation, but this removal might also introduce new problems.

<a id="code-snippet--sgtm-score"></a>
```python
check_scores(sgtm_post, sgtm_y, 24)
```

```text
sMAPE: 5.67%, MAE: 25.50, RMSE: 29.25, CPRS: 17.07
```

The model underestimates the outcome. The cost of the model being TOO flexible. In this data set the global trend is quite clear, a simple AR model should suffice.


## Conclusion {#conclusion}

In this post we studied some time series models to model global trend, local variation, and seasonality. Our goal is not to build the perfect model: every model should be tailor made for the specific data set. Rather, our goal is to build models using composable components and observing their effects on the model outcome, and make changes when necessary.

The models are all based on a smoothed exponential process, which is quite flexible in its wide range of behaviours, but sometimes when modeling data with strong pattern this flexibility might not be necessary.

The original Rlgt package also includes a double seasonality model, and a seasonality modeled by an extra exponentially smoothed process, these models are not implemented here, because they don't enhance the performance, and induce more identifiability issues.
