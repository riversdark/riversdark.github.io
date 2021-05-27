+++
title = "The Pest"
date = 2021-02-24
tags = ["workflow"]
categories = ["Statistics"]
draft = false
toc = true
+++

This is a primer on Bayesian model building and inference.

<!--more-->

This post is based on [the tutorial](https://github.com/jgabry/stancon2018helsinki%5Fintro) given by Jonah Gabry at Stancon 2018, and that's where the data can be found. I ported the tutorial to `NumPyro` because it is a great introduction to Bayesian modeling and workflow.

In this post we'll cover many different modeling techniques, including

-   Poisson model for discrete data
-   Negative Binomial model for overdispersed data
-   hierarchical modeling
-   additive multi-factor modeling
-   time series modeling with linear autoregressive process
-   time series modeling with non-linear non-parametric Gaussian process

The purpose of this post is to get familiar with the Bayesian model expansion process, and the use of data visualisation for model building. There are many nuances in Bayesian modeling that we won't get into, especially the MCMC inference and its result examination. Essentially, we focus on the statistical modeling part while leaving out the details of statistical computation.


## The problem {#the-problem}

New York, as its great reputation merits, has cockroaches everywhere, especially in the dark corners of the apartment compounds. Recently, some residents have been complaining about the omnipresence of the pest; the property manager intends to solve the problem efficiently and effectively, by deploying traps in the apartments. To establish the relationship between the number of traps to set up, and the number of complaints received, an experiment has been carried out:

-   At the beginning of each month, a pest inspector randomly places a number of bait stations throughout the building, without knowledge of the current cockroach levels in the building
-   At the end of the month, the manager records the total number of cockroach complaints in that building.

With this data, we'd like to model how complaints change as the number of traps changes. Since the number of traps installed and the expense for dealing with extra costs both contribute to the final maintenance cost, we'd like to choose the number of traps installed to minimise the cost.

Let's import the packages first.

```python
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns
sns.set_theme(palette='Set2')
colors = sns.color_palette()

import numpyro
numpyro.set_host_device_count(4)

import jax.numpy as jnp
from jax import random
from jax.config import config; config.update("jax_enable_x64", True)
rng = random.PRNGKey(123)
```


## The data {#the-data}

The data provided to us is in a file called `pest.csv`. Let's load the data and see what the structure is:

```python
pest_link = 'data/pest.csv'
pest_data = pd.read_csv(pest_link, index_col=0)
pest_data.index = pest_data.index - 1
pest_data.columns
```

```text
Index(['mus', 'building_id', 'wk_ind', 'date', 'traps', 'floors',
       'sq_footage_p_floor', 'live_in_super', 'monthly_average_rent',
       'average_tenant_age', 'age_of_building', 'total_sq_foot', 'month',
       'complaints'],
      dtype='object')
```

We have access to the following fields:

-   `complaints`: Number of complaints per building per month
-   `building_id`: The unique building identifier
-   `traps`: The number of traps used per month per building
-   `date`: The date at which the number of complaints are recorded
-   `live_in_super`: An indicator for whether the building has a live-in superintendent
-   `age_of_building`: The age of the building
-   `total_sq_foot`: The total square footage of the building
-   `average_tenant_age`: The average age of the tenants per building
-   `monthly_average_rent`: The average monthly rent per building
-   `floors`: The number of floors per building

First, let's see how many buildings we have data for:

```python
print(pest_data.building_id.unique())
```

```text
[37 13  5 26 93 47 45 70 62 98]
```

And make some plots of the raw data. First look at the variable we are trying to model, the `complaints`. We need to pass the `order` keyword to the count plot function because we want to plot [the zero counts](https://stackoverflow.com/questions/45352766/python-seaborn-plotting-frequencies-with-zero-values/45359713#45359713) in the data.

```python
sns.countplot(data=pest_data,
              x='complaints',
              color=colors[0],
              order = range(pest_data.complaints.max()+1));
```

{{< figure src="/ox-hugo/32af0f7384fb84c6d5454bcd8e03d90fa3decd0d.png" >}}

And a scatter plot between the predictor and the outcome. because both `traps` and `complaints` are integers, we need to jitter them a bit so that they don't overlap.

```python
pest_data['traps_jitter'] = pest_data['traps'] + random.normal(rng, [len(pest_data)]) * 0.3
pest_data['complaints_jitter'] = pest_data['complaints'] + random.normal(rng, [len(pest_data)]) * 0.3
sns.scatterplot(data=pest_data, x='traps_jitter', y='complaints_jitter');
```

{{< figure src="/ox-hugo/f63b506b0e387ef0e3d1972af5c639fc9c082891.png" >}}

Just looking at the scatter plot, it's not immediately obvious how these two variables might be related, and there is certainly a large amount of variation. Without extra information, it might be difficult to do predictions of one variable using the other.

The first question we'll look at is just whether the number of `complaints` per building per month is associated with the number of `traps` per building per month, ignoring the temporal and across-building variation (we'll come back to those sources of variation later in the document). How should we model the number of complaints?


## Modeling count data: Poisson distribution {#modeling-count-data-poisson-distribution}

We only have some rudimentary information about what we should expect. The number of complaints should be, naturally, natural numbers, and the Poisson distribution is the common distribution for modeling integer valued variables, so that's where we'll start with.

Having decided to use a Poisson model for the outcome variable, we'll next decide how to relate the predictors to the parameters of this distribution. The Poisson distribution has only one parameter, which decides both the mean and the variance of the outcome. More specifically, both the mean and variance are equal to the rate parameter. When doing statistical modeling, we choose a distribution for some of its properties, then we examine whether the data match the other properties of the distribution. If not, we then consider how to improve the fit by revising the model.

Because the rate parameter is always positive, We will build a linear model with support on \\(\mathbb{R}\\), then use the exponential transformation to limit the support to \\(\mathbb{R}^{+}\\). For each observation we have

\begin{aligned}
\textrm{complaints}\_{n} & \sim \textrm{Poisson}(\lambda\_{n}) \\\\\\
\lambda\_{n} & = \exp{(\eta\_{n})} \\\\\\
\eta\_{n} &= \alpha + \beta  \textrm{traps}\_{n}
\end{aligned}

One thing to notice is that although both the predictor, `traps`, and the outcome, `compalints`, can only be integers, this fact is not immediately obvious to our model. Here we are modeling `complaints` as a Poisson variable, so it's indeed interpreted as an integer value, but for `traps`, as far as our model is concerned, it's just a variable with any possible value on \\(\mathbb{R}\\). For this reason we can transform `traps`, or any other variable we might include in our model later, so that the interpretation and computation are easier; but if we transform the outcome variable `complaints`, we need to consider whether the chosen probability distribution is still appropriate for the transformed variable.

We don't have much prior information about \\(\alpha\\) and \\(\beta\\), but we suspect that the intercept should be positive while the slope negative, so we give them some weakly information priors. To see if the priors actually make sense, we will have to see what kind of predictions the model makes, later, in the prior predictive checks.

Another thing worth noticing is that although to transform a variable \\(\eta\\) on \\(\mathbb{R}\\) to another variable \\(\lambda\\) on \\(\mathbb{R}^+\\), the [exponential transformation](https://en.wikipedia.org/wiki/Exponential%5Ffunction#:~:text=In%20mathematics%2C%20an%20exponential%20function,x%20occurs%20as%20an%20exponent) is a natural choice, its not without its problems. We can take a look at the shape of the exponential function to notice that when \\(\eta\\) is negative, a huge amount of change is \\(\eta\\) leads to little change in \\(\lambda\\) while when it's positive, a small change might leads to (exponentially, as the name indicates) huge jumps in \\(\lambda\\) values.

{{< figure src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Exp.svg/320px-Exp.svg.png" >}}

This can be a problem for us because it's very easy to have numerical overflows when using this transformation, when the rate parameter is too big. For this reason we'll try to transform the predictor variables to a reasonable range, and this also helps to ease the choice of priors.


### Model {#model}

We'll choose some vaguely reasonable priors for the parameters. Technically the priors are also part of the model but because this part of the model is closely related to how we implement the model and also how we transform the data, we keep them separated from the other part of the model.

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\end{aligned}

Is the prior flexible enough? We can check it through the prior predictive checks.

<a id="code-snippet--model-sp"></a>
```python
from numpyro import sample, plate, deterministic
from numpyro.distributions import Normal, Poisson

def sp(traps):
    """Simple Poisson Regression model"""

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + β * traps))
        return sample('y', Poisson(λ))
```


### Prepare data {#prepare-data}

we then collect the data used in modeling into separate dicts. Because
the predictors `X`, and the outcomes, `y`, are treated fundamentally
differently by our models, we also kept them separated. To avoid
numerical overflow we have rescaled `traps`.

```python
sp_X = {
    'traps': pest_data.traps.values / 10,
}
sp_y = {
    'y': pest_data.complaints.values
}
```

To test our model we also create a dict of fake predictors. The fake
predictors would have the same value range as the real data, but more
uniformly distributed to avoid complicate geometries in the data. Also
the number of fake data is not necessarily equal that of the real data;
as a matter of fact we have generated more fake predictors than real
data so that we'll have enough data to guarantee the model works well
and if the model doesn't work well, we can rule out (to a certain degree
of course) that the problem is with our data, so we can focus our
attention on debugging the model.

```python
fobs_shape = [1000]
sp_fX = {
    'traps': random.randint(rng, fobs_shape, 0, 15) / 10,
}
```

And we'll also prepare a dict of fake parameters for the model for model
soundness check.

```python
sp_params = {
    'α': jnp.ones(1),
    'β': -1 * jnp.ones(1)
}
```


### Prior predictive check {#prior-predictive-check}

Because the model is generative, after specifying the model, we can
immediately generate samples from it.

```python
from numpyro.infer import Predictive

def get_prior_pred(model, X, n=1000):
    """Get prior predictive samples from a generative model.
    model: the unconditional model
    X: a dict of predictors
    n: the number of prior samples
    Returned: a dict of model prior samples
    """
    rng_key = random.PRNGKey(10)
    return Predictive(model, num_samples=n)(rng_key, **X)
```

```python
sp_prior_pred = get_prior_pred(sp, sp_X)
sp_prior_pred.keys()
```

```text
dict_keys(['y', 'α', 'β', 'λ'])
```

First let's see what the model predictions are like.

```python
sns.kdeplot(x=sp_prior_pred['α'].flatten());
```

{{< figure src="/ox-hugo/a2ef32db5d13659444978cf273ac1164cdcc0c30.png" >}}

```python
sns.kdeplot(x=sp_prior_pred['β'].flatten());
```

{{< figure src="/ox-hugo/2df45a905b749468b0cec00c3a4f6971950a5de3.png" >}}

There is no surprise here, the kde estimations of \\(\alpha\\) and \\(\beta\\)
correspond to their prior distributions.

```python
sp_prior_pred['y'].max()
```

```text
DeviceArray(6872, dtype=int64)
```

Looks like we have some rare but very large predictions, this is in line
with our earlier concern: when using the exponential transformation,
numerical stability can be a potential problem. But luckily through
rescaling the data we have avoided that. But we'll clip the
visualisation of the prediction for the outcome variable.

```python
def clipped_kde(d, cut=50, color=None):
    """KDE plot clipped between [0, cut].
    """
    d = d.flatten()
    sns.kdeplot(x=d[d <= cut], color=color)
    plt.xlim(left=0, right=cut);

clipped_kde(sp_prior_pred['y'])
```

{{< figure src="/ox-hugo/feeea8760d9328005b73c2ee9e4dcc02e5e2566d.png" >}}

We can see that most of the predictions are zero or close to zero, with
the counts grow exponentially smaller as the number gets larger. But in
general these predictions look reasonable, so we'll move on to model
inference.


### Fit model to fake data {#fit-model-to-fake-data}

However, before doing inference on real data, we should check that our
model works well with simulated data. We'll simulate data according to
the model and then check that we can sufficiently recover the parameter
values used in the simulation.

Here we'll fix the values of the parameters and draw one sample from the
model, then feed the sample back to the model, to see if we can recover
the fixed parameters that has been used to generate that exact data. I
collected the code for running inference and predictions into helper
functions so we can reuse them later.

```python
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS

def run_mcmc(model, X, Y, nwarmup=2000, nsample=2000, nchain=2, no_det=True):
    """Run MCMC inference with a NUTS kernel, and print summary statistics.
    model: a generative model, not conditioned on data.
    X: a dict of predictors.
    Y: a dict of outcome variables.
    no_det: Boolean, exclude deterministic variables in summary"""
    kernel = NUTS(condition(model, data=Y))
    rng_key = random.PRNGKey(10)
    mcmc = MCMC(kernel, num_warmup=nwarmup, num_samples=nsample, num_chains=nchain, progress_bar=False)
    mcmc.run(rng_key, **X, extra_fields=('potential_energy', 'diverging'))
    mcmc.print_summary(exclude_deterministic=no_det)
    return mcmc
```

```python
def recover_params(model, params, X, sites=['y'], **mc_params):
    """Check parameter recovery of a generative model using MCMC.
    model: the model
    params: the fixed params
    X: a dict of predictors
    sites: the list of outcome variable names
    """
    rng_key = random.PRNGKey(10)
    pred = Predictive(
        model,
        posterior_samples=params,
        batch_ndims=0,
        return_sites=sites)(rng_key, **X)
    return run_mcmc(model, X, pred, **mc_params)
```

Now see how the inference works

```python
recover_params(sp, sp_params, sp_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      1.01      0.04      1.01      0.94      1.08   1124.09      1.00
         β     -1.01      0.06     -1.01     -1.11     -0.90   1087.94      1.00

Number of divergences: 0
```

The results seem excellent. We can move on to the real data.


### Fit model to real data {#fit-model-to-real-data}

we now fit the model to the actually observed data.

```python
sp_mcmc = run_mcmc(sp, sp_X, sp_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.59      0.15      2.59      2.35      2.84    798.98      1.00
         β     -1.93      0.23     -1.93     -2.30     -1.55    806.49      1.00

Number of divergences: 0
```

The inference result is not as accurate as with the fake data, but this
is to be expected, because with the fake data we have way more observations.
But as we suspected, the number of bait stations set in a building is
negatively associated with the number of complaints that were made in
the following month.


### Posterior predictive check {#posterior-predictive-check}

After fitting the model, we can make some predictions. we can make
predictions for the original data, or for some new data; but since here
we are trying to compare the model prediction with observed data, we'll
make predictions using the original predictors. Later when we want to
make predictions for the future, we can also use new predictors.

```python
def get_posterior_pred(model, X, mcmc, sites=None):
    """Get posterior predictive samples from a generative model.
    model: the unconditioned model
    X: a dict of predictors
    mcmc: the inferred MCMC object
    sites: the list of outcome variables names
    """
    rng_key = random.PRNGKey(10)
    return Predictive(model, posterior_samples=mcmc.get_samples(), return_sites=sites)(rng_key, **X)

sp_posterior_pred = get_posterior_pred(sp, sp_X, sp_mcmc, sites=['y'])
```


### Density overlay {#density-overlay}

Plot the observations against the predictions

```python
def overlay_kde(y, y_pred, n=200):
    """
    Overlay KDE plots. observed vs. prediction
    """
    for i in range(n):
        sns.kdeplot(x=y_pred[i], color=colors[0])
        sns.kdeplot(x=y, color=colors[1])
        plt.xlim(left=0);

overlay_kde(sp_y['y'], sp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/531a57ad68acc681249ba7726f8d2bca1e0e65b9.png" >}}

Comparing the densities of the observed and predicted data is the most
basic posterior predictive check. If the overall shape of the
distributions does not match, it makes no sense to go into more details
and into the subgroups and special characteristics of the data. Here we
can see that the model under predicts smaller values, over-predict
middle range values, and again under-predict large values. How can we
improve on this? The most obvious choice, is to add more predictors, so
as to increase the variation in the data.


### Specific statistics {#specific-statistics}

Apart from looking at the overall distribution of the prediction, we can
also look at specific statistics **of the overall distribution**. For
example, the proportion of zeros in the data

```python
prop_zero = lambda t: (t == 0).sum(-1) / t.shape[-1]
prop_large = lambda t: (t > 10).sum(-1) / t.shape[-1]
```

```python
def ppc_stat(fn, true, pred):
    """
    Plot certain statistic, true value vs. model prediction
    fn: function to calculate the statistic
    """
    plt.hist(x=fn(pred), color=colors[0])
    plt.axvline(fn(true), color=colors[1], lw=4)
    plt.xlim(left=0)
    plt.title("True vs Predicted");
```

```python
ppc_stat(prop_zero, sp_y['y'], sp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/2d4344295a4c6db02e917d134717a8c49e1fd78c.png" >}}

The plot above shows the observed proportion of zeros (thick vertical
line) and a histogram of the proportion of zeros in each of the
simulated data sets. It is clear that the model does not capture this
feature of the data well at all.

Or we can look at the proportion of large values

```python
ppc_stat(prop_large, sp_y['y'], sp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/e36182613c5ba7e05757bad8dfebb413e48bc7d4.png" >}}

As we saw from the density overlay, the proportion of large values is
also underestimated. So our model is underestimating the tails of the distribution. Helpfully further modeling improvements might help on this.


## Expanding the model: multiple predictors {#expanding-the-model-multiple-predictors}

The simple Poisson model does not work very well so as an instinctive
follow up, we might consider adding more predictor variables. Moreover,
the manager told us that they expect there be a number of other reasons
that one building might have more roach complaints than another. We'll
incorporate this information into our model.

Currently, our model's mean parameter is a rate of complaints per 30
days, but we're modeling a process that occurs over an area as well as
over time. We have the square footage of each building, so if we add
that information into the model, we can interpret our parameters as a
rate of complaints per square foot per 30 days. A quick test shows us
that there appears to be a relationship between the square footage of
the building and the number of complaints received:

```python
sns.lmplot(x='total_sq_foot', y='complaints', data=pest_data);
```

{{< figure src="/ox-hugo/e58594b62872e3179a407e407d43aee6b741b702.png" >}}

Using the property manager's intuition, we also include an extra pieces
of information we know about the buildings - whether there is a live in
super or not - into the model.

```python
sns.violinplot(x='live_in_super', y='complaints', data=pest_data);
```

{{< figure src="/ox-hugo/3a3279941d7275ada97b24ad801bc21067024125.png" >}}

```python
sns.scatterplot(x='traps_jitter', y='complaints_jitter', hue='live_in_super', data=pest_data);
```

{{< figure src="/ox-hugo/fe7dc5963a7062077526e1fd5c4535f076507fab.png" >}}


### Model {#model}

And the model now becomes

\begin{aligned}
\textrm{complaints}\_{n} & \sim \textrm{Poisson}(\textrm{sq\_foot}\_b\lambda\_{n}) \\\\\\
\lambda\_{n} & = \exp{(\eta\_{n} )} \\\\\\
\eta\_{n} &= \alpha + \beta  \textrm{traps}\_{n}  + \beta\_{super}  \textrm{live\_in\_super}\_{b}
\end{aligned}

The term \\(\text{sq\_foot}\\) is called an exposure term. If we log the
term, we can put it in \\(\eta\_{n}\\):

\begin{aligned}
\textrm{complaints}\_{n} & \sim \textrm{Poisson}(\lambda\_{n}) \\\\\\
\lambda\_{n} & = \exp{(\eta\_{n} )} \\\\\\
\eta\_{n} &= \alpha + \beta  \textrm{traps}\_{n}  + \beta\_{super}  \textrm{live\_in\_super}\_{b} + \textrm{log\_sq\_foot}\_b
\end{aligned}

We'll use the same priors for \\(\alpha\\) and \\(\beta\\) as before, as for
\\(\beta\_{\text{super}}\\), we suspect that having a super might reduce that
amount of complaints so

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\beta\_{\text{super}} & \sim \textrm{Normal}(-0.5, 2) \\\\\\
\end{aligned}

Now we can write the model

<a id="code-snippet--model-mp"></a>
```python
def mp(traps, live_in_super, log_sq_foot):
    """Multiple Poisson Regression"""

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    β_super = sample('β_super', Normal(-0.5, 2))

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + β * traps + β_super * live_in_super + log_sq_foot))
        return sample('y', Poisson(λ))
```


### prepare the data {#prepare-the-data}

Notice that the predictors have changed but we're still using the same
outcome variable. As a matter of fact, we'll be using the same outcome
variable throughout this study.

```python
pest_data['log_sq_foot'] = jnp.log(pest_data.total_sq_foot.values / 1e4)
```

We also demean the data for computational stability.

```python
mp_X = {
    'traps': pest_data.traps.values / 10,
    'live_in_super': pest_data.live_in_super.values - pest_data.live_in_super.mean(),
    'log_sq_foot': pest_data.log_sq_foot.values - pest_data.log_sq_foot.mean(),
}
```

and some fake data

```python
mp_fX = {
    'traps': random.randint(rng, fobs_shape, 0, 15) / 10,
    'live_in_super': random.bernoulli(rng, 0.5, fobs_shape) - 0.5,
    'log_sq_foot': random.normal(rng, fobs_shape) * 0.4
}
```

and fake parameters

```python
mp_params = {
    'α': jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'β_super': jnp.array([-0.5])
}
```


### Prior predictive check {#prior-predictive-check}

Like before, we can generate samples from the prior joint distribution.

```python
mp_prior_pred = get_prior_pred(mp, mp_X)
```

```python
mp_prior_pred['y'].max()
```

```text
DeviceArray(7746, dtype=int64)
```

```python
clipped_kde(mp_prior_pred['y'])
```

{{< figure src="/ox-hugo/6aebc55433b019979d5a79fb223b67f93470b289.png" >}}

Judging by the prior prediction, this model does not seem to vary much
from the previous one, but the predictions are still reasonable, though
with a much wider range than the observed data (and this is what want).


### Fit model to fake data {#fit-model-to-fake-data}

Now see how the inference goes

```python
recover_params(mp, mp_params, mp_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      1.04      0.04      1.04      0.98      1.11   1499.80      1.00
         β     -1.04      0.06     -1.04     -1.14     -0.95   1768.77      1.00
   β_super     -0.53      0.06     -0.52     -0.62     -0.43   2143.09      1.00

Number of divergences: 0
```

We've recovered the parameters sufficiently well, so we are now
confident to fit the real data.


### Fit model to real data {#fit-model-to-real-data}

Now let's use the real data and explore the fit.

```python
mp_mcmc = run_mcmc(mp, mp_X, sp_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.58      0.20      2.58      2.25      2.90    889.14      1.00
         β     -2.04      0.30     -2.04     -2.51     -1.55    910.62      1.00
   β_super     -0.26      0.13     -0.25     -0.47     -0.05   1300.00      1.00

Number of divergences: 0
```


### Posterior predictive check {#posterior-predictive-check}

With the simple Poisson model, the predictions don't match the observed
data. Let's see if we have made any improvement.

```python
mp_posterior_pred = get_posterior_pred(mp, mp_X, mp_mcmc, sites=['y'])
```


### Density overlay {#density-overlay}

```python
overlay_kde(sp_y['y'], mp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/50bd78b5ddd58e7fd7124211909c984db21cb753.png" >}}

Looks much better than before because the real data lay somewhat in between the predictions. Still, the disparity is clear.


### Specific statistics {#specific-statistics}

And prediction for small and large values

```python
ppc_stat(prop_zero, sp_y['y'], mp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/f7e0c246a96a1801895db2ff5ee07809a0ebb956.png" >}}

```python
ppc_stat(prop_large, sp_y['y'], mp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/24109ebf26edabd699390ea6866ff722516e4eb1.png" >}}

Compared with the simple Poisson model, we have definitely made some
progress, both with the overall distribution and the specific
statistics; but it looks like we are still under-estimating the variance
in the data. We can keep adding new predictors to increase variation in
the data(if we can find more predictors), but at this stage we might
just consider choosing another distribution that allows more variance
than Poisson. We turn next to the negative binomial distribution.


## Modeling overdispersed data: Negative Binomial distribution {#modeling-overdispersed-data-negative-binomial-distribution}

When modeling the data using a Poisson distribution, we saw that the
model didn't fit as well to the data as we would like. In particular the
model underpredicted low and high numbers of complaints, and
overpredicted the medium number of complaints. This is one indication of
over-dispersion, that is, the observed data have more variance than the
distribution used to model them. A Poisson model doesn't fit
over-dispersed count data very well because the same parameter \\(\lambda\\)
controls both the expected counts and the variance of these counts. The
natural alternative to this is the negative binomial model.

NumPyro doesn't have a native implementation of the Negative Binomial
distribution, but we have two alternatives: the Tensorflow
implementation that can be imported into NumPyro, or another
parameterisation known as the Gamma-Poisson distribution.

```python
from numpyro.contrib.tfp.distributions import NegativeBinomial
```

The Poisson distribution can be interpreted as the number of successes
happening at a certain rate, across certain time period and certain
space, as we are using here; it can also be interpreted as the sum of
Bernoulli trials with a constant probability of success. Similarly, the
Negative Binomial distribution (in NumPyro, because there are many
different variations) can be interpreted as the sum of Bernoulli trials,
but as the sum of successes we have before a certain number of failures
is reached. Like the Poisson, it also has support on all the natural
numbers. In this interpretation, the Negative Binomial distribution has
two parameters, the failure count \\(k\\), and the probability of success,
\\(p\\). And for \\(\text{NegativeBinomial}(k, p)\\), the mean and variance are

\begin{aligned}
\mathbb{m} &= \frac{k}{1-p} - k = \frac{pk}{1-p} \\\\\\
\mathbb{v} &=  \frac{p}{(1-p)}  \frac{k}{(1-p)} = \frac{pk}{(1-p)^2}
\end{aligned}

Although this parameterisation is easy to understand, it's not very
interpretable for us when using it to model count data, and with a
linear model for the rate parameter, like we did for the Poisson model.
One parameterisation that meets such demands is to augment the Poisson
distribution with a **concentration** parameter, \\(\phi\\), as
\\(\text{Neg-Binomial}(y; \lambda, \phi)\\), so that we can separate the
modeling of mean and variance. Using this parameterisation, the mean and
variance are

\begin{aligned}
\mathbb{m}  &= \lambda \\\\\\
\mathbb{v}  &= \lambda + \lambda^2 / \phi .
\end{aligned}

As we can see, the mean is the same as before, but now the variance is
always bigger than the mean: this is how we meet the over-dispersion
demand. As \\(\phi\\) gets larger, the distribution becomes more
concentrated, the term \\(\lambda^2 / \phi\\) approaches zero, the variance
of the negative-binomial approaches \\(\lambda\\), and the Negative Binomial
gets closer and closer to the Poisson.

Combining the two representation, we can model the intensity parameter
as before, and along with the new concentration parameter, we can obtain
the parameterisation used in the TFP implementation

\begin{aligned}
k  &= \phi \\\\\\
p  &= \frac{\lambda }{  \lambda + \phi} \\\\\\
\end{aligned}

The Tensorflow implementation of Negative Binomial also accepts the log
odds parameterisation, which is more convenient for us, and also
numerically more stable:

\\[\log \frac{p}{1-p} = \log \lambda - \log \phi\\]

```python
k, p = 3.5, 0.3
nb = NegativeBinomial(total_count=k, probs=p)

nb.mean(), k*p / (1-p), nb.variance(), k*p / (1-p)**2
```

|             |                     |                    |             |                          |                    |
|-------------|---------------------|--------------------|-------------|--------------------------|--------------------|
| DeviceArray | (1.5 dtype=float32) | 1.5000000000000002 | DeviceArray | (2.142857 dtype=float32) | 2.1428571428571432 |

There is another implementation of the Negative Binomial distribution,
natively implemented in NumPyro, that directly augments the Poisson
distribution, and is called the Gamma-Poisson distribution. Our goal is
to use a distribution to model the count data, with over-dispersion, the
natural choice, if one Poisson distribution with constant \\(\lambda\\) is
too limited, is to use a family of Poissons with varying \\(\lambda\\).
Since \\(\lambda\\) is a positive real number, we may consider putting a
Gamma distribution on it, and this leads to the compound Gamma-Poisson
distribution. It turns out that this is another parameterisation of the
Negative Binomial distribution.

The Gamma-Poisson distribution implementation in NumPyro is
parameterised with a concentration parameter `c` and a rate parameter
`r`, but have nothing to do with the concentration and rate parameters
before. With this parameterisation the parameters are defined as

\begin{aligned}
c &= k \\\\\\
r &= \frac{1-p}{p}\\\\\\
\end{aligned}

Or, using the rate-concentration parameterisation

\begin{aligned}
c  &= \phi \\\\\\
r  &= \frac{\phi}{\lambda} \\\\\\
\end{aligned}

and the mean and the variance can aslo be calculated

\begin{aligned}
\mathbb{m} &=  \frac{c}{r} \\\\\\
\mathbb{v} &=  \frac{ c (1+r) }{r^2}
\end{aligned}

```python
from numpyro.distributions import GammaPoisson

c, r = 3.5, 0.7/0.3
gap = GammaPoisson(c,r)
gap.mean, c/r, gap.variance, c*(1+r)/r**2
```

|     |     |             |                            |                   |
|-----|-----|-------------|----------------------------|-------------------|
| 1.5 | 1.5 | DeviceArray | (2.14285714 dtype=float64) | 2.142857142857143 |

Another difference with the native NumPyro implementation, is that the
mean and variance are attributes while with the Tensorflow one they are
methods. Because this natively implemented version works better with
other NumPyro functionalities, we'll use this distribution to model our
data.


### Model {#model}

using the mathematical manipulation above, we can write the Gamma
Poisson model as:

\begin{aligned}
\text{complaints}\_{n} & \sim \text{Gamma-Poisson}(c, r\_{n}) \\\\\\
c  &= \phi \\\\\\
r\_{n}  &= \frac{\phi}{\lambda\_n} \\\\\\
\log \lambda\_{n} &= \alpha + \beta  {\rm traps}\_{n} + \beta\_{\rm super}  {\rm super}\_{b} + \text{log\_sq\_foot}\_{b}
\end{aligned}

We don't have much information on what \\(\phi\\) should be, apart from it's
positive, we'll put a Log-normal distribution on it.

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\beta\_{\text{super}} & \sim \textrm{Normal}(-0.5, 1) \\\\\\
\phi & \sim \textrm{LogNormal}(0, 1) \\\\\\
\end{aligned}

now translate the mathematical model into code

<a id="code-snippet--model-gap"></a>
```python
from numpyro.distributions import LogNormal

def gap(traps, live_in_super, log_sq_foot):
    """Multiple Gamma-Poisson Regression"""

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    β_super = sample('β_super', Normal(-0.5, 1))
    φ = sample('φ', LogNormal(0, 1))

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + β * traps + β_super * live_in_super + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare the data {#prepare-the-data}

Both the predictors and the outcome are the same as before, we only have
to prepare the fake parameters

```python
gap_params = {
    'α': jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'β_super': jnp.array([-0.5]),
    'φ': jnp.ones(1)
}
```


### Prior predictive check {#prior-predictive-check}

Like before, we can generate samples from the prior joint distribution.
The predictors used are the same as the previous multiple Poisson model.

```python
gap_prior_pred = get_prior_pred(gap, mp_X)
```

The Log-Normal prior for \\(\phi\\)

```python
sns.histplot(gap_prior_pred['φ']);
```

{{< figure src="/ox-hugo/0d47ed1ea464d6570decbfc5b6e2fd142ee8778e.png" >}}

```python
gap_prior_pred['y'].max()
```

```text
DeviceArray(22352, dtype=int64)
```

The prediction now has a much wider range.

```python
clipped_kde(gap_prior_pred['y']);
```

{{< figure src="/ox-hugo/73b28360811294963cf8b291a74badfb5fb04478.png" >}}


### Fit the model to fake data {#fit-the-model-to-fake-data}

```python
recover_params(gap, gap_params, mp_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      1.02      0.08      1.02      0.89      1.14   2183.83      1.00
         β     -1.02      0.10     -1.02     -1.18     -0.85   2163.81      1.00
   β_super     -0.63      0.09     -0.63     -0.77     -0.49   2367.72      1.00
         φ      0.91      0.08      0.91      0.79      1.03   3129.85      1.00

Number of divergences: 0
```

The results seems okay. Move on.


### Fit model the real data {#fit-model-the-real-data}

Now let's use the real data and explore the fit.

```python
gap_mcmc = run_mcmc(gap, mp_X, sp_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.60      0.39      2.60      1.98      3.27   1293.60      1.00
         β     -2.05      0.56     -2.04     -2.99     -1.14   1304.76      1.00
   β_super     -0.25      0.23     -0.24     -0.61      0.12   1757.55      1.00
         φ      1.59      0.34      1.55      1.04      2.10   2136.49      1.00

Number of divergences: 0
```


### Posterior predictive check {#posterior-predictive-check}

```python
gap_posterior_pred = get_posterior_pred(gap, mp_X, gap_mcmc, sites=['y'])
```


### Density overlay {#density-overlay}

```python
overlay_kde(sp_y['y'], gap_posterior_pred['y'])
```

{{< figure src="/ox-hugo/b1b248497dcd9e5ab0ae275693e03b2236165a52.png" >}}

The prediction looks much better now, the real data is well in the range
of model predictions. We have captured the overall distribution of the
data quite well.


### Specific statistics {#specific-statistics}

And let's also look at the proportion of small and large values.

```python
ppc_stat(prop_zero, sp_y['y'], gap_posterior_pred['y'])
```

{{< figure src="/ox-hugo/96a94b8e2bb700a5765b74795703839b83f3e6f2.png" >}}

Now our model captures the proportion of zeros almost perfectly. It's
true that there is still large uncertainty, the predicted proportion of
zeros has a wide range, but we have limited data and there is only that
much we can learn from it. To improve our model prediction, we can
either find more observations, or find more useful predictors for our
existing observations.

```python
ppc_stat(prop_large, sp_y['y'], gap_posterior_pred['y'])
```

{{< figure src="/ox-hugo/c64fd9453b1d2b6e5e0a93056f6887be1ad80378.png" >}}

The proportion of larger predicted values also seems reasonable. So
using the Gamma-Poisson distribution with increased dispersion indeed
fits the data much better.

Now that we have captured the overall distribution of the observations,
what next? We should look closer to the predictions to see if the model
prediction has other problems.

One general way to do this is using the residual plots.


### posterior prediction by groups {#posterior-prediction-by-groups}

We haven't used the fact that the data are clustered by building and
time yet. A posterior predictive check might elucidate whether it would
be a good idea to add these information into the model. Here we are
using information that is known to us but unknown to the model, to see
if the model prediction generalises well.

```python
def plot_grouped_mean(obs, pred, group, nrow, ncol, figsize):
    """Plot the grouped prediction against observation"""
    fig, axes = plt.subplots(nrow, ncol, sharex=False, sharey=True, figsize=figsize)
    ids = group.unique()
    ids.sort()

    for i, id in enumerate(ids):
        ax = axes.flatten()[i]
        mask = group == id
        y = obs[mask]
        p = pred.T[mask.values]
        assert len(y) == p.shape[0]
        y_mean = y.mean()
        p_mean = p.mean(0)
        sns.histplot(x=p_mean, color=colors[0], ax=ax)
        ax.axvline(y_mean, color=colors[1], lw=2)
        ax.set_title(id)
        plt.tight_layout()
```

Let's look at the model prediction, grouped by the buildings and by the time.

<a id="code-snippet--vis-gap-building"></a>
```python
plot_grouped_mean(pest_data['complaints'], gap_posterior_pred['y'], pest_data['building_id'], 2, 5, (12,4))
```

{{< figure src="/ox-hugo/fa8d4707297f4478a59534401a6bb64ea79a6d57.png" >}}

<a id="code-snippet--vis-gap-month"></a>
```python
plot_grouped_mean(pest_data['complaints'], gap_posterior_pred['y'], pest_data['month'], 3, 4, (12,6))
```

{{< figure src="/ox-hugo/04b06f70b6ad3575bd66716ee89b022f17ab6b39.png" >}}

We're getting plausible predictions for most building means but some are
estimated better than others and some have larger uncertainties than we
might expect. And most of the predictions grouped by time are a bit off. If we explicitly model the variation across buildings we
may be able to get much better estimates.


## Going hierarchical: Modeling varying intercepts for each building {#going-hierarchical-modeling-varying-intercepts-for-each-building}

We have been able to fit the overall distribution of the observed data
quite well, but as the model criticism in the previous part has shown,
there is a large amount of uncertainty around our predictions, and some
per group predictions are a bit off. Because we know that our data have
some natural groupings, for example, some observations are from the same
building, and some observations are from the same time period; these
grouping information can help us reduce the variance, and make better predictions, so we'd also
like to incorporate these information into our model.

First let's focus on the grouping by buildings. Since the complaints
made by the inhabitants are closely related to the conditions of each
building, we'd expect there be some differences among the buildings that
are contributing to the complaints we received. As such, it's natural to
expect the intercepts of the linear model for each building be
different, because the intercept captures what are not modeled by the
other included variables; we'll start by adding a different intercept
parameter, \\(\pi\\), to each building, and since we expect these intercepts
be different but nevertheless related (they are all tenant buildings
after all), we'll also put a common prior on these intercepts. This
leads us to the hierarchical intercept model.

Because of this formulation, during the inference process we will not only learn about the parameters for each building, but the hyperparameters that regulate the parameters. These formulation also helps us to make predictions for new buildings, if we later have information about these new buildings.

Since we have already a common intercept \\(\alpha\\), the second intercept will be modeled as a **variation** to the first and thus has zero mean.

\begin{aligned}
\log \lambda\_{n} &= \alpha + \pi\_b + \beta  {\rm traps}\_{n} + \beta\_{\rm super}  {\rm super}\_{b} + \text{log\_sq\_foot}\_{b} \\\\\\
\pi &\sim \text{Normal}(0, \sigma) \\\\\\
\end{aligned}

The subscription in \\(\pi\_b\\) is the index. Some of our predictors vary
only by building, so we can rewrite the above model more efficiently
like so:

\begin{aligned}
\log \lambda\_{n}  &= \alpha + \pi\_b + \beta  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\pi &\sim \text{Normal}(\beta\_{\text{super}}   \text{super}\_b , \sigma) \\\\\\
\end{aligned}

We have more information at the building level as well, like the average
age of the residents, the average age of the buildings, and the average
per-apartment monthly rent so we can add that data into a matrix called
`building_data`, which will have one row per building and four columns:

-   `live_in_super`
-   `age_of_building`
-   `average_tentant_age`
-   `monthly_average_rent`


### Model {#model}

And the final model is

\begin{aligned}
\text{complaints}\_{n} & \sim \text{Gamma-Poisson}(c, r\_{n}) \\\\\\
c  &= \phi \\\\\\
r\_{n}  &= \frac{\phi}{\lambda\_n} \\\\\\
\log \lambda\_{n}  &= \alpha + \pi\_b + \beta  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\pi &\sim \text{Normal}(\texttt{building\_data}  \zeta, \sigma) \\\\\\
\end{aligned}

Now we have to choose a prior for \\(\sigma\\). Because \\(\pi\\) is the
deviation of each building from the common intercept, and from the
earlier models we know that the intercept seems to be between 2-3, which
prior should we give \\(\sigma\\)? Looking at the shape of the Log-Normal
distribution

{{< figure src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/PDF-log%5Fnormal%5Fdistributions.svg/240px-PDF-log%5Fnormal%5Fdistributions.svg.png" >}}

Because we are only modeling the deviation, and each building shouldn't
deviate from the common mean very much, we can choose a prior with large
mass on small values but also a heavy tail for safety sake. Both
\\(\sigma=0.5\\) and \\(\sigma=1\\) have heavy tails but let's go with
\\(\sigma=1\\) because it has heavier tail.

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\phi & \sim \textrm{LogNormal}(0, 1) \\\\\\
\zeta & \sim \textrm{Normal}(0, 1) \\\\\\
\sigma & \sim \textrm{LogNormal}(0, 1) \\\\\\
\end{aligned}

<a id="code-snippet--model-vi"></a>
```python
def vi(traps, log_sq_foot, b, building_data):
    """Multiple Gamma-Poisson Regression
    With Varying Interceps, centred parameterisation
    """

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    φ = sample('φ', LogNormal(0, 1))
    σ = sample('σ', LogNormal(0, 1))

    with plate('N_predictors', size=building_data.shape[1]):
        ζ = sample('ζ', Normal(0, 1))

    with plate('N_buildings', size=building_data.shape[0]):
        π = sample('π', Normal(building_data @ ζ, σ))

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + π[b] + β * traps + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare the data {#prepare-the-data}

We need the building index for each observation

```python
pest_data['b'] = pest_data.building_id.astype('category').cat.codes
```

The building level predictors

```python
bd = pest_data[['b', 'live_in_super', 'age_of_building', 'average_tenant_age', 'monthly_average_rent']]
bd = bd.drop_duplicates().sort_values(by='b').set_index(keys='b')

bd = (bd - bd.mean())
bd['age_of_building'] = bd['age_of_building'] / 10
bd['average_tenant_age'] = bd['average_tenant_age'] / 10
bd['monthly_average_rent'] = bd['monthly_average_rent'] / 1000

print(bd)
```

```text
   live_in_super  age_of_building  average_tenant_age  monthly_average_rent
b
0           -0.3            -0.04           -0.206569             -0.658587
1           -0.3            -0.64           -0.035733             -0.123740
2           -0.3             0.16           -0.258725              0.331981
3           -0.3            -0.24            0.395347              0.159693
4           -0.3             0.96           -0.878173              0.183439
5           -0.3            -0.04           -0.137937             -0.309452
6           -0.3             1.06            0.853178              0.152982
7            0.7            -0.14            1.525874              0.176402
8            0.7            -0.04           -0.478690             -0.010646
9            0.7            -1.04           -0.778571              0.097928
```

All the predictors

```python
vi_X = {
    'traps': pest_data.traps.values / 10,
    'log_sq_foot': pest_data.log_sq_foot.values - pest_data.log_sq_foot.mean(),
    'b': pest_data.b.values,
    'building_data': bd.values
}
```

Fake predictors

```python
vi_fX = dict(
    traps = random.randint(rng, fobs_shape, 0, 15) / 10,
    log_sq_foot = random.normal(rng, fobs_shape) * 0.4,
    b = random.randint(rng, fobs_shape, 0, 10),
    building_data = random.normal(rng, [10, 4])
)
```

Fake parameters

```python
vi_params = {
    'α': jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'φ': jnp.ones(1),
    'σ': jnp.ones(1),
    'ζ': jnp.ones(4)
}
```


### prior predictive check {#prior-predictive-check}

Like before, we can generate samples from the prior joint distribution.
But we can also check the prediction for the hierarchical groups to
better understand the model.

```python
vi_prior_pred = get_prior_pred(vi, vi_X)
vi_prior_pred.keys()
```

```text
dict_keys(['y', 'α', 'β', 'ζ', 'λ', 'π', 'σ', 'φ'])
```

```python
vi_prior_pred['λ'].max()
```

```text
DeviceArray(6.97069025e+19, dtype=float64)
```

This doesn't look very good: there are extremely big and thus
unreasonable rate parameters. With values this big, it's way beyond the
region of reasonableness, and we are likely to have numerical problems.
let's look at the prior distribution.

```python
clipped_kde(vi_prior_pred['λ']);
```

{{< figure src="/ox-hugo/464069865bcbdedf66f37cf57ab1459fe577b131.png" >}}

We can see that most of the \\(\lambda\\)s are small, but we have a very
long tail.

```python
vi_prior_pred['y'].min(), vi_prior_pred['y'].max()
```

|             |                                    |             |                                   |
|-------------|------------------------------------|-------------|-----------------------------------|
| DeviceArray | (-9223372036854775808 dtype=int64) | DeviceArray | (2096574335488622592 dtype=int64) |

```python
jnp.sum(vi_prior_pred['y'] < 0)
```

```text
DeviceArray(15, dtype=int64)
```

As we can see here, we're having numerical overflow. Let's see where the problem is happening

```python
jnp.where(jnp.any(vi_prior_pred['y'] < 0, axis=1))
```

|             |                               |
|-------------|-------------------------------|
| DeviceArray | ((9 600 843 916) dtype=int64) |

we have `jnp.where(jnp.any(vi_prior_pred['y'] < 0, axis=1))[0].shape[0]` `4` samples having overflow

```python
for id in jnp.where(jnp.any(vi_prior_pred['y'] < 0, axis=1))[0]:
    print('sample ', id)
    for k, v in vi_prior_pred.items():
        if k != 'y' and k != 'λ':
            print(k, v[id])
```

```text
sample  9
α -2.076938797918082
β -3.040553873727225
ζ [-0.63783181  0.43989969 -0.53982382  1.25278584]
π [-1.44801680e+00  3.52982695e-01 -3.88033657e-01 -1.26292911e+00
  3.02934036e+00 -2.23284593e+00  1.30826991e+00 -1.95811861e+00
  1.02840308e-03  1.93390199e+00]
σ 2.221601811819647
φ 0.04269677444465673
sample  600
α -0.6539770032396253
β -3.064939454031821
ζ [-0.13668298 -0.05067456  0.68447729  0.51528061]
π [-30.16197783  48.47399513 -14.07733496 -35.8871944   20.70903837
 -18.1317607   38.97131469  26.94894094   2.34846096 -18.93444801]
σ 22.956083097460926
φ 0.24774948672210043
sample  843
α 0.5300731714992636
β 0.11640732396070363
ζ [ 0.36483016 -2.38586066 -0.94384306 -1.16709979]
π [ 0.38850317  1.70733028 -1.9398627   2.15661211 -0.40266936 -0.47976534
 -3.47803733 -0.92939717  1.09012755  3.0866801 ]
σ 0.8629543456132125
φ 0.04311855506177399
sample  916
α -0.013437712439809625
β 0.004238195061725909
ζ [-0.3532105  -0.73562442  0.49912101 -0.96486003]
π [ 0.9844654   0.15567474 -1.36270611 -0.96382073 -1.82849159  0.76609813
 -0.24277404  0.12431519 -1.04068427 -0.64700478]
σ 0.710783024885015
φ 0.04326221274190198
```

As we can see, either we are having too big \\(\sigma\\) which in turn
leads to extreme values in \\(\pi\\), or we are having too small \\(\phi\\). We should probably revise the model to
restraint them a little bit. But for the moment we'll just carry on.
But still, we can already see that as our model gets more complex, it's
important to check prior predictive distributions to catch problems
early on.


### Fit the model to fake data {#fit-the-model-to-fake-data}

Now we fix the parameters

```python
recover_params(vi, vi_params, vi_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      0.64      0.43      0.63     -0.03      1.29    563.31      1.01
         β     -1.00      0.11     -1.00     -1.17     -0.83   2856.13      1.00
      ζ[0]      0.79      0.39      0.82      0.15      1.40   1763.54      1.00
      ζ[1]      0.91      0.42      0.93      0.25      1.58   1254.77      1.00
      ζ[2]      0.63      0.32      0.62      0.09      1.14   1949.56      1.00
      ζ[3]      0.83      0.56      0.87     -0.09      1.71   1479.18      1.00
      π[0]     -0.62      0.46     -0.61     -1.29      0.15    604.24      1.01
      π[1]      1.31      0.43      1.31      0.65      1.98    573.69      1.01
      π[2]      4.53      0.43      4.54      3.88      5.22    559.88      1.01
      π[3]      0.96      0.44      0.96      0.30      1.67    578.04      1.01
      π[4]      0.47      0.44      0.48     -0.22      1.16    583.52      1.01
      π[5]      0.09      0.44      0.10     -0.60      0.78    581.29      1.01
      π[6]     -0.69      0.44     -0.69     -1.38      0.02    623.27      1.01
      π[7]     -0.07      0.44     -0.06     -0.78      0.60    599.43      1.01
      π[8]     -1.96      0.47     -1.95     -2.76     -1.23    689.12      1.01
      π[9]     -0.37      0.45     -0.36     -1.05      0.34    591.63      1.01
         σ      0.96      0.34      0.89      0.49      1.43   1192.16      1.00
         φ      1.08      0.09      1.07      0.94      1.23   2594.44      1.00

Number of divergences: 0
```

We are not recovering \\(\zeta\\) very well, but this is to be expected,
because at the building level we only have 10 data points, yet we have 4
\\(\zeta\\)s, 10 \\(\pi\\)s, and one \\(\sigma\\). We simply don't have enough data
to make good inference.


### Fit the real data {#fit-the-real-data}

Now let's use the real data and explore the fit.

```python
vi_mcmc = run_mcmc(vi, vi_X, sp_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.55      0.44      2.55      1.85      3.26   1705.11      1.00
         β     -1.99      0.60     -2.00     -2.97     -1.02   1773.75      1.00
      ζ[0]     -0.13      0.35     -0.13     -0.73      0.41   1953.98      1.00
      ζ[1]      0.13      0.27      0.13     -0.30      0.57   2546.58      1.00
      ζ[2]      0.04      0.21      0.04     -0.28      0.40   3014.78      1.00
      ζ[3]     -0.01      0.45     -0.00     -0.73      0.76   2714.99      1.00
      π[0]      0.18      0.25      0.17     -0.25      0.56   2288.05      1.00
      π[1]     -0.05      0.26     -0.05     -0.49      0.35   2531.30      1.00
      π[2]      0.21      0.24      0.20     -0.21      0.60   2155.54      1.00
      π[3]     -0.02      0.27     -0.02     -0.49      0.39   2334.16      1.00
      π[4]      0.19      0.28      0.18     -0.26      0.63   3091.94      1.00
      π[5]     -0.11      0.23     -0.10     -0.53      0.24   2605.33      1.00
      π[6]      0.09      0.30      0.10     -0.40      0.58   2581.49      1.00
      π[7]      0.06      0.29      0.05     -0.41      0.53   2310.20      1.00
      π[8]     -0.14      0.25     -0.15     -0.56      0.27   2528.11      1.00
      π[9]     -0.31      0.30     -0.31     -0.81      0.19   2043.78      1.00
         σ      0.31      0.17      0.28      0.05      0.54    773.08      1.00
         φ      1.60      0.35      1.56      1.03      2.13   3372.03      1.00

Number of divergences: 9
```

HMC is having divergent transitions, we might need to have a closer look at the inference result. Having divergent transitions (even only one divergence) means that the posterior hasn't been properly explored, the posterior might be biased; and thus in consequence everything we do with the inference result could be put into doubt!

```python
vi_post_samples = vi_mcmc.get_samples()
for k, v in vi_post_samples.items():
    print(k, v.shape)
```

```text
α (4000,)
β (4000,)
ζ (4000, 4)
λ (4000, 120)
π (4000, 10)
σ (4000,)
φ (4000,)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.histplot(vi_prior_pred['σ'], ax=axes[0])
sns.histplot(vi_post_samples['σ'], ax=axes[1]);
```

{{< figure src="/ox-hugo/61413bc34ed37b5c84e4878e1370be9416a64138.png" >}}

we have learned a lot from the model, but we don't know if the result can be trusted.


### Diagnostics {#diagnostics}

We get a bunch of divergent transitions, which is an indication that
there may be regions of the posterior that have not been explored by the
Markov chains. Divergences are discussed in more detail in the original
Stancon tutorial, and the [bayesplot](https://mc-stan.org/bayesplot/index.html) package in R, the [ArviZ](https://arviz-devs.github.io/arviz/) package in Python all
have many functions to visualise and analyse the divergences, and Michael Betancourt's [paper on
Hamiltonian Monte Carlo in general](https://arxiv.org/abs/1701.02434) and [diagnosing biased inference case study in specific](https://betanalpha.github.io/assets/case%5Fstudies/divergences%5Fand%5Fbias.html) contain many great insights on this topic.

Unfortunate for the didactive purpose and fortunate from a modeling
point of view, we only have two divergent cases so the plots are not
very helpful here, we'll just print out the divergent transitions and
see if we can spot any problem.

```python
jnp.where(vi_mcmc.get_extra_fields()['diverging'] == True)
```

|             |                                                          |
|-------------|----------------------------------------------------------|
| DeviceArray | ((238 319 441 944 1647 1652 1654 1657 1887) dtype=int64) |

```python
for id in jnp.where(vi_mcmc.get_extra_fields()['diverging'] == True)[0]:
    print('sample ', id)
    for k, v in vi_post_samples.items():
        if k in ['σ']:
            print(k, v[id])
```

```text
sample  238
σ 0.1613309871400151
sample  319
σ 0.05119697028673526
sample  441
σ 0.14897054679154811
sample  944
σ 0.06360336320877438
sample  1647
σ 0.038141660833332515
sample  1652
σ 0.038141660833332515
sample  1654
σ 0.038141660833332515
sample  1657
σ 0.041001380436514086
sample  1887
σ 0.0885683063876693
```

Again, we are seeing similar problems as encountered earlier, in prior predictive check. Here we are having extremely small \\(\sigma\\) which is problematic.


### Model, revised {#model-revised}

We have located two problems with our model:

1.  the prior for \\(\sigma\\) is too diffuse and we are having unreasonably large predictions
2.  when the sampled value for \\(\phi\\) is too small, the posterior tends to degenerate.

To solve these problems, we can put some more restrictive priors on these parameters, or we can reparameterise the model so that even when the parameters take extreme values, the posterior geometry is still smooth enough to not cause trouble for the HMC algorithm. We first try the first solution.

change to the model prior

\begin{aligned}
\phi & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\sigma & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\end{aligned}

<a id="code-snippet--model-vi2"></a>
```python
def vi2(traps, log_sq_foot, b, building_data):
    """Multiple Gamma-Poisson Regression
    With Varying Interceps, more restrictive prior
    """

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    φ = sample('φ', LogNormal(0, 0.5))
    σ = sample('σ', LogNormal(0, 0.5))

    with plate('N_predictors', size=building_data.shape[1]):
        ζ = sample('ζ', Normal(0, 1))

    with plate('N_buildings', size=building_data.shape[0]):
        π = sample('π', Normal(building_data @ ζ, σ))

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + π[b] + β * traps + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare the data {#prepare-the-data}

There is no change in the data.


### prior predictive check {#prior-predictive-check}

```python
vi2_prior_pred = get_prior_pred(vi2, vi_X)
```

```python
vi2_prior_pred['y'].min(), vi2_prior_pred['y'].max()
```

|             |                 |             |                       |
|-------------|-----------------|-------------|-----------------------|
| DeviceArray | (0 dtype=int64) | DeviceArray | (2128827 dtype=int64) |

Great, now there is no longer any numerical overflow

```python
clipped_kde(vi2_prior_pred['y']);
```

{{< figure src="/ox-hugo/008d2873ca4af66543797be031c9313b13d88377.png" >}}


### Model, reparameterised {#model-reparameterised}

The second solution is to sample an auxiliary variable \\(\pi\_0\\) from the standard normal
first, then transform it to \\(\pi\\), so that the sampling of \\(\sigma\\) doesn't affect the sampling of \\(\pi\\)

\begin{aligned}
\text{complaints}\_{n} & \sim \text{Gamma-Poisson}(c, r\_{n}) \\\\\\
c  &= \phi \\\\\\
r\_{n}  &= \frac{\phi}{\lambda\_n} \\\\\\
\log \lambda\_{n}  &= \alpha + \pi\_b + \beta  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\pi &= \texttt{building\_data}  \zeta + \sigma \pi\_0 \\\\\\
\end{aligned}

and the priors

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\phi & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\zeta & \sim \textrm{Normal}(0, 1) \\\\\\
\pi\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\sigma & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\end{aligned}

<a id="code-snippet--model-vincp"></a>
```python
def vincp(traps, log_sq_foot, b, building_data):
    """Multiple Gamma-Poisson Regression
    With Varying Interceps, noncentred parameterisation
    """

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    φ = sample('φ', LogNormal(0, 0.5))
    σ = sample('σ', LogNormal(0, 0.5))

    with plate('N_predictors', size=building_data.shape[1]):
        ζ = sample('ζ', Normal(0, 1))

    with plate('N_buildings', size=building_data.shape[0]):
        π0 = sample('π0', Normal(0, 1))

    π = deterministic('π', building_data @ ζ + σ * π0)

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + π[b] + β * traps + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare the data {#prepare-the-data}

We have to change the fixed parameters used for model checking. With our
models getting more complex, even assigning fake values to parameters
gets more and more difficult because the fixed parameters can take us to
some strange regions of the parameter space and beats the purpose of
model checking

```python
vincp_params = {
    'α': jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'φ': jnp.ones(1),
    'σ': jnp.ones(1),
    'ζ': 0.5 * jnp.ones(4),
    'π0': 0.2 * jnp.ones(10),
}
```


### prior predictive check {#prior-predictive-check}

```python
vincp_prior_pred = get_prior_pred(vincp, vi_X)
vincp_prior_pred['y'].min(), vincp_prior_pred['y'].max()
```

|             |                 |             |                       |
|-------------|-----------------|-------------|-----------------------|
| DeviceArray | (0 dtype=int64) | DeviceArray | (2128827 dtype=int64) |

```python
clipped_kde(vincp_prior_pred['y']);
```

{{< figure src="/ox-hugo/008d2873ca4af66543797be031c9313b13d88377.png" >}}


### Fit the model to fake data {#fit-the-model-to-fake-data}

```python
recover_params(vincp, vincp_params, vi_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      1.14      0.21      1.14      0.80      1.48   2345.17      1.00
         β     -0.93      0.09     -0.92     -1.09     -0.78   4707.84      1.00
      ζ[0]      0.46      0.19      0.46      0.16      0.76   2025.63      1.00
      ζ[1]      0.49      0.20      0.50      0.17      0.82   1911.34      1.00
      ζ[2]      0.56      0.15      0.56      0.31      0.81   2063.94      1.00
      ζ[3]      0.40      0.28      0.40     -0.06      0.84   2144.07      1.00
     π0[0]     -0.01      0.93     -0.00     -1.46      1.61   3469.59      1.00
     π0[1]      0.03      0.54      0.03     -0.91      0.89   2066.92      1.00
     π0[2]      0.10      0.82      0.12     -1.30      1.41   2738.48      1.00
     π0[3]      0.32      0.50      0.30     -0.45      1.18   2333.01      1.00
     π0[4]     -0.30      0.85     -0.29     -1.74      1.03   2625.25      1.00
     π0[5]      0.23      0.60      0.23     -0.74      1.21   2125.67      1.00
     π0[6]      0.23      0.77      0.25     -1.00      1.55   2226.08      1.00
     π0[7]      0.08      0.79      0.08     -1.27      1.31   2808.30      1.00
     π0[8]     -0.50      0.75     -0.49     -1.62      0.86   2049.12      1.00
     π0[9]     -0.09      0.72     -0.09     -1.29      1.06   2345.71      1.00
         σ      0.40      0.17      0.36      0.15      0.65   1245.83      1.00
         φ      1.02      0.07      1.02      0.90      1.14   4775.76      1.00

Number of divergences: 0
```

Now the model is taking significantly longer to fit.


### Fit model to the real data {#fit-model-to-the-real-data}

```python
vincp_mcmc = run_mcmc(vincp, vi_X, sp_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.51      0.49      2.51      1.68      3.28   2342.06      1.00
         β     -1.95      0.65     -1.95     -3.02     -0.86   2561.34      1.00
      ζ[0]     -0.10      0.46     -0.08     -0.89      0.60   2083.20      1.00
      ζ[1]      0.12      0.35      0.12     -0.44      0.67   2397.96      1.00
      ζ[2]      0.03      0.28      0.03     -0.41      0.47   2055.67      1.00
      ζ[3]      0.01      0.57      0.02     -0.86      1.00   2848.88      1.00
     π0[0]      0.48      0.81      0.47     -0.88      1.80   2979.43      1.00
     π0[1]     -0.01      0.74      0.01     -1.27      1.15   2823.34      1.00
     π0[2]      0.48      0.69      0.48     -0.62      1.64   2946.83      1.00
     π0[3]     -0.14      0.75     -0.13     -1.34      1.12   2841.09      1.00
     π0[4]      0.21      0.82      0.22     -1.10      1.62   2549.09      1.00
     π0[5]     -0.41      0.67     -0.39     -1.54      0.65   3306.15      1.00
     π0[6]     -0.36      0.81     -0.34     -1.80      0.85   3380.99      1.00
     π0[7]      0.34      0.89      0.33     -1.05      1.81   3119.66      1.00
     π0[8]     -0.12      0.78     -0.12     -1.42      1.13   2745.75      1.00
     π0[9]     -0.20      0.84     -0.21     -1.60      1.14   3116.92      1.00
         σ      0.51      0.19      0.48      0.21      0.79   1404.57      1.00
         φ      1.51      0.31      1.48      0.99      1.97   3492.37      1.00

Number of divergences: 0
```

Great, no more divergence!


### Posterior predictive check {#posterior-predictive-check}

```python
vincp_posterior_pred = get_posterior_pred(vincp, vi_X, vincp_mcmc, sites=['y'])
```


### Density overlay {#density-overlay}

```python
overlay_kde(sp_y['y'], vincp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/af77f9cf010f4e2f26bffa0c77f7904bea3ab02c.png" >}}

The prediction looks much better now, the real data is well in the range
of model predictions. We have captured the overall distribution of the
data quite well.


### Specific statistics {#specific-statistics}

```python
ppc_stat(prop_zero, sp_y['y'], vincp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/4be5423dba8611e8bbf85835cebce4918c61a4c8.png" >}}

```python
ppc_stat(prop_large, sp_y['y'], vincp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/8a79f9d36cd0ee97e511f9e11882574ecef6bda2.png" >}}

the upper tail seem to be a little overestimated, but not much.


### posterior prediction by groups {#posterior-prediction-by-groups}

<a id="code-snippet--vis-vincp-building"></a>
```python
plot_grouped_mean(pest_data['complaints'], vincp_posterior_pred['y'], pest_data['building_id'], 2, 5, (12,4))
```

{{< figure src="/ox-hugo/e06d1ba596acb0e941e92913cefbaf17bd9dce56.png" >}}

We weren't doing terribly bad with the building-specific means before, but
now they are all well-captured by our model. The model is also able to
do a decent job estimating within-building variability.


## Extending the hierarchy: Modeling varying slopes for each building {#extending-the-hierarchy-modeling-varying-slopes-for-each-building}

Perhaps if the building information affects the model intercept, it
would also affect the coefficient on `traps`. We can add this to our
model and observe the fit. Like the varying intercepts, the varying
slopes will also use a non-centered parameterisation,
and we are giving the corresponding variables the same prior.


### Model {#model}

The observational model

\begin{aligned}
\text{complaints}\_{n} & \sim \text{Gamma-Poisson}(c, r\_{n}) \\\\\\
c  &= \phi \\\\\\
r\_{n}  &= \frac{\phi}{\lambda\_n} \\\\\\
\log \lambda\_{n}  &= \alpha + \pi\_b + (\beta + \kappa\_b)  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\pi &= \texttt{building\_data}  \zeta + \sigma \pi\_0 \\\\\\
\kappa &= \texttt{building\_data}  \gamma + \tau \kappa\_0 \\\\\\
\end{aligned}

the priors

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\phi & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\zeta & \sim \textrm{Normal}(0, 1) \\\\\\
\gamma & \sim \textrm{Normal}(0, 1) \\\\\\
\pi\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\kappa\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\sigma & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\tau & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\end{aligned}

<a id="code-snippet--model-vs"></a>
```python
def vs(traps, log_sq_foot, b, building_data):
    """Multiple Gamma-Poisson Regression
    With group level Varying Interceps and Varying Slopes, Non-centered parameterisation.
    """

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    φ = sample('φ', LogNormal(0, 0.5))
    σ = sample('σ', LogNormal(0, 0.5))
    τ = sample('τ', LogNormal(0, 0.5))

    with plate('N_predictors', size=building_data.shape[1]):
        ζ = sample('ζ', Normal(0, 1))
        γ = sample('γ', Normal(0, 1))

    with plate('N_buildings', size=building_data.shape[0]):
        π0 = sample('π0', Normal(0, 1))
        κ0 = sample('κ0', Normal(0, 1))

    π = deterministic('π', building_data @ ζ + σ * π0)
    κ = deterministic('κ', building_data @ γ + τ * κ0)

    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + π[b] + (β + κ[b]) * traps + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare the data {#prepare-the-data}

As the model gets more and more complex, we are having more and more
model parameters. As such, with limited data, it will be more and more
difficult to get accurate inference. It's always good to think harder about the model
assumptions, use better priors, but sometimes we just need more data to make more reliable inference. Lucky for us, we've
gotten some new data that extends the number of time points for which we
have observations for each building. So from now on we'll use the
extended data set.

```python
pest_more_link = 'data/more_pest.csv'
```

```python
pest_more = pd.read_csv(pest_more_link, index_col=0)
pest_more['m'] = pest_more.m - 1
pest_more['b'] = pest_more.b - 1
pest_more.index = pest_more.index - 1
```

Like before, we extract the building level data and put them in a
separate data set.

```python
bd_more = (pest_more[['live_in_super', 'age_of_building', 'average_tenant_age', 'monthly_average_rent', 'b']]
           .drop_duplicates()
           .sort_values(by='b')
           .set_index('b', drop=True))
bd_more = bd_more - bd_more.mean()

print(bd_more)
```

```text
   live_in_super  age_of_building  average_tenant_age  monthly_average_rent
b
0           -0.3            -0.24            0.395347              0.159693
1           -0.3            -0.64           -0.035733             -0.123740
2           -0.3            -0.04           -0.206569             -0.658587
3           -0.3             0.16           -0.258725              0.331981
4            0.7            -0.04           -0.478690             -0.010646
5           -0.3            -0.04           -0.137937             -0.309452
6           -0.3             0.96           -0.878173              0.183439
7            0.7            -0.14            1.525874              0.176402
8           -0.3             1.06            0.853178              0.152982
9            0.7            -1.04           -0.778571              0.097928
```

Again collect the predictor variables into a dict.

```python
vs_X = {
    'traps': pest_more.traps.values / 10,
    'log_sq_foot': pest_more.log_sq_foot.values - pest_more.log_sq_foot.mean(),
    'b': pest_more.b.values,
    'building_data': bd_more.values
}
```

We can use the previous fake dataset, because the model inputs are the
same. And the fixed parameters

```python
vs_params = {
    'α': jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'φ': jnp.ones(1),
    'σ': jnp.ones(1),
    'τ': jnp.ones(1),
    'ζ': 0.5 * jnp.ones(4),
    'γ': 0.5 * jnp.ones(4),
    'π0': 0.2 * jnp.ones(10),
    'κ0': 0.2 * jnp.ones(10),
}
```

Finally, our outcome variable now has new data

```python
vs_y = {
    'y': pest_more.complaints.values
}
```


### prior predictive check {#prior-predictive-check}

```python
vs_prior_pred = get_prior_pred(vs, vi_X)
print(vs_prior_pred['y'].min(), vs_prior_pred['y'].max())
```

```text
0 31577606
```

seems reasonable.

```python
clipped_kde(vs_prior_pred['y']);
```

{{< figure src="/ox-hugo/60842e796d1984e446c3dd19be95c213347c78b4.png" >}}


### Fit the model to fake data {#fit-the-model-to-fake-data}

```python
recover_params(vs, vs_params, vi_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      1.16      0.25      1.16      0.76      1.55   2391.07      1.00
         β     -0.71      0.30     -0.70     -1.15     -0.21   2498.25      1.00
      γ[0]      0.40      0.25      0.40      0.00      0.82   2590.93      1.00
      γ[1]      0.58      0.27      0.59      0.15      1.02   2329.58      1.00
      γ[2]      0.41      0.21      0.41      0.07      0.75   2297.67      1.00
      γ[3]      0.66      0.39      0.67     -0.01      1.27   2756.58      1.00
      ζ[0]      0.49      0.21      0.50      0.16      0.83   2453.15      1.00
      ζ[1]      0.40      0.24      0.41      0.02      0.79   2512.04      1.00
      ζ[2]      0.62      0.18      0.62      0.32      0.91   2538.12      1.00
      ζ[3]      0.26      0.32      0.26     -0.23      0.81   2619.06      1.00
     κ0[0]     -0.18      0.95     -0.18     -1.70      1.39   4797.00      1.00
     κ0[1]     -0.03      0.63     -0.02     -1.07      1.01   4156.15      1.00
     κ0[2]      0.08      0.85      0.10     -1.38      1.42   3702.79      1.00
     κ0[3]      0.02      0.62      0.02     -0.95      1.05   3605.81      1.00
     κ0[4]     -0.38      0.85     -0.38     -1.82      0.94   3655.65      1.00
     κ0[5]      0.58      0.68      0.56     -0.46      1.76   3343.59      1.00
     κ0[6]      0.05      0.81      0.03     -1.32      1.37   3989.87      1.00
     κ0[7]     -0.17      0.85     -0.17     -1.49      1.26   4845.48      1.00
     κ0[8]     -0.31      0.80     -0.29     -1.65      0.97   4015.75      1.00
     κ0[9]      0.33      0.77      0.32     -0.95      1.55   3608.79      1.00
     π0[0]      0.05      0.93      0.04     -1.49      1.52   4800.03      1.00
     π0[1]      0.14      0.61      0.15     -0.85      1.14   3677.97      1.00
     π0[2]      0.17      0.83      0.17     -1.10      1.59   4414.17      1.00
     π0[3]      0.32      0.62      0.30     -0.63      1.34   4469.05      1.00
     π0[4]     -0.05      0.86     -0.04     -1.42      1.42   3677.41      1.00
     π0[5]     -0.24      0.64     -0.24     -1.29      0.77   2933.09      1.00
     π0[6]      0.14      0.78      0.13     -1.24      1.33   3555.17      1.00
     π0[7]      0.10      0.84      0.10     -1.26      1.49   3851.29      1.00
     π0[8]     -0.32      0.77     -0.33     -1.55      0.96   3868.74      1.00
     π0[9]     -0.28      0.77     -0.29     -1.56      0.95   3503.53      1.00
         σ      0.45      0.19      0.41      0.17      0.71   2235.45      1.00
         τ      0.51      0.20      0.48      0.22      0.82   2283.63      1.00
         φ      1.03      0.07      1.03      0.91      1.14   7815.43      1.00

Number of divergences: 0
```


### Fit model to the real data {#fit-model-to-the-real-data}

```python
vs_mcmc = run_mcmc(vs, vs_X, vs_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.74      0.38      2.74      2.13      3.37   1776.43      1.00
         β     -3.03      0.62     -3.02     -4.08     -2.05   2049.56      1.00
      γ[0]     -0.81      0.85     -0.85     -2.19      0.59   3445.56      1.00
      γ[1]      0.80      0.71      0.81     -0.33      1.95   2403.11      1.00
      γ[2]      1.14      0.72      1.16      0.06      2.38   2100.07      1.00
      γ[3]      0.86      0.92      0.86     -0.68      2.33   3735.20      1.00
      ζ[0]     -0.21      0.63     -0.23     -1.20      0.83   2790.94      1.00
      ζ[1]      0.40      0.52      0.38     -0.45      1.25   2602.01      1.00
      ζ[2]     -0.91      0.49     -0.93     -1.67     -0.07   2278.52      1.00
      ζ[3]     -0.41      0.75     -0.42     -1.60      0.85   3022.90      1.00
     κ0[0]      1.71      0.63      1.67      0.60      2.67   2928.89      1.00
     κ0[1]     -0.64      0.70     -0.65     -1.77      0.52   2964.97      1.00
     κ0[2]     -0.92      0.71     -0.94     -2.12      0.21   3675.59      1.00
     κ0[3]      0.08      0.63      0.09     -0.91      1.14   2542.32      1.00
     κ0[4]     -0.63      0.70     -0.66     -1.74      0.52   2487.32      1.00
     κ0[5]     -0.44      0.75     -0.44     -1.73      0.74   3192.48      1.00
     κ0[6]     -0.18      0.72     -0.18     -1.40      0.95   2829.99      1.00
     κ0[7]     -0.52      0.85     -0.51     -1.95      0.83   3858.64      1.00
     κ0[8]      0.96      0.71      0.97     -0.21      2.14   2438.67      1.00
     κ0[9]     -0.16      0.84     -0.16     -1.54      1.21   3287.98      1.00
     π0[0]     -0.77      0.83     -0.78     -2.05      0.66   3082.05      1.00
     π0[1]      0.78      0.72      0.79     -0.41      1.94   2524.85      1.00
     π0[2]      0.12      0.71      0.13     -1.06      1.25   2433.69      1.00
     π0[3]      0.30      0.74      0.29     -0.88      1.48   3206.49      1.00
     π0[4]      0.68      0.73      0.66     -0.50      1.91   2818.51      1.00
     π0[5]      0.25      0.64      0.26     -0.79      1.30   2437.01      1.00
     π0[6]      0.33      0.78      0.33     -0.94      1.57   3099.38      1.00
     π0[7]      0.08      0.90      0.11     -1.44      1.54   3727.74      1.00
     π0[8]     -0.52      0.82     -0.52     -1.91      0.79   3505.20      1.00
     π0[9]     -0.83      0.89     -0.85     -2.27      0.62   3371.43      1.00
         σ      0.85      0.35      0.80      0.31      1.38   1476.77      1.00
         τ      1.49      0.52      1.41      0.69      2.28   1452.03      1.00
         φ      1.60      0.19      1.59      1.27      1.89   4132.89      1.00

Number of divergences: 0
```


### Posterior predictive check {#posterior-predictive-check}

```python
vs_posterior_pred = get_posterior_pred(vs, vs_X, vs_mcmc, sites=['y'])
print(vs_posterior_pred['y'].min(), vs_posterior_pred['y'].max())
```

```text
0 614
```


### Density overlay {#density-overlay}

```python
overlay_kde(vs_y['y'], vs_posterior_pred['y'])
```

{{< figure src="/ox-hugo/7470f8dde036ac839146dfce34720a531218ccbf.png" >}}

At this stage, it's difficult to look at the overall distribution and
see if our model has performed better. We just use the density overlay
to check that that our model is working as expected.

Nevertheless, we can see that compared to earlier models, the range of outcome values is growing bigger.


### Specific statistics {#specific-statistics}

```python
ppc_stat(prop_zero, vs_y['y'], vs_posterior_pred['y'])
```

{{< figure src="/ox-hugo/7b15a3b0aeb9fc6f3a7abfcbdec7f6681f31110a.png" >}}

The model seems to be underestimating zeros a little bit, but the variance has reduced significantly.

```python
ppc_stat(prop_large, vs_y['y'], vs_posterior_pred['y'])
```

{{< figure src="/ox-hugo/0f86c9553b66a58616b16bd7f6cdb0174e80daee.png" >}}


### posterior prediction by groups {#posterior-prediction-by-groups}

<a id="code-snippet--vis-vs-building"></a>
```python
plot_grouped_mean(pest_more['complaints'], vs_posterior_pred['y'], pest_more['b'], 2, 5, (12,4))
```

{{< figure src="/ox-hugo/cb3ec246dfb5edc6b08d50bc2536cf48e5a1fd47.png" >}}

compared to the early model, now the prediction variance is further more
reduced. But since we are also using more data, it's not sure which part
has contributed more, the data or the model.

We haven't model the monthly effect yet, but let's see how the monthly
prediction looks like

<a id="code-snippet--vis-vs-month"></a>
```python
plot_grouped_mean(pest_more['complaints'], vs_posterior_pred['y'], pest_more['m'], 6, 6, (12,12))
```

{{< figure src="/ox-hugo/a5731718814ca44790cda4923601d7a8097a76a5.png" >}}

This doesn't look good, some of the predictions are quite far off. This
implies that by adding the time information, we might be able to further
improve our model.


## Adding new hierarchies: Modeling varying intercepts for each month {#adding-new-hierarchies-modeling-varying-intercepts-for-each-month}

After looking at the grouping among buildings, and observing that this
grouping information indeed helps to improve the model fit, we move on
to another grouping variable, the different months at which we observed
the data. Specifically, we'll add a **second variation to the intercept**
to the linear model

\\[
\log \lambda\_{n}  = \alpha + \pi\_b + \theta\_m + (\beta + \kappa\_b)  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\\]

Previously, when modeling the varying intercepts (and also the varying
slopes) for each building, we have assumed that all the varying
intercepts are drawn independently from the same prior distribution

\\[ \pi \sim \text{Normal}(\texttt{building\_data}  \zeta, \sigma)\\]

We made this independence assumption because we have no knowledge on how
the buildings might be related to each other; however, when grouping the
observations by when they are observed, we know immediately that there
are some relationships among these groups, because these groups follow a
specific order: the observations from February follows those from
January, and thos from March follow those from February, etc. As such,
we have some certain structure among the variables, and to make use of
this structure, we'll model them jointly with some **structured prior**.

In this section we'll model the varying monthly effects on the intercept
with an AR(1) model, in which the conditional mean of each variable
depends on the previous variable. Considering the months are
sequentially ordered, this seems like a natural choice:

\\[\theta\_m \sim \text{Normal}(\rho  \theta\_{m-1}, \upsilon)\\]

Using the error term representation, which represents the conditional
distribution as a rescaling and translation of the standard normal
distribution (the error term), and which is more commonly used and more
convenient for mathematical manipulation in time series models, the
AR(1) model can be represented as

\begin{aligned}
\theta\_m &= \rho  \theta\_{m-1} +\upsilon \epsilon\_m \\\\\\
\epsilon\_m &\sim \text{Normal}(0, 1) \\\\\\
\end{aligned}

In the AR(1) model we have two (hyper)parameters, the correlation
coefficient \\(\rho\\) and the conditional standard deviation \\(\upsilon\\).
And we also need to assign a prior for \\(m\_1\\) since there is no previous
variable to condition on. Generally speaking we can assign any prior to
these variables, but in practice with AR process, we often assume that
the process is stationary, and this assumption applies extra constraints
on the variables.

By stationarity, all variables in the AR process have the same marginal
distribution. For normally distributed variables this means

\begin{aligned}
\text{Var}(\theta\_m) &= \text{Var}(\theta\_{m-1}) = \dots = \text{Var}(\theta\_{1}) \\\\\\
\mathbb{E}(\theta\_m) &= \mathbb{E}(\theta\_{m-1}) = \dots = \mathbb{E}(\theta\_{1}) \\\\\\
\end{aligned}

so

\begin{aligned}
\text{Var}(\theta\_m) &= \text{Var}(\rho \theta\_{m-1} + \upsilon \epsilon\_m)  \\\\\\
&= \text{Var}(\rho \theta\_{m-1}) + \text{Var}(\upsilon \epsilon\_m) \\\\\\
&= \rho^2 \text{Var}( \theta\_{m})  + \upsilon^2 \\\\\\
\end{aligned}

Clearly we need \\(\rho \in (-1, 1)\\) for this to make sense. We then
obtain the marginal variance:

\\[
\text{Var}(\theta\_m) = \frac{\upsilon^2}{1 - \rho^2}
\\]

For the mean of \\(\theta\_m\\) things are a bit simpler:

\begin{aligned}
\mathbb{E}(\theta\_m) &= \mathbb{E}(\rho  \theta\_{m-1} + \upsilon \epsilon\_t) \\\\\\
&= \mathbb{E}(\rho  \theta\_{m-1}) + \mathbb{E}(\upsilon \epsilon\_t) \\\\\\
&= \mathbb{E}(\rho  \theta\_{m-1})  + \upsilon 0\\\\\\
&= \rho \mathbb{E}(\theta\_m)
\end{aligned}

which for \\(\rho \neq 1\\) yields \\(\mathbb{E}(\theta\_{m}) = 0\\).

We now have the marginal distribution for \\(\theta\_{m}\\), which, in our
case, we will use for \\(\theta\_1\\):

\\[
\theta\_1 \sim \text{Normal}\left(0, \frac{\upsilon}{\sqrt{1 - \rho^2}}\right) \\\\\\
\\]

We need \\(\rho \in (-1, 1)\\), but there are no obvious distributions with
support on \\((-1,1)\\), we can define a raw variable on \\([0,1]\\) and then
transform it to \\([-1,1]\\). Specifically,

\\[
\rho\_0 \in [0, 1] \\\\\\
\rho = 2 \times \rho\_0 - 1
\\]

Given the description of the process above, it seems like there could be
either positive or negative associations between the months, but when
observing more complaints this month, we probably would also expect more
complaints next month, so there should be a bit more weight placed on
positive \\(\rho\\)s: we'll put an informative prior that pushes the
parameter \\(\rho\\) towards 0.5.

Since the noise terms for all time periods are independent and normally
distributed, we'll also model the standard deviation \\(\upsilon\\)
separately. Apart from being positive, we don't have any information on
where it should be, so we'll put a general prior on it.


### Model {#model}

Now the complete model. Note that because Python uses zero based
indexing, the months begin at zero and ends at 11.

\begin{aligned}
\text{complaints}\_{n} & \sim \text{Gamma-Poisson}(c, r\_{n}) \\\\\\
c  &= \phi \\\\\\
r\_{n}  &= \frac{\phi}{\lambda\_n} \\\\\\
\log \lambda\_{n}  &= \alpha + \pi\_b + \theta\_m + (\beta + \kappa\_b)  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\pi &= \texttt{building\_data}  \zeta + \sigma \pi\_0 \\\\\\
\kappa &= \texttt{building\_data}  \gamma + \tau \kappa\_0 \\\\\\
\theta\_m &= \rho  \theta\_{m-1} + \upsilon \epsilon\_m,  m = 1, ..., 11 \\\\\\
\theta\_0 &\sim \left( \upsilon / \sqrt{1 - \rho^2} \right) \epsilon\_0 \\\\\\
\rho &= 2 \times \rho\_0 - 1 \\\\\\
\end{aligned}

The priors

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\phi & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\zeta & \sim \textrm{Normal}(0, 1) \\\\\\
\gamma & \sim \textrm{Normal}(0, 1) \\\\\\
\sigma & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\tau & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\pi\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\kappa\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\epsilon &\sim \text{Normal}(0, 1) \\\\\\
\rho\_0 &\sim \text{Beta}(3, 1)\\\\\\
\upsilon & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\end{aligned}

<a id="code-snippet--model-ar"></a>
```python
from numpyro.distributions import Beta, HalfNormal
from numpyro.contrib.control_flow import scan

def ar(traps, log_sq_foot, b, building_data, m, M):
    """Hierarchical Gamma-Poisson Regression
    Building level: Varying intercepts and Varying Slopes, Non-centered parameterisation.
    Monthly level: varying intercepts. AR(1), Non-centered parameterisation.
    """

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    φ = sample('φ', LogNormal(0, 0.5))
    σ = sample('σ', LogNormal(0, 0.5))
    τ = sample('τ', LogNormal(0, 0.5))

    with plate('N_predictors', size=building_data.shape[1]):
        ζ = sample('ζ', Normal(0, 1))
        γ = sample('γ', Normal(0, 1))

    with plate('N_buildings', size=building_data.shape[0]):
        π0 = sample('π0', Normal(0, 1))
        κ0 = sample('κ0', Normal(0, 1))

    π = deterministic('π', building_data @ ζ + σ * π0)
    κ = deterministic('κ', building_data @ γ + τ * κ0)

    # AR(1) process, non-centered
    ν = sample('ν', LogNormal(0, 0.5))
    ρ0 = sample('ρ0', Beta(10, 5))
    ρ = deterministic('ρ', 2 * ρ0 - 1)

    with plate('N_months', size=M):
        ε = sample('ε', Normal(0, 1))

    θ0 = ν * ε[0] / jnp.sqrt(1 - ρ**2)

    def f(p, e):
        c = ρ * p + ν * e
        return c, c

    _, θ_rest = scan(f, θ0, ε[1:], length=M-1)
    θ = deterministic('θ', jnp.squeeze(jnp.concatenate((θ0[None], θ_rest))))

    # observations
    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(
            α + π[b] + θ[m] + (β + κ[b]) * traps + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare data {#prepare-data}

Compared to the previous model, we now also have the month indicator. We
only have data for 36 months, but we can feed model to 42 months (6
months more) so we can also look at the future prediction of the time
series model.

```python
ar_X = {
    'traps': pest_more.traps.values / 10,
    'log_sq_foot': pest_more.log_sq_foot.values - pest_more.log_sq_foot.mean(),
    'b': pest_more.b.values,
    'm': pest_more.m.values,
    'building_data': bd_more.values,
    'M': 42
}
```

and update the fake data accordingly

```python
ar_fX = dict(
    traps=random.randint(rng, fobs_shape, 0, 15) / 10,
    log_sq_foot=random.normal(rng, fobs_shape) * 0.4,
    b=random.randint(rng, fobs_shape, 0, 10),
    m=random.randint(rng, fobs_shape, 0, 12),
    building_data=random.normal(rng, [10, 4]),
    M=42
)
```

and the fixed parameters. With the model getting more and more complex,
it's more and more difficult to have reasonable fake parameters in the
reasonable region.

```python
ar_params = {
    'α': 2 * jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'φ': jnp.ones(1),
    'σ': 0.5 * jnp.ones(1),
    'τ': 0.5 * jnp.ones(1),
    'ζ': 0.2 * random.normal(rng, [4]),
    'γ': 0.2 * random.normal(rng, [4]),
    'π0': 0.2 * random.normal(rng, [10]),
    'κ0': 0.2 * random.normal(rng, [10]),
    'ε': 0.5 * random.normal(rng, [42]),
    'ρ': 0.5 * jnp.ones(1),
    'ν': 0.5 * jnp.ones(1),
}
```

The outcome is the same as before.


### prior predictive check {#prior-predictive-check}

```python
ar_prior_pred = get_prior_pred(ar, ar_X)
print(ar_prior_pred['λ'].min(), ar_prior_pred['λ'].max())
print(ar_prior_pred['y'].min(), ar_prior_pred['y'].max())
```

```text
9.281219746258625e-10 2415289530.338127
0 14609871872
```

we can see that as the model grows more and more complex, so is the corresponding parameter space; and consequentially, the outcome range has also significantly grown.

```python
clipped_kde(ar_prior_pred['y']);
```

{{< figure src="/ox-hugo/644a817c420e45bd175facd191cb046e44dcba9a.png" >}}

```python
sns.histplot(ar_prior_pred['ν']);
```

{{< figure src="/ox-hugo/69e6b6122287744f92c27e3e6e423418d46aef08.png" >}}

we can also look at the prior for the AR(1) model

```python
fig, ax = plt.subplots(figsize=(16, 8))
for i in random.randint(rng, [10], 0, 1000):
    ax.plot(ar_prior_pred['θ'][i])
```

{{< figure src="/ox-hugo/990251947919be6c89d54705d5c5a065e43d4541.png" >}}

we can see that a priori \\(\theta\\) is centered at zero and their is no
strong correlation between them. Also the plots are quite wiggly.


### Fit the model to fake data {#fit-the-model-to-fake-data}

```python
recover_params(ar, ar_params, ar_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.00      0.36      2.01      1.45      2.57   1311.33      1.00
         β     -0.88      0.31     -0.87     -1.36     -0.36   1964.77      1.00
      γ[0]     -0.03      0.28     -0.02     -0.48      0.44   2434.52      1.00
      γ[1]     -0.01      0.32     -0.01     -0.56      0.45   2015.00      1.00
      γ[2]     -0.18      0.24     -0.18     -0.60      0.20   2071.82      1.00
      γ[3]     -0.13      0.42     -0.13     -0.80      0.56   2213.09      1.00
      ε[0]     -0.29      0.53     -0.27     -1.16      0.58   1762.74      1.00
      ε[1]     -1.06      0.52     -1.04     -1.95     -0.27   1529.23      1.00
      ε[2]      0.48      0.52      0.47     -0.33      1.35   2148.85      1.00
      ε[3]     -0.03      0.45     -0.04     -0.77      0.68   1654.14      1.00
      ε[4]      0.81      0.52      0.80     -0.10      1.59   1981.06      1.00
      ε[5]      0.57      0.52      0.56     -0.25      1.43   1635.77      1.00
      ε[6]      0.05      0.51      0.05     -0.76      0.88   2036.65      1.00
      ε[7]     -0.92      0.50     -0.91     -1.72     -0.10   1476.45      1.00
      ε[8]      0.63      0.51      0.63     -0.21      1.47   2250.66      1.00
      ε[9]      1.17      0.53      1.14      0.31      2.03   1444.02      1.00
     ε[10]     -0.99      0.57     -0.98     -1.95     -0.10   2239.23      1.00
     ε[11]     -0.02      0.49     -0.03     -0.87      0.73   1717.68      1.00
     ε[12]     -0.00      0.97     -0.00     -1.69      1.53   6636.68      1.00
     ε[13]      0.04      0.99      0.04     -1.63      1.64   6297.39      1.00
     ε[14]      0.02      0.98      0.04     -1.58      1.58   6084.91      1.00
     ε[15]     -0.01      1.01     -0.02     -1.61      1.70   5016.35      1.00
     ε[16]      0.02      0.99      0.01     -1.60      1.62   5098.14      1.00
     ε[17]      0.02      0.98      0.00     -1.61      1.58   6915.53      1.00
     ε[18]     -0.01      0.99     -0.01     -1.50      1.73   5828.44      1.00
     ε[19]      0.00      1.00      0.00     -1.62      1.62   6039.67      1.00
     ε[20]      0.00      0.99      0.01     -1.69      1.62   5757.75      1.00
     ε[21]      0.00      1.00      0.00     -1.71      1.58   5689.53      1.00
     ε[22]      0.00      0.99      0.00     -1.62      1.65   5833.59      1.00
     ε[23]      0.00      0.98      0.01     -1.62      1.53   5455.88      1.00
     ε[24]     -0.00      0.99     -0.01     -1.66      1.55   6042.50      1.00
     ε[25]      0.01      0.99      0.01     -1.53      1.70   5920.46      1.00
     ε[26]      0.00      1.00      0.01     -1.57      1.73   5748.49      1.00
     ε[27]     -0.00      1.03     -0.00     -1.61      1.74   6595.91      1.00
     ε[28]     -0.00      1.00     -0.00     -1.62      1.65   5990.48      1.00
     ε[29]     -0.01      1.00     -0.00     -1.58      1.66   6240.64      1.00
     ε[30]      0.01      0.99      0.02     -1.58      1.64   5595.67      1.00
     ε[31]      0.01      0.99      0.01     -1.68      1.61   6618.35      1.00
     ε[32]      0.01      0.98      0.01     -1.67      1.54   6337.46      1.00
     ε[33]     -0.01      0.98     -0.01     -1.60      1.56   5148.25      1.00
     ε[34]      0.01      1.01      0.01     -1.63      1.65   5675.38      1.00
     ε[35]     -0.01      0.98      0.01     -1.56      1.63   5968.75      1.00
     ε[36]      0.01      1.00      0.02     -1.55      1.74   5445.33      1.00
     ε[37]     -0.01      1.01     -0.01     -1.65      1.63   5501.50      1.00
     ε[38]      0.01      0.98     -0.01     -1.61      1.61   5284.04      1.00
     ε[39]      0.01      1.02     -0.00     -1.70      1.68   5991.65      1.00
     ε[40]     -0.01      0.98     -0.02     -1.55      1.60   5958.40      1.00
     ε[41]      0.00      1.02      0.02     -1.70      1.64   5830.59      1.00
      ζ[0]      0.06      0.23      0.07     -0.34      0.40   1758.90      1.00
      ζ[1]      0.05      0.26      0.05     -0.35      0.49   1767.25      1.00
      ζ[2]     -0.08      0.20     -0.09     -0.41      0.25   2064.29      1.00
      ζ[3]     -0.33      0.36     -0.33     -0.94      0.23   1923.60      1.00
     κ0[0]      0.23      0.94      0.22     -1.23      1.82   3638.51      1.00
     κ0[1]     -0.83      0.65     -0.82     -1.85      0.25   2981.98      1.00
     κ0[2]      0.08      0.87      0.08     -1.42      1.48   3643.42      1.00
     κ0[3]     -0.54      0.59     -0.52     -1.44      0.45   2580.64      1.00
     κ0[4]     -0.19      0.85     -0.18     -1.65      1.13   2961.68      1.00
     κ0[5]      0.85      0.65      0.83     -0.25      1.88   3054.49      1.00
     κ0[6]     -0.10      0.78     -0.09     -1.29      1.25   2986.16      1.00
     κ0[7]      0.37      0.83      0.36     -0.92      1.81   2951.16      1.00
     κ0[8]     -0.12      0.73     -0.12     -1.32      1.06   2531.05      1.00
     κ0[9]      0.18      0.73      0.16     -0.98      1.35   2886.71      1.00
         ν      0.51      0.14      0.50      0.30      0.71   1643.07      1.00
     π0[0]     -0.05      0.93     -0.05     -1.52      1.50   3892.69      1.00
     π0[1]     -0.07      0.62     -0.05     -1.16      0.90   2709.52      1.00
     π0[2]      0.29      0.90      0.29     -1.25      1.69   3078.28      1.00
     π0[3]      0.09      0.62      0.07     -0.88      1.13   2810.47      1.00
     π0[4]     -0.10      0.90     -0.10     -1.58      1.34   2755.45      1.00
     π0[5]     -0.57      0.67     -0.55     -1.76      0.45   2150.59      1.00
     π0[6]      0.35      0.78      0.33     -0.93      1.61   2336.91      1.00
     π0[7]     -0.08      0.86     -0.08     -1.43      1.37   3342.55      1.00
     π0[8]      0.10      0.78      0.08     -1.14      1.39   2544.39      1.00
     π0[9]      0.10      0.77      0.11     -1.20      1.31   3060.70      1.00
        ρ0      0.64      0.11      0.65      0.47      0.83   2510.77      1.00
         σ      0.48      0.20      0.44      0.18      0.74   1818.93      1.00
         τ      0.61      0.22      0.58      0.27      0.92   2112.60      1.00
         φ      0.99      0.05      0.98      0.90      1.07   5199.47      1.00

Number of divergences: 0
```


### Fit model to the real data {#fit-model-to-the-real-data}

```python
ar_mcmc = run_mcmc(ar, ar_X, vs_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         α      2.02      0.54      2.05      1.21      2.98   1818.75      1.00
         β     -2.22      0.40     -2.22     -2.83     -1.55   4038.21      1.00
      γ[0]     -1.05      0.67     -1.07     -2.09      0.12   4927.54      1.00
      γ[1]      1.09      0.51      1.09      0.24      1.91   3974.61      1.00
      γ[2]     -0.05      0.53     -0.05     -0.93      0.78   3692.44      1.00
      γ[3]      0.44      0.80      0.45     -0.88      1.73   4794.99      1.00
      ε[0]     -2.18      0.71     -2.20     -3.36     -1.03   2640.51      1.00
      ε[1]      0.31      0.74      0.30     -0.87      1.52   4071.47      1.00
      ε[2]     -0.80      0.68     -0.80     -2.05      0.20   4539.50      1.00
      ε[3]     -0.08      0.66     -0.08     -1.16      0.99   4421.24      1.00
      ε[4]     -0.84      0.68     -0.83     -1.99      0.24   4792.29      1.00
      ε[5]     -0.16      0.65     -0.16     -1.31      0.83   4596.87      1.00
      ε[6]     -0.66      0.67     -0.67     -1.81      0.38   4457.60      1.00
      ε[7]      0.61      0.61      0.60     -0.42      1.58   4178.00      1.00
      ε[8]      0.76      0.50      0.75     -0.09      1.56   4346.88      1.00
      ε[9]     -0.91      0.50     -0.91     -1.69     -0.06   4986.65      1.00
     ε[10]     -0.51      0.56     -0.52     -1.38      0.46   5146.55      1.00
     ε[11]      1.57      0.51      1.55      0.72      2.38   3156.49      1.00
     ε[12]      0.67      0.43      0.65      0.00      1.41   3791.60      1.00
     ε[13]      1.08      0.41      1.07      0.41      1.77   3674.55      1.00
     ε[14]     -1.32      0.46     -1.31     -2.07     -0.58   3268.35      1.00
     ε[15]      0.51      0.44      0.51     -0.18      1.25   4540.85      1.00
     ε[16]      0.29      0.42      0.29     -0.39      0.99   4949.63      1.00
     ε[17]     -0.08      0.43     -0.08     -0.78      0.59   5090.65      1.00
     ε[18]      0.97      0.43      0.95      0.30      1.69   3597.31      1.00
     ε[19]     -0.24      0.42     -0.24     -0.96      0.41   4256.40      1.00
     ε[20]      0.94      0.41      0.93      0.24      1.58   4453.61      1.00
     ε[21]      1.07      0.39      1.06      0.43      1.70   3536.15      1.00
     ε[22]      0.16      0.39      0.17     -0.47      0.80   3660.65      1.00
     ε[23]     -0.01      0.40     -0.01     -0.64      0.65   4098.68      1.00
     ε[24]      0.26      0.40      0.25     -0.42      0.90   4497.73      1.00
     ε[25]      0.53      0.40      0.53     -0.08      1.21   4497.78      1.00
     ε[26]      0.48      0.40      0.47     -0.18      1.11   3900.19      1.00
     ε[27]      0.48      0.40      0.48     -0.18      1.13   4033.09      1.00
     ε[28]     -0.88      0.45     -0.86     -1.61     -0.13   3622.97      1.00
     ε[29]      0.13      0.43      0.14     -0.62      0.80   4719.16      1.00
     ε[30]      1.22      0.43      1.21      0.56      1.94   3515.74      1.00
     ε[31]      0.78      0.40      0.76      0.12      1.43   3709.84      1.00
     ε[32]      1.14      0.39      1.14      0.47      1.75   3543.07      1.00
     ε[33]     -1.06      0.46     -1.04     -1.78     -0.29   2738.35      1.00
     ε[34]     -0.11      0.41     -0.10     -0.83      0.53   4466.52      1.00
     ε[35]      0.81      0.41      0.81      0.15      1.49   4142.27      1.00
     ε[36]      0.01      1.00      0.01     -1.61      1.64   7108.09      1.00
     ε[37]     -0.00      0.97     -0.01     -1.64      1.56   6040.23      1.00
     ε[38]     -0.00      1.00     -0.01     -1.69      1.61   5403.71      1.00
     ε[39]      0.00      1.01      0.01     -1.57      1.77   6051.97      1.00
     ε[40]     -0.00      0.98     -0.00     -1.63      1.59   7230.53      1.00
     ε[41]     -0.00      1.00     -0.00     -1.66      1.57   6178.58      1.00
      ζ[0]     -0.36      0.54     -0.36     -1.24      0.54   4018.46      1.00
      ζ[1]      0.39      0.43      0.39     -0.32      1.08   3797.31      1.00
      ζ[2]     -0.19      0.39     -0.20     -0.85      0.42   3651.05      1.00
      ζ[3]      0.05      0.63      0.05     -0.97      1.08   3935.45      1.00
     κ0[0]      0.94      0.74      0.94     -0.32      2.11   4154.48      1.00
     κ0[1]     -0.07      0.80     -0.09     -1.33      1.26   4597.71      1.00
     κ0[2]     -0.24      0.83     -0.25     -1.67      1.04   4508.86      1.00
     κ0[3]     -0.20      0.75     -0.21     -1.52      0.95   4611.82      1.00
     κ0[4]      0.14      0.80      0.13     -1.17      1.41   5144.78      1.00
     κ0[5]     -0.31      0.85     -0.30     -1.63      1.13   6040.85      1.00
     κ0[6]      0.14      0.83      0.14     -1.23      1.45   4361.30      1.00
     κ0[7]     -0.65      0.92     -0.65     -2.11      0.91   5227.79      1.00
     κ0[8]      0.40      0.83      0.42     -0.97      1.72   3990.94      1.00
     κ0[9]     -0.39      0.88     -0.40     -1.89      1.00   5148.84      1.00
         ν      0.59      0.10      0.59      0.44      0.75   1432.49      1.00
     π0[0]      0.79      0.81      0.79     -0.62      2.02   5109.54      1.00
     π0[1]     -0.16      0.73     -0.16     -1.39      0.99   3352.20      1.00
     π0[2]     -0.11      0.75     -0.09     -1.39      1.05   4328.65      1.00
     π0[3]     -0.50      0.76     -0.50     -1.83      0.68   4641.05      1.00
     π0[4]      0.20      0.84      0.21     -1.10      1.64   4389.02      1.00
     π0[5]      0.00      0.62      0.00     -1.01      1.00   4130.04      1.00
     π0[6]      0.33      0.83      0.32     -1.05      1.65   4828.66      1.00
     π0[7]     -0.20      0.92     -0.20     -1.69      1.31   5188.84      1.00
     π0[8]     -0.10      0.84     -0.10     -1.51      1.25   4058.60      1.00
     π0[9]     -0.16      0.93     -0.16     -1.70      1.36   6177.02      1.00
        ρ0      0.89      0.04      0.89      0.83      0.96   1813.80      1.00
         σ      0.57      0.22      0.53      0.25      0.91   2849.52      1.00
         τ      0.76      0.29      0.71      0.27      1.17   3041.78      1.00
         φ      6.66      1.23      6.53      4.65      8.55   4023.89      1.00

Number of divergences: 0
```

The inference result is quite different from earlier models


### Posterior predictive check {#posterior-predictive-check}

```python
ar_posterior_pred = get_posterior_pred(ar, ar_X, ar_mcmc)
```


### Density overlay {#density-overlay}

```python
overlay_kde(vs_y['y'], ar_posterior_pred['y'])
```

{{< figure src="/ox-hugo/9be8030d2216a6079ea911e0683e489909b6aced.png" >}}


### Specific statistics {#specific-statistics}

```python
ppc_stat(prop_zero, vs_y['y'], ar_posterior_pred['y'])
```

{{< figure src="/ox-hugo/30a3fbf8ad560690cf8b1469d5247523449e563d.png" >}}

```python
ppc_stat(prop_large, vs_y['y'], ar_posterior_pred['y'])
```

{{< figure src="/ox-hugo/7602788fc559324da6445b1de21ab5d3d0dc50ff.png" >}}


### AR(1) model {#ar--1--model}

<a id="code-snippet--vis-ar-ar"></a>
```python
fig, ax = plt.subplots(figsize=(16, 8))
plt.axvline(35, color=colors[1], alpha=0.5, lw=4)
for i in random.randint(rng, [200], 0, 1000):
    ax.plot(ar_posterior_pred['θ'][i], color=colors[0], alpha=0.2)
```

{{< figure src="/ox-hugo/7419dbfe4029ccad9b38cf0191ff1c3c02f2bec9.png" >}}

Now we can clearly see the monthly trend, although there are many abrupt
changes in the trend. We can also see that the periods where we don't have data, after 36th month, the prediction has much bigger uncertainty. This isn't very realistic, we'd imagine that if
something changes over time, it should more realistically change
gradually. This is why we might consider modeling the time series with a
Gaussian process.


### posterior prediction by groups {#posterior-prediction-by-groups}

<a id="code-snippet--vis-ar-building"></a>
```python
plot_grouped_mean(pest_more['complaints'], ar_posterior_pred['y'], pest_more['b'], 2, 5, (12,4))
```

{{< figure src="/ox-hugo/7be4e2b257c571ae26c7f3dd2b824199aa6b5648.png" >}}

We have been able to further reduce the prediction variance.

<a id="code-snippet--vis-ar-month"></a>
```python
plot_grouped_mean(pest_more['complaints'], ar_posterior_pred['y'], pest_more['m'], 6, 6, (12,12))
```

{{< figure src="/ox-hugo/b64e17c3222f74cf9287bd1347fd41d4dbc94f7c.png" >}}

and the monthly predictions are also quite good. Except for the first
month, because with an AR(1) model, there is no previous period to
condition on, the first period is not really modeled in the AR(1)
process, only as an independent normal distribution.

Now our model is also doing quite well with the prediction among
different groups.


## Going nonparametric: Modeling monthly effects with Gaussian process {#going-nonparametric-modeling-monthly-effects-with-gaussian-process}

We have been using the AR(1) process to model the relationship between
the varying monthly effect. The AR(1) process, like other Markov models,
using a number of parameters to directly model the local relationship between
neighbouring variables. But we can also use a non-parametric method,
the Gaussian process, to jointly model the relationship between all the
variables. In fact, as shown below, the AR(1) process can be seen to be
implicitly defining a specific Gaussian Process. Using a Gaussian
Process, we can model the structured prior with more flexibility.


### Joint density for AR(1) process {#joint-density-for-ar--1--process}

With Gaussian process, any finite number of variables are modeled as a
multivariate normal distribution. To see its relation with the AR(1)
process, we first derive the joint distribution of the variables defined
by the AR(1) process:

From before we know that the marginal distributions for the monthly
effects are

\begin{aligned}
\theta\_1 & \sim \text{Normal}\left(0, \frac{\upsilon}{\sqrt{1 - \rho^2}}\right) \\\\\\
\theta\_m & \sim \text{Normal}\left(\rho  \theta\_{m-1}, \upsilon\right) \forall t > 1
\end{aligned}

Rewriting our process in terms of the errors will make the derivation of
the joint distribution clearer

\begin{aligned}
\theta\_m & = \rho  \theta\_{m-1} + \upsilon\epsilon\_t  \\\\\\
\theta\_1 & \sim \frac{\upsilon}{\sqrt{1 - \rho^2}} \epsilon\_1 \\\\\\
\epsilon\_t & \sim \text{Normal}\left(0, 1\right) \; t = 1, 2, \dots,
\end{aligned}

We can see that our first term \\(\theta\_1\\) is normally distributed, and
the subsequent terms are sums of normal random variables, and thus are
also normally distributed; we can further show that jointly the vector,
\\(\theta\\), the \\(t\\)-th element being the scalar variable \\(\theta\_m\\), is
multivariate normal distributed:

Take \\(t=3\\) as an example. The first element is fairly
straightforward, because it mirrors our earlier parameterization of the
AR(1) prior. The only difference is that we're explicitly adding the
last two terms of \\(\epsilon\\) into the equation so we can use matrix
algebra for our transformation.

\\[ \theta\_1 = \frac{\nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + 0 \times \epsilon\_2 + 0 \times \epsilon\_3\\ \\]

The second element:

\begin{aligned}
\theta\_2 & = \rho \theta\_1 + \nu\epsilon\_2 + 0 \times \epsilon\_3  \\\\\\
 & = \rho \left(\frac{\nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1\right) + \nu\epsilon\_2 + 0 \times \epsilon\_3  \\\\\\
 & = \frac{\rho \nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + \nu\epsilon\_2 + 0 \times \epsilon\_3  \\\\\\
\end{aligned}

While the third element

\begin{aligned}
\theta\_3 & = \rho  \theta\_2 + \nu\epsilon\_3 \\\\\\
 & = \rho \left(\frac{\rho \nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + \nu\epsilon\_2\right) + \nu \epsilon\_3  \\\\\\
& = \frac{\rho^2 \nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + \rho  \nu\epsilon\_2 +  \nu\epsilon\_3  \\\\\\
\end{aligned}

Writing this all together:

\begin{aligned}
\theta\_1 & = \frac{\nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + 0 \times \epsilon\_2 + 0 \times \epsilon\_3\\\\\\
\theta\_2 & = \frac{\rho \nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + \nu\epsilon\_2 + 0 \times \epsilon\_3  \\\\\\
\theta\_3 & = \frac{\rho^2 \nu}{\sqrt{1 - \rho^2}} \times \epsilon\_1 + \rho  \nu\epsilon\_2 +  \nu\epsilon\_3  \\\\\\
\end{aligned}

Now we have been able to represent all the variables, instead of using
the variable from the **previous one period**, but error terms from **all
the previous periods**. A model that model the current variable with
previous error terms is known as a MA (moving average) model, and from
the previous manipulation we can see that an AR(1) model is equivalent
to a MA(∞) model.

In matrix form:

\\[
\theta = \begin{bmatrix} \upsilon / \sqrt{1 - \rho^2} & 0 & 0 \\\\\\
\rho \upsilon / \sqrt{1 - \rho^2} & \upsilon & 0 \\\\\\
\rho^2 \upsilon / \sqrt{1 - \rho^2} & \rho \upsilon  & \upsilon
\end{bmatrix} \times \epsilon
\\]

Because the elements in \\(\epsilon\\) are all independent and standard
normal distributed, we know that \\(\epsilon\\) is multivariate-normal
distributed with independent components,
\\(\epsilon \sim \text{MultiNormal}(0, I\_M)\\), \\(I\_M\\) being the identity
matrix. And let

\\[
L = \begin{bmatrix} \upsilon / \sqrt{1 - \rho^2} & 0 & 0 \\\\\\
\rho \upsilon / \sqrt{1 - \rho^2} & \upsilon & 0 \\\\\\
\rho^2 \upsilon / \sqrt{1 - \rho^2} & \rho \upsilon  & \upsilon
\end{bmatrix}
\\]

\\(\theta = L\times\epsilon\\) is also multivariate normal distributed with
\\(\theta \sim \text{MultiNormal}(0, L L^T)\\). So instead of defining the
conditional distributions of each variable separately, we can also
define them like this, as a multivariate normal distribution.

The covariance

\begin{aligned}
L L^T &=
\begin{bmatrix} \upsilon / \sqrt{1 - \rho^2} & 0 & 0 \\\\\\
\rho \upsilon / \sqrt{1 - \rho^2} & \upsilon & 0 \\\\\\
\rho^2 \upsilon / \sqrt{1 - \rho^2} & \rho \upsilon  & \upsilon
\end{bmatrix} \times
\begin{bmatrix} \upsilon / \sqrt{1 - \rho^2} & \rho \upsilon / \sqrt{1 - \rho^2} & \rho^2 \upsilon / \sqrt{1 - \rho^2} \\\\\\
0 & \upsilon & \rho \upsilon \\\\\\
0 & 0  & \upsilon
\end{bmatrix} \\\\\\
&=
 \begin{bmatrix} \nu / (1 - \rho^2) & \rho  \nu / (1 - \rho^2) &  \rho^2  \nu / (1 - \rho^2)\\\\\\
\rho  \nu / (1 - \rho^2)  & \nu / (1 - \rho^2)  & \rho  \nu / (1 - \rho^2) \\\\\\
\rho^2 \nu / (1 - \rho^2) & \rho  \nu / (1 - \rho^2) & \nu / (1 - \rho^2)
\end{bmatrix} \\\\\\
&=
\left( \frac{\nu}{1 - \rho^2} \right)
 \begin{bmatrix} 1 & \rho  &  \rho^2 \\\\\\
\rho  & 1  & \rho  \\\\\\
\rho^2 & \rho & 1
\end{bmatrix}
\end{aligned}

This of course generalises to any number of higher dimensions. As before
we can see that all the variables have the same marginal distribution,
and the correlation between neighbouring variables are \\(\rho\\), and
decreases as the distance gets further.


### Cholesky decomposition {#cholesky-decomposition}

Now that we have \\(\theta \sim \text{MultiNormal}(0, L L^T)\\), we can
directly sample from it. But as it turned out, it's more efficient to
sample first \\(\epsilon\\), and then transform \\(\theta = L\times\epsilon\\).
In the above AR(1) process we have already explicitly constructed \\(L\\) so
this is easy, but in a general case when a covariance matrix is given,
we need to decompose \\(\Sigma\\) into \\(L L^T\\) first, this process is called
the **Cholesky decomposition** .


### Extension to Gaussian processes {#extension-to-gaussian-processes}

The prior we defined for \\(\theta\\) is strictly speaking a Gaussian process.
Gaussian process is a stochastic process that is distributed as jointly
multivariate normal for any finite value of \\(M\\) and as we have seen, the
AR(1) process is exactly that. In Gaussian process notation, we could
write the above prior for \\(\theta\\) as:

\\[ \theta\_m \sim \text{GP}\left( 0, K(t | \upsilon,\rho) \right) \\]

In which \\(K(t | \upsilon,\rho)\\) is called a kernel function and it
defines the covariance matrix of the process over the domain \\(t\\),

\\[ \text{Cov}(\theta\_m,\theta\_{m+h}) = k(m, m+h | \upsilon, \rho). \\]

And now we have a Gaussian process, with its covariance matrix
parameterised by two hyperparameters, \\(\upsilon\\) and \\(\rho\\). What if we
want to use a different covariance? It turns out there are many
elementwise kernel functions

\\[
K\_{[m,m+h]} = k(m, m+h | \theta)
\\]

where \\(\theta\\) are the hyperparameters, that can be applied to construct
covariance matrices, a very popular one, and one we are going to use
here, is called the **exponentiated quadratic function**:

\begin{aligned}
k(m, m+h | \theta) & = \chi^2  \exp \left( - \dfrac{1}{2\ell^2} ((m+h) - m)^2 \right) \\\\\\
& = \chi^2  \exp \left( - \dfrac{h^2}{2\ell^2} \right).
\end{aligned}

The exponentiated quadratic kernel has two hyperparameters, \\(\chi\\), the
marginal standard deviation of the stochastic process, which determines
the scale of the outputand and length-scale \\(\ell\\), which determines the
smoothness of the process. The larger \\(\chi\\), the larger the output of
the process; the larger \\(\ell\\), the smoother the process.

To implement the Gaussian process in practice we'll make use of the
Cholsky decomposition we have introduced above. We first draw

\\[\epsilon \sim \text{MultiNormal}(0, I\_M)\\]

and because the elements are independent we can simply draw multiple
samples from one dimensional standard normal; then we can construct the
covariance matrix \\(\Sigma\\) using the squared exponential kernel,
decomposed it to get \\(L\\), finally we transform it to get

\\[\theta = L\times\epsilon \sim \text{MultiNormal}(0, \Sigma)\\]

As we can see here, although Gaussian process is a non-parametric
process defined on infinite dimensions but in practice, we are always
dealing with finite dimensions and all we need to do is some linear
algebra.


### Model {#model}

Now we can write out the complete model using GP for the time series
effects.

\begin{aligned}
\text{complaints}\_{n} & \sim \text{Gamma-Poisson}(c, r\_{n}) \\\\\\
c  &= \phi \\\\\\
r\_{n}  &= \frac{\phi}{\lambda\_n} \\\\\\
\log \lambda\_{n}  &= \alpha + \pi\_b + \theta\_m + (\beta + \kappa\_b)  {\rm traps}\_{n} + \text{log\_sq\_foot}\_b\\\\\\
\pi &= \texttt{building\_data}  \zeta + \sigma \pi\_0 \\\\\\
\kappa &= \texttt{building\_data}  \gamma + \tau \kappa\_0 \\\\\\
\theta &= L \times \epsilon \\\\\\
L &= \text{Cholsky}(\Sigma) \\\\\\
\Sigma &= \text{K}({\rm m}, \chi, \ell) \\\\\\
\end{aligned}

\\(\text{K}\\) is the exponentiated square kernel function

<a id="code-snippet--kernel"></a>
```python
# exponentiated square kernel with diagonal noise term
def K(X, Z, χ, ell, jitter=1.0e-8):
    deltaXsq = jnp.power((X[:, None] - Z) / ell, 2.0)
    k = jnp.power(χ, 2.0) * jnp.exp(-0.5 * deltaXsq)
    k += jitter * jnp.eye(X.shape[0])
    return k
```

We are still faced with the problem of choosing priors for \\(\chi\\) and
\\(\ell\\). The prior for \\(\chi\\) should not be too difficult, because it
controls the output scale, and since we have been modeling the intercept
quite extensively, we have a good idea of what scale it should take, so
we'll put a weakly informative prior on it. As for \\(\ell\\), the sensible
prior well depend upon the distance between the neighbouring points, and
more specifically, how do we determine the numerical distance between
the months? we can simply use the indicators from 0 to 35, but this is a
relatively large range, and besides, if the minimal distance in our data
is 1, there is no way our model can learn any length scale that's less
than one because we don't have information over that limit.

Now the prior model

\begin{aligned}
\alpha & \sim \textrm{Normal}(1, 2) \\\\\\
\beta & \sim \textrm{Normal}(-1, 2) \\\\\\
\phi & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\zeta & \sim \textrm{Normal}(0, 1) \\\\\\
\gamma & \sim \textrm{Normal}(0, 1) \\\\\\
\sigma & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\tau & \sim \textrm{LogNormal}(0, 0.5) \\\\\\
\pi\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\kappa\_0 & \sim \textrm{Normal}(0, 1) \\\\\\
\epsilon &\sim \text{MultiNormal}(0, I\_M) \\\\\\
\chi &\sim \textrm{LogNormal}(0, 0.5)\\\\\\
\ell & \sim \textrm{Gamma}(10, 2) \\\\\\
\end{aligned}

<a id="code-snippet--model-gp"></a>
```python
from numpyro.distributions import Gamma
from jax.scipy.linalg import cholesky

def gp(traps, log_sq_foot, b, building_data, m, M):
    """Hierarchical Gamma-Poisson Regression
    Building level: Varying intercepts and Varying Slopes, Non-centered parameterisation.
    Monthly level: varying intercepts. Gaussian process, Non-centered parameterisation.
    """

    α = sample('α', Normal(1, 2))
    β = sample('β', Normal(-1, 2))
    φ = sample('φ', LogNormal(0, 0.5))
    σ = sample('σ', LogNormal(0, 0.5))
    τ = sample('τ', LogNormal(0, 0.5))

    with plate('N_predictors', size=building_data.shape[1]):
        ζ = sample('ζ', Normal(0, 1))
        γ = sample('γ', Normal(0, 1))

    with plate('N_buildings', size=building_data.shape[0]):
        π0 = sample('π0', Normal(0, 1))
        κ0 = sample('κ0', Normal(0, 1))

    π = deterministic('π', building_data @ ζ + σ * π0)
    κ = deterministic('κ', building_data @ γ + τ * κ0)

    # GP, non-centered
    χ = sample('χ', LogNormal(0, 0.5))
    ell = sample('ell', Gamma(10, 2))

    with plate('N_months', size=M):
        ε = sample('ε', Normal(0, 1))

    ms = jnp.arange(0, M)
    Σ = K(ms, ms, χ, ell)
    L = cholesky(Σ, lower=True)

    θ = deterministic('θ', jnp.matmul(L, ε))

    # observations
    with plate("N", size=len(traps)):
        λ = deterministic('λ', jnp.exp(α + π[b] + θ[m] + (β + κ[b]) * traps + log_sq_foot))
        return sample('y', GammaPoisson(φ, φ / λ))
```


### Prepare data {#prepare-data}

The predictors, the fake predictors, and the observations are all the
same as before, but we have to prepare the fake parameters.

```python
gp_params = {
    'α': 2 * jnp.ones(1),
    'β': -1 * jnp.ones(1),
    'φ': jnp.ones(1),
    'σ': 0.5 * jnp.ones(1),
    'τ': 0.5 * jnp.ones(1),
    'ζ': 0.2 * random.normal(rng, [4]),
    'γ': 0.2 * random.normal(rng, [4]),
    'π0': 0.2 * random.normal(rng, [10]),
    'κ0': 0.2 * random.normal(rng, [10]),
    'ε': 0.5 * random.normal(rng, [42]),
    'χ': 0.5 * jnp.ones(1),
    'ell': 0.5 * jnp.ones(1),
}
```


### prior predictive check {#prior-predictive-check}

```python
gp_prior_pred = get_prior_pred(gp, ar_X)
print(gp_prior_pred['y'].min(), gp_prior_pred['y'].max())
for k, v in gp_prior_pred.items():
    print(k, v.shape)
```

```text
0 165065344
ell (1000,)
y (1000, 360)
α (1000,)
β (1000,)
γ (1000, 4)
ε (1000, 42)
ζ (1000, 4)
θ (1000, 42)
κ (1000, 10)
κ0 (1000, 10)
λ (1000, 360)
π (1000, 10)
π0 (1000, 10)
σ (1000,)
τ (1000,)
φ (1000,)
χ (1000,)
```

```python
sns.kdeplot(gp_prior_pred['ell']);
```

{{< figure src="/ox-hugo/959cc1a009f65e0df1511d4efc1c40462327431a.png" >}}

let's also look at the prior for the monthly effects. First the individual functions

<a id="code-snippet--vis-gp-prior-individual"></a>
```python
fig, ax = plt.subplots(figsize=(16, 8))
for i in random.randint(rng, [10], 0, 1000):
    ax.plot(gp_prior_pred['θ'][i], alpha=0.5)
```

{{< figure src="/ox-hugo/1acf7ac9f96977800d1f92bd199f01fae9299e01.png" >}}

Compared with before, the prior for the monthly effects is much
smoother. And the overall shape

<a id="code-snippet--vis-gp-prior-general"></a>
```python
fig, ax = plt.subplots(figsize=(16, 8))
for i in random.randint(rng, [400], 0, 1000):
    ax.plot(gp_prior_pred['θ'][i], color=colors[0], alpha=0.1)
```

{{< figure src="/ox-hugo/18cecf15424ac54b3b97f44274664f4442b3fe58.png" >}}


### Fit the model to fake data {#fit-the-model-to-fake-data}

```python
recover_params(gp, gp_params, ar_fX);
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
       ell      0.99      0.22      0.97      0.64      1.34   1163.45      1.00
         α      2.09      0.41      2.08      1.41      2.72   1554.71      1.00
         β     -0.81      0.29     -0.81     -1.27     -0.34   2292.02      1.00
      γ[0]     -0.04      0.26     -0.04     -0.44      0.41   2635.29      1.00
      γ[1]     -0.09      0.29     -0.09     -0.57      0.36   2309.17      1.00
      γ[2]     -0.21      0.23     -0.22     -0.56      0.17   2082.79      1.00
      γ[3]      0.00      0.40      0.00     -0.63      0.65   2246.11      1.00
      ε[0]     -0.08      0.49     -0.08     -0.89      0.69   1566.23      1.00
      ε[1]     -0.53      0.43     -0.52     -1.19      0.18   2394.57      1.00
      ε[2]      0.40      0.49      0.39     -0.38      1.22   1834.07      1.00
      ε[3]     -0.67      0.50     -0.68     -1.54      0.10   2153.73      1.00
      ε[4]      1.13      0.55      1.11      0.26      2.09   1953.03      1.00
      ε[5]     -0.36      0.63     -0.37     -1.38      0.70   1838.90      1.00
      ε[6]     -0.33      0.54     -0.34     -1.22      0.53   1942.93      1.00
      ε[7]     -0.10      0.55     -0.12     -0.99      0.79   2149.46      1.00
      ε[8]      0.67      0.56      0.66     -0.20      1.59   1879.40      1.00
      ε[9]     -0.13      0.70     -0.07     -1.27      1.01   1685.53      1.00
     ε[10]     -1.29      0.71     -1.31     -2.53     -0.26   1545.06      1.00
     ε[11]      1.32      0.66      1.32      0.18      2.35   3058.87      1.00
     ε[12]      0.02      1.01      0.02     -1.59      1.69   6472.44      1.00
     ε[13]     -0.00      1.00     -0.00     -1.61      1.71   7164.28      1.00
     ε[14]      0.00      1.01      0.02     -1.65      1.63   7037.57      1.00
     ε[15]     -0.00      1.02      0.01     -1.71      1.67   6782.78      1.00
     ε[16]      0.03      0.97      0.04     -1.43      1.69   5946.62      1.00
     ε[17]      0.01      0.99      0.01     -1.59      1.67   6932.34      1.00
     ε[18]     -0.01      0.97     -0.02     -1.56      1.59   6569.84      1.00
     ε[19]      0.00      1.01      0.01     -1.68      1.67   8033.05      1.00
     ε[20]     -0.01      0.98     -0.01     -1.57      1.63   6489.19      1.00
     ε[21]     -0.01      0.99      0.01     -1.69      1.51   7715.44      1.00
     ε[22]      0.00      1.00     -0.01     -1.70      1.55   6900.60      1.00
     ε[23]      0.01      0.97      0.02     -1.54      1.65   6635.62      1.00
     ε[24]     -0.00      0.99     -0.00     -1.67      1.60   7996.66      1.00
     ε[25]      0.00      0.99     -0.00     -1.59      1.67   6896.78      1.00
     ε[26]      0.00      1.04      0.00     -1.66      1.76   6595.81      1.00
     ε[27]      0.01      1.00      0.01     -1.60      1.64   5883.69      1.00
     ε[28]     -0.01      1.01     -0.01     -1.68      1.65   7266.64      1.00
     ε[29]      0.02      0.99      0.04     -1.57      1.70   7414.01      1.00
     ε[30]      0.00      1.01     -0.00     -1.71      1.64   7297.08      1.00
     ε[31]     -0.01      0.97     -0.03     -1.45      1.71   7038.31      1.00
     ε[32]     -0.00      0.99      0.01     -1.76      1.50   7057.45      1.00
     ε[33]      0.00      1.00     -0.00     -1.62      1.63   6911.22      1.00
     ε[34]      0.02      1.00      0.05     -1.64      1.58   5661.94      1.00
     ε[35]     -0.01      1.00     -0.02     -1.64      1.64   6493.23      1.00
     ε[36]     -0.02      0.99     -0.01     -1.62      1.60   6615.04      1.00
     ε[37]      0.01      0.99     -0.00     -1.60      1.66   5453.64      1.00
     ε[38]      0.01      1.00      0.02     -1.76      1.52   7035.79      1.00
     ε[39]      0.00      1.00     -0.02     -1.63      1.68   8376.90      1.00
     ε[40]     -0.01      1.01     -0.02     -1.69      1.64   6665.20      1.00
     ε[41]      0.01      1.02      0.01     -1.72      1.61   7669.84      1.00
      ζ[0]     -0.02      0.21     -0.02     -0.35      0.33   2873.91      1.00
      ζ[1]      0.06      0.24      0.06     -0.33      0.48   2609.62      1.00
      ζ[2]     -0.01      0.19     -0.01     -0.32      0.29   2402.43      1.00
      ζ[3]     -0.41      0.32     -0.43     -0.89      0.14   2440.78      1.00
     κ0[0]      0.18      0.91      0.17     -1.29      1.65   4303.11      1.00
     κ0[1]     -0.64      0.67     -0.63     -1.68      0.46   3204.27      1.00
     κ0[2]      0.09      0.86      0.08     -1.35      1.46   4117.93      1.00
     κ0[3]     -0.24      0.60     -0.21     -1.15      0.82   3132.91      1.00
     κ0[4]     -0.30      0.86     -0.29     -1.67      1.11   3210.99      1.00
     κ0[5]      0.62      0.67      0.62     -0.53      1.68   3107.19      1.00
     κ0[6]      0.08      0.76      0.08     -1.13      1.37   3127.11      1.00
     κ0[7]      0.47      0.84      0.46     -0.91      1.87   3368.57      1.00
     κ0[8]     -0.32      0.75     -0.33     -1.50      0.96   2705.35      1.00
     κ0[9]      0.06      0.75      0.06     -1.14      1.29   2939.94      1.00
     π0[0]      0.10      0.93      0.10     -1.34      1.74   4009.58      1.00
     π0[1]     -0.17      0.63     -0.17     -1.18      0.88   3548.05      1.00
     π0[2]      0.16      0.87      0.16     -1.29      1.60   4180.48      1.00
     π0[3]     -0.01      0.62     -0.01     -1.10      0.95   4104.80      1.00
     π0[4]     -0.21      0.88     -0.21     -1.65      1.23   3567.60      1.00
     π0[5]     -0.04      0.65     -0.03     -1.11      0.97   3204.83      1.00
     π0[6]      0.36      0.77      0.35     -0.93      1.61   3533.38      1.00
     π0[7]      0.13      0.84      0.13     -1.36      1.44   4347.31      1.00
     π0[8]     -0.24      0.77     -0.23     -1.45      1.06   3032.38      1.00
     π0[9]      0.03      0.79      0.03     -1.30      1.31   3287.84      1.00
         σ      0.44      0.19      0.41      0.16      0.70   2281.87      1.00
         τ      0.55      0.21      0.51      0.22      0.85   2315.64      1.00
         φ      0.95      0.05      0.95      0.87      1.04   6292.90      1.00
         χ      0.71      0.28      0.65      0.33      1.12   1316.37      1.00

Number of divergences: 1
```

there are a few divergent transitions.


### Fit model to the real data {#fit-model-to-the-real-data}

```python
gp_mcmc = run_mcmc(gp, ar_X, vs_y)
```

```text

                mean       std    median      5.0%     95.0%     n_eff     r_hat
       ell      1.89      0.30      1.92      1.39      2.41    458.67      1.00
         α      2.10      0.48      2.12      1.34      2.89   3102.83      1.00
         β     -2.15      0.38     -2.14     -2.74     -1.49   5313.36      1.00
      γ[0]     -1.07      0.70     -1.08     -2.17      0.09   6592.57      1.00
      γ[1]      1.04      0.53      1.04      0.19      1.92   5100.79      1.00
      γ[2]     -0.04      0.53     -0.04     -0.89      0.82   5660.29      1.00
      γ[3]      0.40      0.80      0.41     -0.88      1.71   8232.27      1.00
      ε[0]     -1.89      0.57     -1.88     -2.78     -0.91   2062.68      1.00
      ε[1]     -0.01      0.62      0.00     -1.05      0.95   4330.22      1.00
      ε[2]     -0.95      0.68     -0.95     -2.10      0.11   5246.07      1.00
      ε[3]     -0.74      0.73     -0.74     -1.87      0.48   5418.46      1.00
      ε[4]     -0.79      0.77     -0.80     -2.11      0.45   4088.29      1.00
      ε[5]     -0.20      0.87     -0.21     -1.68      1.16   1928.03      1.00
      ε[6]      0.23      0.83      0.28     -1.09      1.64   1313.17      1.00
      ε[7]     -0.34      0.86     -0.33     -1.75      1.06   1410.53      1.00
      ε[8]     -0.90      0.80     -0.91     -2.15      0.46   2647.05      1.00
      ε[9]      0.15      1.02      0.16     -1.46      1.92    961.68      1.00
     ε[10]      1.02      0.77      1.00     -0.28      2.24   2947.18      1.00
     ε[11]      0.37      0.83      0.38     -0.99      1.74   2545.57      1.00
     ε[12]     -0.25      0.80     -0.24     -1.56      1.09   1751.65      1.00
     ε[13]     -0.12      0.75     -0.14     -1.24      1.22   3348.76      1.00
     ε[14]      0.31      0.80      0.31     -0.99      1.66   1581.31      1.00
     ε[15]      0.40      0.76      0.40     -0.79      1.68   2593.28      1.00
     ε[16]     -0.02      0.72     -0.03     -1.17      1.16   4790.57      1.00
     ε[17]      0.24      0.72      0.25     -0.93      1.41   3753.51      1.00
     ε[18]      0.39      0.74      0.39     -0.77      1.66   4044.91      1.00
     ε[19]      0.72      0.78      0.73     -0.60      1.97   1878.41      1.00
     ε[20]      0.61      0.85      0.64     -0.76      2.01   1640.95      1.00
     ε[21]     -0.03      0.73     -0.03     -1.22      1.16   3517.36      1.00
     ε[22]      0.22      0.69      0.21     -0.92      1.31   5808.59      1.00
     ε[23]      0.58      0.71      0.56     -0.52      1.81   5710.58      1.00
     ε[24]      0.57      0.73      0.55     -0.60      1.80   6716.44      1.00
     ε[25]      0.39      0.79      0.41     -0.85      1.72   3163.80      1.00
     ε[26]     -0.09      0.85     -0.11     -1.56      1.20   1542.36      1.00
     ε[27]     -0.22      0.80     -0.23     -1.52      1.11   2914.43      1.00
     ε[28]      0.55      0.88      0.54     -0.96      1.88   1249.82      1.00
     ε[29]      1.09      0.74      1.08     -0.13      2.31   5341.08      1.00
     ε[30]      0.60      0.83      0.63     -0.68      2.04   3046.75      1.00
     ε[31]     -0.03      0.96     -0.04     -1.47      1.58   1215.70      1.00
     ε[32]     -0.23      0.85     -0.25     -1.49      1.31   1874.72      1.00
     ε[33]      0.44      0.94      0.43     -1.12      1.93   1124.54      1.00
     ε[34]      1.01      0.80      1.01     -0.34      2.27   6280.95      1.00
     ε[35]      0.36      0.93      0.38     -1.23      1.85   7738.96      1.00
     ε[36]     -0.02      0.96     -0.02     -1.56      1.59  11199.95      1.00
     ε[37]     -0.01      0.99     -0.01     -1.66      1.60   9371.78      1.00
     ε[38]      0.03      1.01      0.04     -1.72      1.62  12640.21      1.00
     ε[39]     -0.01      1.01     -0.02     -1.59      1.84  10566.93      1.00
     ε[40]     -0.01      1.03     -0.01     -1.60      1.77  11165.29      1.00
     ε[41]     -0.01      1.01     -0.00     -1.77      1.58   8641.86      1.00
      ζ[0]     -0.35      0.53     -0.35     -1.22      0.52   4919.83      1.00
      ζ[1]      0.45      0.44      0.45     -0.23      1.22   5325.06      1.00
      ζ[2]     -0.21      0.41     -0.22     -0.86      0.45   5166.77      1.00
      ζ[3]      0.04      0.62      0.04     -0.99      1.05   5470.76      1.00
     κ0[0]      0.94      0.76      0.95     -0.29      2.18   5978.95      1.00
     κ0[1]     -0.07      0.80     -0.08     -1.32      1.35   7675.49      1.00
     κ0[2]     -0.29      0.82     -0.28     -1.66      1.05   7195.75      1.00
     κ0[3]     -0.22      0.76     -0.20     -1.43      1.05   6386.57      1.00
     κ0[4]      0.19      0.80      0.19     -1.10      1.51   8271.44      1.00
     κ0[5]     -0.20      0.85     -0.18     -1.61      1.16   7982.53      1.00
     κ0[6]      0.10      0.83      0.10     -1.26      1.49   6342.57      1.00
     κ0[7]     -0.62      0.96     -0.61     -2.10      1.02   8385.83      1.00
     κ0[8]      0.35      0.82      0.36     -1.08      1.61   6622.21      1.00
     κ0[9]     -0.40      0.89     -0.39     -1.82      1.08   6759.59      1.00
     π0[0]      0.78      0.80      0.78     -0.59      2.01   5723.36      1.00
     π0[1]     -0.10      0.73     -0.09     -1.36      1.02   5842.51      1.00
     π0[2]     -0.07      0.76     -0.07     -1.30      1.19   5913.65      1.00
     π0[3]     -0.49      0.77     -0.49     -1.75      0.77   6191.62      1.00
     π0[4]      0.21      0.81      0.20     -1.18      1.44   6071.13      1.00
     π0[5]     -0.07      0.62     -0.07     -1.08      0.95   4880.43      1.00
     π0[6]      0.32      0.84      0.31     -1.01      1.67   6398.09      1.00
     π0[7]     -0.17      0.94     -0.16     -1.66      1.43   7709.35      1.00
     π0[8]     -0.08      0.86     -0.09     -1.49      1.33   6987.89      1.00
     π0[9]     -0.20      0.92     -0.19     -1.70      1.34   8743.86      1.00
         σ      0.57      0.23      0.53      0.26      0.94   3485.11      1.00
         τ      0.75      0.29      0.70      0.32      1.19   3982.11      1.00
         φ      5.56      0.97      5.47      4.03      7.10   2966.55      1.00
         χ      1.15      0.27      1.11      0.73      1.54   1220.73      1.00

Number of divergences: 0
```

```python
gp_posterior_samples = gp_mcmc.get_samples()

for k, v in gp_posterior_samples.items():
    print(k, v.shape)
```

```text
ell (4000,)
α (4000,)
β (4000,)
γ (4000, 4)
ε (4000, 42)
ζ (4000, 4)
θ (4000, 42)
κ (4000, 10)
κ0 (4000, 10)
λ (4000, 360)
π (4000, 10)
π0 (4000, 10)
σ (4000,)
τ (4000,)
φ (4000,)
χ (4000,)
```

<a id="code-snippet--vis-gp-ell"></a>
```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sns.histplot(gp_prior_pred['ell'], ax=axes[0])
sns.histplot(gp_posterior_samples['ell'], ax=axes[1]);
plt.xticks(jnp.arange(10));
```

{{< figure src="/ox-hugo/76e4a74a5f17ee2aebdac7ae545c0cad1d74cfb4.png" >}}

notice that the posterior is essentially cut off at 1 because there is
no way we can infer any value less than 1 from the data, since the
smallest distance in the data (the months) is already 1; we see here
that the scaling of the data is essential to the inference we can do.

<a id="code-snippet--vis-gp-chi"></a>
```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sns.histplot(gp_prior_pred['χ'], ax=axes[0])
sns.histplot(gp_posterior_samples['χ'], ax=axes[1]);
```

{{< figure src="/ox-hugo/033657fef9cc591eb16a214eb0af4a5c71e89bf5.png" >}}

<a id="code-snippet--vis-gp-chi-ell"></a>
```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sns.histplot(gp_prior_pred['χ'][:1000] / gp_prior_pred['ell'], ax=axes[0])
sns.histplot(gp_posterior_samples['χ'][:1000] / gp_prior_pred['ell'], ax=axes[1]);
```

{{< figure src="/ox-hugo/b9586fad7b309c0c6f3ed20f666c84ace5b58611.png" >}}


### Posterior predictive check {#posterior-predictive-check}

```python
gp_posterior_pred = get_posterior_pred(gp, ar_X, gp_mcmc)
print(gp_posterior_pred['y'].min(), gp_posterior_pred['y'].max())

for k, v in gp_posterior_pred.items():
    print(k, v.shape)
```

```text
0 402
y (4000, 360)
θ (4000, 42)
κ (4000, 10)
λ (4000, 360)
π (4000, 10)
```


### Density overlay {#density-overlay}

```python
overlay_kde(vs_y['y'], gp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/ba431a65f8103c873688bac63b8638e876b272ea.png" >}}


### Specific statistics {#specific-statistics}

```python
ppc_stat(prop_zero, vs_y['y'], gp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/eae6606e18b9a283556f11021245e53c630ca938.png" >}}

```python
ppc_stat(prop_large, vs_y['y'], gp_posterior_pred['y'])
```

{{< figure src="/ox-hugo/f617e4e7435d811c158dcd6e986d0b89aef6a024.png" >}}


### GP model {#gp-model}

Let's look at the Gaussian process part of the model, and compare it with the AR(1) model

<a id="code-snippet--vis-gp-ar"></a>
```python
ar_mean = ar_posterior_pred['θ'].mean(axis=0)
ar_std = ar_posterior_pred['θ'].std(axis=0)
gp_mean = gp_posterior_pred['θ'].mean(axis=0)
gp_std = gp_posterior_pred['θ'].std(axis=0)
x = jnp.arange(42)

fig, ax = plt.subplots(figsize=(16, 8))
plt.axvline(35, color=colors[2], alpha=0.5, lw=4)
ax.plot(x, ar_mean, 'o-', label='AR')
ax.fill_between(x, ar_mean-ar_std, ar_mean+ar_std, alpha=0.3)
ax.plot(x, gp_mean, 'o-', label='GP', color=colors[1])
ax.fill_between(x, gp_mean-gp_std, gp_mean+gp_std, color=colors[1], alpha=0.3)
plt.legend();
```

{{< figure src="/ox-hugo/a8b9c3d1f6f6ca91cee8af35389b92de9f6b5049.png" >}}

Although the overall predictions are similar, we see that the
specific predictions of the Gaussian process is quite different from the
AR(1) process. GP avoids the sharp spikes and sudden jumps that's
evident in the AR(1) process by enforcing a global stationarity
and local smoothness, the result is more regularised. We are much less
likely to see extreme predictions.


### posterior prediction by groups {#posterior-prediction-by-groups}

<a id="code-snippet--vis-gp-building"></a>
```python
plot_grouped_mean(pest_more['complaints'], gp_posterior_pred['y'], pest_more['b'], 2, 5, (12,4))
```

{{< figure src="/ox-hugo/dc04cbb297e5f221501db72241b60dc4aee0de5a.png" >}}

We have been able to further reduce the prediction variance.

<a id="code-snippet--vis-gp-month"></a>
```python
plot_grouped_mean(pest_more['complaints'], gp_posterior_pred['y'], pest_more['m'], 6, 6, (12,12))
```

{{< figure src="/ox-hugo/be4c6e982e2fa4bffe2103d2cb3b4689e91fed5a.png" >}}

generally speaking, the monthly prediction looks good.


## Ending thoughts {#ending-thoughts}

We won't go on expanding the model, because it looks like our model is already good enough. But good enough for what? Each model building practice should have the specific model, and the specific decisions to make about the model in mind. With observed social observations, we can never build a model good enough to capture all the aspects of the data, or how the data is really generated. All that we can do, when a problem is at hand, is to construct a model that solves or approximately solves the problems. Since we didn't touch on the decision making part in this post, we'll stop the model building here because the model outcome data seems to resemble the observed data enough.
