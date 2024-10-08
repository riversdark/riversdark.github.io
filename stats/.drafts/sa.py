#%%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display

from mpl_toolkits.mplot3d import Axes3D


#%%
def abs_peaks_func(x, y):
    # in contrast to the peaks function: all negative values are multiplied by (-1)
    return jnp.abs(
        3.0 * (1 - x) ** 2 * jnp.exp(-(x**2) - (y + 1) ** 2)
        - 10.0 * (x / 5 - x**3 - y**5) * jnp.exp(-(x**2) - y**2)
        - 1.0 / 3 * jnp.exp(-((x + 1) ** 2) - y**2)
    )


#%%
n = 100  # number of dimension
pdf = np.zeros([n, n])
sigma = jnp.zeros([n, n])
# s = jnp.zeros([n, n])
x = -3.0
for i in range(0, n):
    y = -3.0
    for j in range(0, n):
        pdf[j, i] = abs_peaks_func(x, y)
        y = y + 6.0 / (n - 1)
    x = x + 6.0 / (n - 1)

pdf = jnp.array(pdf)
pdf = pdf / pdf.max()
energy = -jnp.log(pdf)


#%%
def plot_2d_surface(ax, x, y, pdf, title=None):
    x, y = jnp.meshgrid(x, y)
    im = ax.imshow(
        pdf,
        cmap=plt.cm.coolwarm,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto'
    )
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    plt.tight_layout()

X = jnp.arange(0, 100 + 100.0 / (n - 1), 100.0 / (n - 1))
Y = jnp.arange(0, 100 + 100.0 / (n - 1), 100.0 / (n - 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

plot_2d_surface(ax1, Y, X, pdf, title="pdf")
plot_2d_surface(ax2, X, Y, energy, title="energy")

plt.tight_layout()
plt.show()


#%% the heat bath
temperature = 16  # initial temperature for the plots
stepT = 8  # how many steps should the Temperature be *0.2  for
x = np.arange(0, 100 + 100.0 / (n - 1), 100.0 / (n - 1))
y = np.arange(0, 100 + 100.0 / (n - 1), 100.0 / (n - 1))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

fig.suptitle("Heat bath with varying temperatures (T)", fontsize=16)

for i in range(stepT):
    sigma = np.exp(-(energy) / temperature)
    sigma = sigma / sigma.max()
    ttl = f"T={temperature:0.2f}"
    
    ax = axes[i]
    im = ax.imshow(sigma, cmap=plt.cm.coolwarm,
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='auto')
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax)
    
    temperature = temperature * 0.5

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top margin to accommodate the overall title
plt.show()


#%% SA algorithm
def sim_anneal(proposal="gaussian", sigma=10, seed=jax.random.PRNGKey(0)):
    seed1, seed2 = jax.random.split(seed)
    x_start = jnp.array(
        [
            jnp.floor(jax.random.uniform(seed1, minval=0, maxval=100)),
            jnp.floor(jax.random.uniform(seed2, minval=0, maxval=100)),
        ]
    )  # x_start
    xcur = x_start.astype(int)  # x current
    n_samples = 300  # number of samples to keep
    T = 1  # start temperature
    alpha = 0.99  # cooling schedule

    # list of visited points, temperatures, probabilities
    x_hist = xcur  # will be (N,2) array
    prob_hist = []
    temp_hist = []

    nreject = 0
    iis = 0  # number of accepted points
    n_proposed_points = 0  # num proposed points
    while n_proposed_points < n_samples:
        _, seed = jax.random.split(seed)
        n_proposed_points = n_proposed_points + 1
        if proposal == "uniform":
            seeds = jax.random.split(seed)
            xnew = jnp.array(
                [
                    jnp.floor(jax.random.uniform(seeds[0], minval=0, maxval=100)),
                    jax.random.uniform(seeds[1], minval=0, maxval=100),
                ]
            )
            # print(xnew)
        elif proposal == "gaussian":
            xnew = xcur + jax.random.normal(seed, shape=(2,)) * sigma
            xnew = jnp.maximum(xnew, 0)
            xnew = jnp.minimum(xnew, 99)
        else:
            raise ValueError("Unknown proposal")
        xnew = xnew.astype(int)

        # compare energies
        Ecur = energy[xcur[0], xcur[1]]
        Enew = energy[xnew[0], xnew[1]]
        deltaE = Enew - Ecur
        # print([n_proposed_points, xcur, xnew, Ecur, Enew, deltaE])

        temp_hist.append(T)
        T = alpha * T
        p_accept = jnp.exp(-1.0 * deltaE / T)
        # print(p_accept)
        p_accept = min(1, p_accept)
        test = jax.random.uniform(jax.random.split(seed)[0], minval=0, maxval=1)
        # print(test)
        if test <= p_accept:
            xcur = xnew
            iis = iis + 1
        else:
            nreject += 1

        x_hist = jnp.vstack((x_hist, xcur))
        prob_hist.append(pdf[xcur[0], xcur[1]])

    n_proposed_points = n_proposed_points + 1
    print(f"nproposed {n_proposed_points}, naccepted {iis}, nreject {nreject}")
    return x_hist, prob_hist, temp_hist


# # Run experiments

# In[ ]:


proposals = ["gaussian", "uniform"]
x_hist = {}
prob_hist = {}
temp_hist = {}
for proposal in proposals:
    print(proposal)
    x_hist[proposal], prob_hist[proposal], temp_hist[proposal] = sim_anneal(
        proposal=proposal, seed=jax.random.PRNGKey(25)
    )


# In[ ]:


for proposal in proposals:
    fig, ax1 = plt.subplots()
    
    # Add proposal name as plot title
    fig.suptitle(f"Proposal: {proposal.capitalize()}")
    
    # Temperature plot on left y-axis
    ax1.plot(temp_hist[proposal], "r--", label="temperature")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("Temperature", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    
    # Probability plot on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(prob_hist[proposal], "g-", label="probability")
    ax2.set_ylabel("Probability", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0.55, 0.35), fontsize=8)
    
    sns.despine(right=False)


# # Plot the trace 

# In[ ]:


global_markersize = 6
step_markersize = 4
for proposal in proposals:
    probs = prob_hist[proposal]
    xa = x_hist[proposal]

    fig = plt.figure()
    ax = plt.gca()
    contour = ax.imshow(pdf.transpose(), aspect="auto", extent=[0, 100, 100, 0], interpolation="none")
    # fig.colorbar(contour, ax=ax)

    # Global maximum with red circle

    # Plot the trace with white line connecting the points
    ax.plot(xa[:, 0], xa[:, 1], "w-", linewidth=1, alpha=0.5)
    
    ax.plot(xa[:, 0], xa[:, 1], "w.", markersize=step_markersize)
    
    # Plot the global maximum
    ind = np.unravel_index(np.argmax(pdf, axis=None), pdf.shape)
    ax.plot(ind[0], ind[1], "ro", markersize=global_markersize, label="global maxima")

    # Highlight starting point
    ax.plot(xa[0, 0], xa[0, 1], "go", markersize=global_markersize, label="starting point")

    # Highlight ending point
    ax.plot(xa[-1, 0], xa[-1, 1], "bo", markersize=global_markersize, label="ending point")

    ax.set_ylabel("$x2$")
    ax.set_xlabel("$x1$")
    ax.legend(framealpha=0.5)

