"""Generate the five assignment graphs as JPG files."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "font.size": 10})


def save(fig, filename):
    """Save a figure as a readable JPG and release its memory."""
    fig.tight_layout()
    fig.savefig(OUT / filename, format="jpg", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Q1(a): L(mu,x) and minimizers
x = np.linspace(-1, 4, 800)
fig, ax = plt.subplots(figsize=(7, 4.5))
for mu in [0, 1, 2, 3]:
    L = x**2 + mu * (1 - x)
    xm = mu / 2
    ax.plot(x, L, label=fr"$\mu={mu}$")
    ax.plot(xm, xm**2 + mu * (1 - xm), "o", ms=6)
    ax.annotate(fr"$x(\mu)={xm:g}$", (xm, xm**2 + mu * (1 - xm)),
                xytext=(5, 7), textcoords="offset points", fontsize=8)
ax.set(xlabel="$x$", ylabel="$L(\mu,x)$", title="Q1(a): Lagrangian curves and minimizers")
ax.legend()
save(fig, "q1a_lagrangian_curves.jpg")

# Q1(b): dual function
mu = np.linspace(0, 5, 600)
d = mu - mu**2 / 4
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(mu, d, lw=2, label=r"$d(\mu)=\mu-\mu^2/4$")
ax.plot(2, 1, "ro", label=r"$(\mu^*,d^*)=(2,1)$")
ax.axhline(1, color="gray", ls="--", lw=0.8)
ax.set(xlabel=r"$\mu\geq0$", ylabel="$d(\mu)$", title="Q1(b): Dual function")
ax.legend()
save(fig, "q1b_dual_function.jpg")

# Q1(c): primal function
xx = np.linspace(-1, 3, 800)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(xx[xx >= 1], xx[xx >= 1]**2, lw=2, label=r"$g(x)=x^2$ for $x\geq1$")
ax.axvline(1, color="gray", ls="--", lw=0.8, label="feasibility boundary")
ax.annotate(r"$g(x)=+\infty$ for $x<1$", xy=(-0.5, 5), ha="center", color="darkred")
ax.set(xlim=(-1, 3), ylim=(0, 9), xlabel="$x$", ylabel="$g(x)$", title="Q1(c): Primal function")
ax.legend()
save(fig, "q1c_primal_function.jpg")

# Q2(a): two saddle-point slices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
mu2 = np.linspace(0, 5, 400)
ax1.plot(mu2, np.ones_like(mu2), lw=2)
ax1.plot(2, 1, "ro")
ax1.set(xlabel=r"$\mu$", ylabel=r"$L(\mu,x^*)$", title=r"$x^*=1$: $L(\mu,1)=1$")
ax1.annotate("common value = 1", (2, 1), xytext=(8, 10), textcoords="offset points")
xs = np.linspace(-2, 4, 500)
ax2.plot(xs, xs**2 + 2 * (1 - xs), lw=2)
ax2.plot(1, 1, "ro")
ax2.set(xlabel="$x$", ylabel=r"$L(\mu^*,x)$", title=r"$\mu^*=2$: $L(2,x)=(x-1)^2+1$")
ax2.annotate("minimum = 1", (1, 1), xytext=(8, 10), textcoords="offset points")
save(fig, "q2a_saddle_slices.jpg")

# Q3(b): three affine Lagrangian lines and their lower envelope
mu3 = np.linspace(0, 5, 600)
lines = {0: mu3, 1: np.ones_like(mu3), 2: 4 - mu3}
fig, ax = plt.subplots(figsize=(7, 4.3))
for xv, line in lines.items():
    ax.plot(mu3, line, label=fr"$x={xv}$")
env = np.minimum.reduce(list(lines.values()))
ax.plot(mu3, env, "k--", lw=2, label="lower envelope")
ax.set(xlabel=r"$\mu\geq0$", ylabel=r"$L(\mu,x)$", title="Q3(b): Lower envelope of affine Lagrangians")
ax.legend()
save(fig, "q3b_lower_envelope.jpg")

print(f"Saved JPG figures in {OUT.resolve()}")
