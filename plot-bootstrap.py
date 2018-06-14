"""
Plot the distribution of shape parameters found from circle fits with
bootstrap resampling
"""
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import circle_fit

try:
    REGION_FILE = str(sys.argv[1])
    FITS_FILE = str(sys.argv[2])
    DELTA_THETA = float(sys.argv[3])
    REPLACEMENT_FRACTION = float(sys.argv[4])
    FIGFILE = str(sys.argv[5])
except:
    sys.exit(f"Usage: {sys.argv[0]} REGION_FILE FITS_FILE DELTA_THETA REPLACEMENT_FRACTION FIGFILE")


def iqr(x):
    """Inter-quartile range"""
    return np.diff(np.nanpercentile(x, [25, 75]))[0]

def mad(x):
    """Median absolute deviation rescaled to sigma for Gaussian"""
    return 1.4826*np.nanmedian(np.abs(x - np.nanmedian(x)))


fit = circle_fit.FitWithErrors(
    REGION_FILE, FITS_FILE,
    delta_theta=DELTA_THETA,
    nbootstrap=100,
    fraction=REPLACEMENT_FRACTION,
    verbose=False,
)


sns.set_color_codes()
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

ax = axes[0, 0]
ax.scatter(fit.shape_dist.Pi, fit.shape_dist.Lambda,
           marker=".", alpha=0.2, edgecolors="none")
ax.plot(fit.shape.Pi, fit.shape.Lambda, '+', color="orange")

text = rf"$\Pi' = {fit.shape.Pi:.2f} \pm {mad(fit.shape_dist.Pi):.2f}$"
text += "\n" + rf"$\Lambda' = {fit.shape.Lambda:.2f} \pm {mad(fit.shape_dist.Lambda):.2f}$"
rcorr = fit.shape_dist.corr[0, 1]
text += "\n" + rf"$r = {rcorr:.4f}$"
ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize="x-small")

ax.set(
    xlabel=r"Projected planitude, $\Pi'$",
    ylabel=r"Projected alatude, $\Lambda'$",
    xlim=[0.0, 5.0],
    ylim=[0.0, 5.0],
)

ax = axes[0, 1]
ax.scatter(fit.shape_dist.Lambda, fit.shape_dist.dLambda,
           marker=".", alpha=0.2, edgecolors="none")
ax.plot(fit.shape.Lambda, fit.shape.dLambda, '+', color="orange")

text = rf"$\Lambda' = {fit.shape.Lambda:.2f} \pm {mad(fit.shape_dist.Lambda):.2f}$"
text += "\n" + rf"$\Delta\Lambda' = {fit.shape.dLambda:.2f} \pm {mad(fit.shape_dist.dLambda):.2f}$"
rcorr = fit.shape_dist.corr[1, 2]
text += "\n" + rf"$r = {rcorr:.4f}$"
ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize="x-small")

ax.set(
    xlabel=r"Projected alatude, $\Lambda'$",
    ylabel=r"Alatude asymmetry, $\Delta\Lambda'$",
    xlim=[0.0, 5.0],
    ylim=[-1.2, 1.2],
)

ax = axes[1, 0]
ax.scatter(fit.shape_dist.R0, fit.shape_dist.Pi,
           marker=".", alpha=0.2, edgecolors="none")
ax.plot(fit.shape.R0, fit.shape.Pi, '+', color="orange")
text = rf"$R_0' = {fit.shape.R0:.2f} \pm {mad(fit.shape_dist.R0):.2f}$"
text += "\n" + rf"$\Pi' = {fit.shape.Pi:.2f} \pm {mad(fit.shape_dist.Pi):.2f}$"
rcorr = fit.shape_dist.corr[3, 0]
text += "\n" + rf"$r = {rcorr:.4f}$"
ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize="x-small")


ax.set(
    xlabel=r"Apex distance, $R_0'$",
    ylabel=r"Projected planitude, $\Pi'$",
    xlim=[0.8*fit.shape.R0, 1.2*fit.shape.R0],
    ylim=[0.0, 5.0],
)

ax = axes[1, 1]
ax.scatter(fit.shape_dist.angle, fit.shape_dist.dLambda,
           marker=".", alpha=0.2, edgecolors="none")
ax.plot(fit.shape.angle, fit.shape.dLambda, '+', color="orange")
text = rf"$\theta = {fit.shape.angle:.1f} \pm {mad(fit.shape_dist.angle):.1f}$"
text += "\n" + rf"$\Delta\Lambda' = {fit.shape.dLambda:.2f} \pm {mad(fit.shape_dist.dLambda):.2f}$"
rcorr = fit.shape_dist.corr[4, 2]
text += "\n" + rf"$r = {rcorr:.4f}$"
ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize="x-small")

ax.set(
    xlabel=r"Axis angle, $\theta$",
    ylabel=r"Alatude asymmetry, $\Delta\Lambda'$",
    xlim=[fit.shape.angle - 30.0, fit.shape.angle + 30.0],
    ylim=[-1.2, 1.2],
)

sns.despine()
fig.tight_layout()
fig.savefig(FIGFILE)

