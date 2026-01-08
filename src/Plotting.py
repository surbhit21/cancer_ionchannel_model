
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
mpl.rcParams.update({
    "font.size": 12,          # default text
    "axes.titlesize": 14,     # axes title
    "axes.labelsize": 14,     # x and y labels
    "xtick.labelsize": 14,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
    "legend.fontsize": 14,    # legend
    "figure.titlesize": 16    # figure title
})

def ensure_dir_exists(file_path):
    """
    Ensures that the directory for the given file path exists.
    Creates directories recursively if they do not exist.

    Parameters:
    - file_path (str): Full path to the file (e.g., 'plots/figs/myplot.png')

    Returns:
    - None
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
def _to_seconds(t_ms: np.ndarray) -> np.ndarray:
    """Convert time from milliseconds to seconds."""
    return t_ms * 1e-3
def _save_show(fig: plt.Figure, savepath: Optional[Path], dpi: int = 300, show: bool = True) -> None:
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_vm_and_ci(
    t_ms: np.ndarray,
    Vm_mV: np.ndarray,
    ci_mM: np.ndarray,
    *,
    savepath: Optional[Path] = None,
    show: bool = True,
    dpi: int = 300,
) -> None:
    """Two stacked subplots: Vm and ci vs time."""
    t_s = _to_seconds(t_ms)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t_s, Vm_mV, c="blue", label="Vm")
    ax[1].plot(t_s, ci_mM, c="red", label="ci")

    ax[0].set_ylabel("Membrane voltage (Vm in mV)")
    ax[1].set_ylabel("Cytosolic calcium (ci in mM)")
    ax[1].set_xlabel("Time (s)")
    ax[0].set_xlabel("Time (s)")  # if you want both labels; otherwise remove this line

    # Put legend per-axis (avoids empty legend issues)
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")

    _save_show(fig, savepath, dpi=dpi, show=show)


def plot_currents(
    t_ms: np.ndarray,
    currents: dict[str, np.ndarray],
    *,
    ylabel: str = "Current (nA)",
    savepath: Optional[Path] = None,
    show: bool = True,
    labs  : dict[str, str] = {},
    dpi: int = 300,
) -> None:
    """Overlay currents vs time. Pass a dict like {'IL': IL, 'INa': INa, ...}."""
    t_s = _to_seconds(t_ms)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, trace in currents.items():
        ax.plot(t_s, trace, label=labs[name] if name in labs else name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()

    _save_show(fig, savepath, dpi=dpi, show=show)

def plot_gating_variables(
    v_ms: np.ndarray,
    currents: dict[str, np.ndarray],
    *,
    ylabels: np.ndarray = ["Current (nA)"],
    savepath: Optional[Path] = None,
    labs: dict[str, str] = {},
    show: bool = True,
    dpi: int = 300,
) -> None:
    """Overlay gating variable values vs voltage. Pass a dict like {'h': h, 'm': m, ...}."""
    

    fig, ax = plt.subplots(figsize=(16,6),ncols=2)
    for name, trace in currents.items():
        ax[0].plot(v_ms, trace[0], label=labs[name] if name in labs else name)
        ax[1].plot(v_ms, trace[1], label=labs[name] if name in labs else name)
    ax[0].set_xlabel("Voltage (mV)")
    ax[0].set_ylabel(ylabels[0])
    ax[1].set_xlabel("Voltage (mV)")
    ax[1].set_ylabel(ylabels[1])
    ax[0].legend()
    ax[1].legend()
    _save_show(fig, savepath, dpi=dpi, show=show)


def plot_normalised_vm_and_ci(
    t_ms: np.ndarray,
    Vm_mV: np.ndarray,
    ci_mM: np.ndarray,
    *,
    savepath: Optional[Path] = None,
    show: bool = True,
    dpi: int = 300,
) -> None:
    """
    Twin-axis plot of normalised Vm and ci.
      Vm_norm = (Vm - Vm0)/abs(Vm0)
      ci_norm = ci/ci0
    Uses same y-limits on both axes like your original code.
    """
    t_s = _to_seconds(t_ms)

    Vm0 = float(Vm_mV[0])
    ci0 = float(ci_mM[0])

    Vm_norm = (Vm_mV - Vm0) / abs(Vm0)
    ci_norm = ci_mM / ci0

    min_y1 = np.min(Vm_norm)
    max_y1 = np.max(Vm_norm)
    min_y2 = np.min(ci_norm)
    max_y2 = np.max(ci_norm)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(t_s, Vm_norm, color="blue", label="Vm (normalised)")
    ax1.set_ylabel("Normalised $V_m$", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2.plot(t_s, ci_norm, color="red", label="Ca (normalised)")
    ax2.set_ylabel("Normalised $c_i$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylim(min_y1 * 1.1, max_y1 * 1.1)
    ax2.set_ylim(min_y2 * 1.1, max_y2 * 1.1)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    _save_show(fig, savepath, dpi=dpi, show=show)



