
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

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
    dpi: int = 300,
) -> None:
    """Overlay currents vs time. Pass a dict like {'IL': IL, 'INa': INa, ...}."""
    t_s = _to_seconds(t_ms)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, trace in currents.items():
        ax.plot(t_s, trace, label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()

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

    min_y = float(min(np.min(Vm_norm), np.min(ci_norm)))
    max_y = float(max(np.max(Vm_norm), np.max(ci_norm)))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(t_s, Vm_norm, color="blue", label="Vm (normalised)")
    ax1.set_ylabel("Normalised $V_m$", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2.plot(t_s, ci_norm, color="red", label="Ca (normalised)")
    ax2.set_ylabel("Normalised $c_i$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylim(min_y * 1.1, max_y * 1.1)
    ax2.set_ylim(min_y * 1.1, max_y * 1.1)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    _save_show(fig, savepath, dpi=dpi, show=show)
