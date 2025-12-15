from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple


@dataclass
class Params:
    # Maximal conductances (mS/cm^2)
    gNa: float = 0.15
    gCa: float = 0.03
    gK:  float = 1.1
    gCl: float = 0

    # NEW: Ca-activated K conductance
    gKCa: float = 0.2   # tune this (0.05–2.0 is a reasonable sweep)

    # Leak (placeholder)
    gL: float = 0.02
    EL: float = -60.0

    # Reversal potentials (mV)
    ENa: float = 55.0
    EK:  float = -75.0
    ECl: float = -20.0
    ECa: float = 120.0

    # Exponents
    p: int = 2
    q: int = 1

    # Membrane capacitance (µF/cm^2)
    C: float = 1.6

    # extenal input
    I_ext: float = 0.08
    
    # Calcium dynamics
    crest: float = 1.5e-5
    gamma: float = 1e-3
    f: float = 1
    r_cm: float = 1e-3
    F: float = 9.6485e4

    # NEW: KCa activation parameters (ci in mM)
    Kd_KCa: float = 1.0e-2  # mM (0.2 µM). Try 1e-4–1e-3
    n_KCa: float = 2 # Hill coefficient (2–4 common)


# -----------------------------
# "Known approximation" placeholders for x_inf(V) and tau_x(V)
# -----------------------------
def sigmoid_x_inf(Vm_mV: float, Vhalf: float, k: float) -> float:
    return 1.0 / (1.0 + np.exp(-(Vm_mV - Vhalf) / k))


def vdep_tau_ms(Vm_mV: float, tau_min: float, tau_max: float, Vhalf: float, k: float) -> float:
    s = 1.0 / (1.0 + np.exp((Vm_mV - Vhalf) / k))
    return tau_min + (tau_max - tau_min) * s


GateApprox = Dict[str, Dict[str, float]]
DEFAULT_GATE_APPROX: GateApprox = {
    "m": {"Vhalf": -35.0, "k": 7.0,  "tau_min": 0.05, "tau_max": 1.0,  "Vtau": -40.0, "ktau": 10.0},
    "h": {"Vhalf": -55.0, "k": -7.0, "tau_min": 0.5,  "tau_max": 8.0,  "Vtau": -55.0, "ktau": 12.0},
    "n": {"Vhalf": -30.0, "k": 10.0, "tau_min": 0.5,  "tau_max": 5.0,  "Vtau": -35.0, "ktau": 12.0},
    "s": {"Vhalf": -45.0, "k": 7.0,  "tau_min": 1.0,  "tau_max": 20.0, "Vtau": -30.0, "ktau": 10.0},
    "r": {"Vhalf": -20.0, "k": 8.0,  "tau_min": 1.0,  "tau_max": 15.0, "Vtau": -25.0, "ktau": 10.0},
}


def gate_rhs(x: float, Vm: float, gate_name: str, gate_approx: GateApprox) -> float:
    x_inf = sigmoid_x_inf(Vm, gate_approx[gate_name]["Vhalf"], gate_approx[gate_name]["k"])
    tau = vdep_tau_ms(
        Vm,
        gate_approx[gate_name]["tau_min"],
        gate_approx[gate_name]["tau_max"],
        gate_approx[gate_name]["Vtau"],
        gate_approx[gate_name]["ktau"],
    )
    return -(x - x_inf) / tau


# -----------------------------
# NEW: Ca-dependent activation (Hill)
# -----------------------------
def w_kca(ci_mM: float, p: Params) -> float:
    ci = max(ci_mM, 0.0)
    num = ci ** p.n_KCa
    den = num + (p.Kd_KCa ** p.n_KCa)
    return 0.0 if den == 0.0 else num / den


# -----------------------------
# Currents + full ODE system
# State y = [Vm, m, h, n, s, r, ci]
# -----------------------------
def currents(Vm: float, m: float, h: float, n: float, s: float, r: float, ci: float, p: Params) -> Tuple[float, float, float, float, float, float]:
    INa = p.gNa * (m**3) * h * (Vm - p.ENa)
    IK  = p.gK  * (n**4)       * (Vm - p.EK)
    ICa = p.gCa * (s**p.p)     * (Vm - p.ECa)
    ICl = p.gCl * (r**p.q)     * (Vm - p.ECl)
    IL  = p.gL               * (Vm - p.EL)

    # NEW: Ca-activated K current
    IKCa = p.gKCa * w_kca(ci, p) * (Vm - p.EK)

    return IL, INa, ICa, IK, ICl, IKCa


def ode_system(t_ms: float, y: np.ndarray, p: Params, gate_approx: GateApprox) -> np.ndarray:
    Vm, m, h, n, s, r, ci = y

    IL, INa, ICa, IK, ICl, IKCa = currents(Vm, m, h, n, s, r, ci, p)

    # Membrane equation: C dVm/dt = -(sum currents)
    dVm = (-(IL + INa + ICa + IK + ICl + IKCa) + p.I_ext) / p.C

    # Gating variables
    dm = gate_rhs(m, Vm, "m", gate_approx)
    dh = gate_rhs(h, Vm, "h", gate_approx)
    dn = gate_rhs(n, Vm, "n", gate_approx)
    ds = gate_rhs(s, Vm, "s", gate_approx)
    dr = gate_rhs(r, Vm, "r", gate_approx)
    
    if int(t_ms) == 3000:
        print("Vm", Vm, "ICa", ICa, "ci", ci, "wKCa", w_kca(ci, p))

    # Calcium dynamics (unchanged)
    dci = -(p.f * 3.0 * ICa) / (2000. * p.r_cm * p.F) - p.gamma * (ci - p.crest)

    return np.array([dVm, dm, dh, dn, ds, dr, dci], dtype=float)


if __name__ == "__main__":
    p = Params()
    gate_approx = DEFAULT_GATE_APPROX

    Vm0 = -60.0
    m0 = sigmoid_x_inf(Vm0, gate_approx["m"]["Vhalf"], gate_approx["m"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h"]["Vhalf"], gate_approx["h"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["n"]["Vhalf"], gate_approx["n"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["s"]["Vhalf"], gate_approx["s"]["k"])
    r0 = sigmoid_x_inf(Vm0, gate_approx["r"]["Vhalf"], gate_approx["r"]["k"])
    ci0 = 0.002

    y0 = np.array([Vm0, m0, h0, n0, s0, r0, ci0], dtype=float)

    t_span = (0.0, 15000.0)  # ms (give it time to settle into a limit cycle)
    t_eval = np.linspace(*t_span, 15001)

    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p, gate_approx),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    print("Integration success:", sol.success)
    print("Initial Vm (mV):", sol.y[0, 0])
    print("Initial ci (mM):", sol.y[6, 0])
    print("Final Vm (mV):", sol.y[0, -1])
    print("Final ci (mM):", sol.y[6, -1])

    
    w = (sol.y[6]**p.n_KCa) / (sol.y[6]**p.n_KCa + p.Kd_KCa**p.n_KCa)
    print(w[-2000:].min(), w[-2000:].max())

    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(sol.t, sol.y[0],c="blue")
    ax[1].plot(sol.t, sol.y[6],c="red")
    ax[0].set_xlabel('Time (ms)')
    ax[1].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Membrane voltage (Vm in mV)')
    ax[1].set_ylabel('Cytosolic calcium (ci in mM)')
    plt.legend()
    plt.savefig('./HHType_with_KCa.png',dpi=300)
    plt.show()
    
    t = sol.t
    Vm = sol.y[0]
    ci = sol.y[6]

    # Initial values
    Vm0 = Vm[0]
    ci0 = ci[0]

    # Normalised signals
    Vm_norm = (Vm - Vm0) / abs(Vm0)   # relative Vm change
    ci_norm = ci / ci0               # Ca fold-change

    # Create figure and twin axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Plot Vm (left axis)
    ax1.plot(t, Vm_norm, color="blue", label="Vm (normalised)")
    ax1.set_ylabel("Normalised $V_m$", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot Ca (right axis)
    ax2.plot(t, ci_norm, color="red", label="Ca (normalised)")
    ax2.set_ylabel("Normalised $c_i$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Shared x-axis
    ax1.set_xlabel("Time (ms)")

    # Optional: combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.savefig('./norm_HHType_with_KCa.png',dpi=300)
    plt.show()