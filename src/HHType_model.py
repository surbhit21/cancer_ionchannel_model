from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple


# -----------------------------
# Parameters (from the PDF where available)
# -----------------------------
@dataclass
class Params:
    # Maximal conductances (mS/cm^2)
    gNa: float = 0.85     # ḡNa :contentReference[oaicite:1]{index=1}
    gCa: float = 0.052    # ḡCa :contentReference[oaicite:2]{index=2}
    gK:  float = 1.1      # ḡK  :contentReference[oaicite:3]{index=3}
    gCl: float = 0.1      # ḡCl :contentReference[oaicite:4]{index=4}

    # Leak (NOT specified in the PDF snippet; set your own)
    gL: float = 0.02      # mS/cm^2 (placeholder)
    EL: float = -60.0     # mV (placeholder)

    # Reversal potentials (mV)
    ENa: float = 55.0     # :contentReference[oaicite:5]{index=5}
    EK:  float = -75.0    # :contentReference[oaicite:6]{index=6}
    ECl: float = -20.0    # :contentReference[oaicite:7]{index=7}
    ECa: float = 120.0    # :contentReference[oaicite:8]{index=8}

    # Exponents
    p: int = 2            # Ca activation exponent :contentReference[oaicite:9]{index=9}
    q: int = 1            # Cl activation exponent :contentReference[oaicite:10]{index=10}

    # Membrane capacitance (µF/cm^2)
    C: float = 1.6        # :contentReference[oaicite:11]{index=11}

    # Calcium dynamics
    crest: float = 1.5e-5 # mM :contentReference[oaicite:12]{index=12}
    gamma: float = 1e-4   # ms^-1 :contentReference[oaicite:13]{index=13}
    f: float = 0.005      # :contentReference[oaicite:14]{index=14}
    r_cm: float = 1e-3    # cm :contentReference[oaicite:15]{index=15}
    F: float = 9.6485e4   # C/mol :contentReference[oaicite:16]{index=16}


# -----------------------------
# "Known approximation" placeholders for x_inf(V) and tau_x(V)
# You can later replace these with alpha/beta fits:
# x_inf = alpha/(alpha+beta), tau = 1/(alpha+beta)
# -----------------------------
def sigmoid_x_inf(Vm_mV: float, Vhalf: float, k: float) -> float:
    """Steady-state activation/inactivation (0..1)."""
    return 1.0 / (1.0 + np.exp(-(Vm_mV - Vhalf) / k))


def vdep_tau_ms(Vm_mV: float, tau_min: float, tau_max: float, Vhalf: float, k: float) -> float:
    """
    Simple voltage-dependent time constant (ms), bounded in [tau_min, tau_max].
    """
    s = 1.0 / (1.0 + np.exp((Vm_mV - Vhalf) / k))
    return tau_min + (tau_max - tau_min) * s


# Gate parameter presets (PLACEHOLDERS).
# Tune these based on your patch clamp fits later.
GateApprox = Dict[str, Dict[str, float]]
DEFAULT_GATE_APPROX: GateApprox = {
    "m": {"Vhalf": -35.0, "k": 7.0,  "tau_min": 0.05, "tau_max": 1.0,  "Vtau": -40.0, "ktau": 10.0},
    "h": {"Vhalf": -55.0, "k": -7.0, "tau_min": 0.5,  "tau_max": 8.0,  "Vtau": -55.0, "ktau": 12.0},
    "n": {"Vhalf": -30.0, "k": 10.0, "tau_min": 0.5,  "tau_max": 5.0,  "Vtau": -35.0, "ktau": 12.0},
    "s": {"Vhalf": -25.0, "k": 6.0,  "tau_min": 1.0,  "tau_max": 20.0, "Vtau": -30.0, "ktau": 10.0},
    "r": {"Vhalf": -20.0, "k": 8.0,  "tau_min": 1.0,  "tau_max": 15.0, "Vtau": -25.0, "ktau": 10.0},
}


def gate_rhs(x: float, Vm: float, gate_name: str, gate_approx: GateApprox) -> float:
    """
    HH-form gating ODE:
      dx/dt = -(x - x_inf(V))/tau_x(V)
    :contentReference[oaicite:17]{index=17}
    """
    g = gate_approx[gate_name]
    x_inf = sigmoid_x_inf(Vm, g["Vhalf"], g["k"])
    tau = vdep_tau_ms(Vm, g["tau_min"], g["tau_max"], g["Vtau"], g["ktau"])
    return -(x - x_inf) / tau


# -----------------------------
# Currents + full ODE system
# State y = [Vm, m, h, n, s, r, ci]
# -----------------------------
def currents(Vm: float, m: float, h: float, n: float, s: float, r: float, p: Params) -> Tuple[float, float, float, float, float]:
    """
    Classical HH-style currents:
      INa = gNa * m^3 * h * (Vm - ENa)
      IK  = gK  * n^4     * (Vm - EK)
      ICa = gCa * s^p     * (Vm - ECa)
      ICl = gCl * r^q     * (Vm - ECl)
      IL  = gL            * (Vm - EL)
    :contentReference[oaicite:18]{index=18}
    """
    INa = p.gNa * (m**3) * h * (Vm - p.ENa)
    IK  = p.gK  * (n**4)       * (Vm - p.EK)
    ICa = p.gCa * (s**p.p)     * (Vm - p.ECa)
    ICl = p.gCl * (r**p.q)     * (Vm - p.ECl)
    IL  = p.gL               * (Vm - p.EL)
    return IL, INa, ICa, IK, ICl


def ode_system(t_ms: float, y: np.ndarray, p: Params, gate_approx: GateApprox, Iapp_uAcm2=0.0) -> np.ndarray:
    Vm, m, h, n, s, r, ci = y

    IL, INa, ICa, IK, ICl = currents(Vm, m, h, n, s, r, p)

    # Membrane equation:
    #   C dVm/dt = -(IL + INa + ICa + IK + ICl)   (optionally + Iapp)
    # :contentReference[oaicite:19]{index=19}
    dVm = (-(IL + INa + ICa + IK + ICl) + Iapp_uAcm2) / p.C

    # Gating variables
    dm = gate_rhs(m, Vm, "m", gate_approx)
    dh = gate_rhs(h, Vm, "h", gate_approx)
    dn = gate_rhs(n, Vm, "n", gate_approx)
    ds = gate_rhs(s, Vm, "s", gate_approx)
    dr = gate_rhs(r, Vm, "r", gate_approx)

    # Calcium dynamics:
    #   dci/dt = -(f * 3 * ICa)/(2000*r*F) - gamma*(ci - crest)
    # :contentReference[oaicite:20]{index=20}
    dci = -(p.f * 3.0 * ICa) / (2000.0 * p.r_cm * p.F) - p.gamma * (ci - p.crest)

    return np.array([dVm, dm, dh, dn, ds, dr, dci], dtype=float)


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    p = Params()

    # Initial conditions (pick reasonable ones; these are placeholders)
    Vm0 = -60.0
    # initialize gates at x_inf(Vm0) for stability
    gate_approx = DEFAULT_GATE_APPROX
    m0 = sigmoid_x_inf(Vm0, gate_approx["m"]["Vhalf"], gate_approx["m"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h"]["Vhalf"], gate_approx["h"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["n"]["Vhalf"], gate_approx["n"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["s"]["Vhalf"], gate_approx["s"]["k"])
    r0 = sigmoid_x_inf(Vm0, gate_approx["r"]["Vhalf"], gate_approx["r"]["k"])
    ci0 = p.crest

    y0 = np.array([Vm0, m0, h0, n0, s0, r0, ci0], dtype=float)

    t_span = (0.0, 2000.0)  # ms
    t_eval = np.linspace(t_span[0], t_span[1], 20001)

    # Example: small constant applied current to perturb
    Iapp = 10  # uA/cm^2 (placeholder)

    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p, gate_approx, Iapp_uAcm2=Iapp),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    # Minimal quick check output
    print("Integration success:", sol.success)
    print("Final Vm (mV):", sol.y[0, -1])
    print("Final ci (mM):", sol.y[6, -1])

    plt.plot(sol.t, sol.y[0],label = 'Vm (mV)')
    plt.plot(sol.t, sol.y[6],label = 'ci (mM)')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.show()