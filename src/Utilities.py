from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple

@dataclass
class Params:
    # Maximal conductances (mS/cm^2)
    # voltage-gated channels, for Na+, Ca2+, K+, Cl-
    gNa: float = 0.15
    gCa: float = 0.03
    gK:  float = 1.1
    gCl: float = 0.0

    # Ca-activated K, Na and Cl conductance (mS/cm^2)
    gKCa: float = 1.1  # tune this (0.05–2.0 is a reasonable sweep)
    gTRPM4 : float = 0.4    #models TRPM4 channels (typical range 0.05 - 0.4 mS/cm^2)
    gANO1 : float = 1.  #models ANO1 channels (0.2 - 1.5 mS/cm^2)

    # Leak (placeholder)
    gL: float = 0.02
    EL: float = -60.0

    # Reversal potentials (mV)
    ENa: float = 67.0
    EK:  float = -80.0
    ECl: float = -77.0
    ECa: float = 120.0
    ETRPM4 : float = 0.0  # approx reversal for non-selective cation channel

    # Exponents
    p: int = 2
    q: int = 1

    # Membrane capacitance (µF/cm^2)
    C: float = 3.2

    # extenal input
    I_ext: float = 0.00
    
    # Calcium dynamics
    crest: float = 1.5e-5
    gamma: float = 1e-3
    f: float = 1
    r_cm: float = 1e-3
    F: float = 9.6485e4

    # KCa activation parameters (ci in mM)
    Kd_KCa: float = 1.0e-2  # mM (0.2 µM). Try 1e-4–1e-3
    n_KCa: float = 2 # Hill coefficient (2–4 common)

    Kd_TRPM4: float = 1.0e-2  # mM (0.2 µM). Try 1e-4–1e-3
    n_TRPM4: float = 2 # Hill coefficient (2–4 common)   

    Kd_ANO1: float = 1.0e-2  # mM (0.2 µM). Try 1e-4–1e-3
    n_ANO1: float = 2 # Hill coefficient (2–4 common)
    
    
    # parameters for the CICR and SERCA pumps 
    
    rho_er: float = 10.0      # V_ER / V_cyt volume ratio (typical 5–20)
    cER_rest: float = 0.05    # mM, resting store Ca (tune)

    v_rel: float = 0.01   # mM/ms, max CICR release rate (tune)
    K_act: float = 1e-3       # mM, Ca activation for release (0.5 µM)
    n_act: float = 2.0        # Hill for activation

    # K_ER: float = 0.02         # mM, store-dependence (optional saturation)
    # n_ER: float = 2.0         # Hill for store dependence

    v_serca: float = 2e-2     # mM/ms, max SERCA uptake (tune)
    K_serca: float = 3e-4     # mM, half-sat SERCA (~0.3 µM)
    n_serca: float = 2.0      # Hill SERCA

    k_leak: float = 1e-3      # 1/ms, passive leak ER -> cyt (small)
    
    gate_sigma: float = 1e-4
    ca_sigma: float = 1e-5     
    seed : int = 0
    
    dt = 1
    
        
def sigmoid_x_inf(Vm_mV: float, Vhalf: float, k: float) -> float:
    return 1.0 / (1.0 + np.exp(-(Vm_mV - Vhalf) / k))


def vdep_tau_ms(Vm_mV: float, tau_min: float, tau_max: float, Vhalf: float, k: float) -> float:
    s = 1.0 / (1.0 + np.exp((Vm_mV - Vhalf) / k))
    return tau_min + (tau_max - tau_min) * s



def J_release(ci: float, cER: float, p: Params) -> float:
    # Ca-triggered release * store-availability
    act = (ci**p.n_act) / (ci**p.n_act + p.K_act**p.n_act + 1e-30)
    # avail = (cER**p.n_ER) / (cER**p.n_ER + p.K_ER**p.n_ER + 1e-30)
    return p.v_rel * act *(cER-ci)

def J_serca(ci: float, p: Params) -> float:
    return p.v_serca * (ci**p.n_serca) / (ci**p.n_serca + p.K_serca**p.n_serca + 1e-30)

def J_leak(ci: float, cER: float, p: Params) -> float:
    # leak from ER to cyt proportional to gradient
    return p.k_leak * (cER - ci)


GateApprox = Dict[str, Dict[str, float]]
DEFAULT_GATE_APPROX: GateApprox = {
    "m_ca": {"Vhalf": -35.0, "k": 9.0,  "tau_min": 0.1, "tau_max": 2.0,  "Vtau": -40.0, "ktau": 2.0},
    "h_ca": {"Vhalf": -55.0, "k": -9.0, "tau_min": 1.0,  "tau_max": 16.0,  "Vtau": -55.0, "ktau": 15.0},
    "m_k": {"Vhalf": -30.0, "k": 12.0, "tau_min": 1.0,  "tau_max": 20.0,  "Vtau": -35.0, "ktau": 15.0},
    "m_Ca": {"Vhalf": -45.0, "k": 9.0,  "tau_min": 2.0,  "tau_max": 40.0, "Vtau": -30.0, "ktau": 15.0},
    "m_Cl": {"Vhalf": -20.0, "k": 10.0,  "tau_min": 2.0,  "tau_max": 30.0, "Vtau": -25.0, "ktau": 15.0}

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
    return (-(x - x_inf) / tau)


# -----------------------------
# Ca-dependent activation (Hill) of Kca, TRPM4, ANO1 channels
# -----------------------------
def w_kca(ci_mM: float, p: Params) -> float:
    ca_dep_conductances = [0.0,0.0,0.0]
    ci = max(ci_mM, 0.0)
    num = ci ** p.n_KCa
    den = num + (p.Kd_KCa ** p.n_KCa)
    ca_dep_conductances[0] = 0.0 if den == 0.0 else num / den
    
    num = ci ** p.n_TRPM4
    den = num + (p.Kd_TRPM4 ** p.n_TRPM4)
    ca_dep_conductances[1] = 0.0 if den == 0.0 else num / den

    num = ci ** p.n_ANO1
    den = num + (p.Kd_ANO1 ** p.n_ANO1)
    ca_dep_conductances[2] = 0.0 if den == 0.0 else num / den
    return ca_dep_conductances 


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

    # Ca-activated K current
    conductances = w_kca(ci, p)
    IKCa = p.gKCa * conductances[0] * (Vm - p.EK)
    I_TRPM4 = p.gTRPM4 * conductances[1] * (Vm - p.ETRPM4)
    I_ANO1 = p.gANO1 * conductances[2] * (Vm - p.ECl)

    return IL, INa, ICa, IK, ICl, IKCa, I_TRPM4, I_ANO1

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def ode_system(t_ms: float, y: np.ndarray, p: Params, gate_approx: GateApprox) -> np.ndarray:
    Vm, m, h, n, s, r, ci, cER = y[:8]

    IL, INa, ICa, IK, ICl, IKCa, I_TRPM4, I_ANO1 = currents(Vm, m, h, n, s, r, ci, p)

    # Membrane equation: C dVm/dt = -(sum currents)
    dVm = p.dt * (-(IL + INa + ICa + IK + ICl + IKCa + I_TRPM4 + I_ANO1) + p.I_ext) / p.C

    # Gating variables
    dm = p.dt * gate_rhs(m, Vm, "m_ca", gate_approx ) 
    dh = p.dt * gate_rhs(h, Vm, "h_ca", gate_approx)
    dn = p.dt * gate_rhs(n, Vm, "m_k", gate_approx)
    ds = p.dt * gate_rhs(s, Vm, "m_Ca", gate_approx)
    dr = p.dt * gate_rhs(r, Vm, "m_Cl", gate_approx)
    
    if int(t_ms) == 3000:
        print("Vm", Vm, "ICa", ICa, "ci", ci, "wKCa", w_kca(ci, p))

    # membrane -> cytosol influx (your original term)
    J_mem = -(p.f * 3.0 * ICa) / (2000.0 * p.r_cm * p.F)  # mM/ms

    # store fluxes (mM/ms)
    Jrel   = J_release(ci, cER, p)
    Jserca = J_serca(ci, p)
    Jlk    = J_leak(ci, cER, p)
    Jexit = p.gamma * (ci - p.crest)

    # cytosol Ca: influx + (ER->cyt) - (cyt->ER) + leak, plus your extrusion to baseline
    dci = p.dt * (J_mem + (Jrel + Jlk) - Jserca - Jexit)

    # store Ca: opposite sign, scaled by volume ratio (conservation across compartments)
    dcER = p.dt * ((p.rho_er) * (Jserca - (Jrel + Jlk)))
   
    return np.array([dVm, dm, dh, dn, ds, dr, dci, dcER,IL, INa, ICa, IK, ICl, IKCa, I_TRPM4, I_ANO1,J_mem,Jrel,Jserca,Jlk,Jexit], dtype=float)

def ode_deterministic(t_ms: float, y: np.ndarray, p: Params, gate_approx: GateApprox) -> np.ndarray:
    Vm, m, h, n, s, r, ci, cER = y

    IL, INa, ICa, IK, ICl, IKCa, I_TRPM4, I_ANO1 = currents(Vm, m, h, n, s, r, ci, p)
    dVm = (-(IL + INa + ICa + IK + ICl + IKCa + I_TRPM4 + I_ANO1) + p.I_ext) / p.C

    dm =  gate_rhs(m, Vm, "m_ca", gate_approx ) 
    dh = gate_rhs(h, Vm, "h_ca", gate_approx)
    dn = gate_rhs(n, Vm, "m_k", gate_approx)
    ds = gate_rhs(s, Vm, "m_Ca", gate_approx)
    dr = gate_rhs(r, Vm, "m_Cl", gate_approx)

    J_mem = -(p.f * 3.0 * ICa) / (2000.0 * p.r_cm * p.F)
    Jrel   = J_release(ci, cER, p)
    Jserca = J_serca(ci, p)
    Jlk    = J_leak(ci, cER, p)
    Jexit = p.gamma * (ci - p.crest)
    dci  = (J_mem + (Jrel + Jlk) - Jserca - Jexit)
    dcER = ((p.rho_er) * (Jserca - (Jrel + Jlk)))

    return np.array([dVm, dm, dh, dn, ds, dr, dci, dcER], dtype=float),np.array([IL, INa, ICa, IK, ICl, IKCa, I_TRPM4, I_ANO1], dtype=float),np.array([J_mem,Jrel,Jserca,Jlk,Jexit], dtype=float)

def simulate_sde_euler_maruyama(
    y0: np.ndarray,
    t_span: tuple[float, float],
    p: Params,
    gate_approx: GateApprox,
):
    rng = np.random.default_rng(p.seed)

    t0, t1 = t_span
    n_steps = int(np.floor((t1 - t0) / p.dt)) + 1

    t = np.linspace(t0, t0 + p.dt*(n_steps-1), n_steps)
    Y = np.zeros((len(y0), n_steps), dtype=float)
    Currs = np.zeros((6, n_steps), dtype=float)  # to store currents if needed
    ca_fluxes = np.zeros((5, n_steps), dtype=float)  # to store ca fluxes if needed
    Y[:, 0] = y0

    sqrt_dt = np.sqrt(p.dt)

    for k in range(n_steps - 1):
        y = Y[:, k]
        dy,Currs[:, k],ca_fluxes[:,k] = ode_deterministic(t[k], y, p, gate_approx)
        # additive noise only on gates (m,h,n,s,r) = indices 1..5
        dW = rng.standard_normal(5)  # N(0,1)
        dW_store = rng.standard_normal()
        eta = sqrt_dt * dW_store
        dy[1:6] += p.gate_sigma * (dW / sqrt_dt)  # convert to "derivative" form? (see note below)
        # Euler update with proper SDE scaling:
        y_next = y + dy * p.dt
        y_next[1:6] += p.gate_sigma * sqrt_dt * dW  # correct EM increment
        y_next[6] += p.ca_sigma * eta  # correct EM increment for ca in cytosol
        y_next[7] -= (p.rho_er) * p.ca_sigma * eta # correct EM increment for ca in ER
        
        # clamp gating variables to [0,1]
        y_next[1:6] = np.clip(y_next[1:6], 0.0, 1.0)

        # (optional) prevent negative Ca
        y_next[6] = max(y_next[6], 0.0)
        y_next[7] = max(y_next[7], 0.0)

        Y[:, k+1] = y_next

    return t, Y, Currs, ca_fluxes



def get_activation_deactivation_curves(
    V_mV: np.ndarray,
    gate_approx: GateApprox,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return activation/inactivation curves for all gates over given Vm range."""
    curves = {}
    for gate_name in gate_approx.keys():
        x_inf = sigmoid_x_inf(
            V_mV,
            gate_approx[gate_name]["Vhalf"],
            gate_approx[gate_name]["k"],
        )
        taus = vdep_tau_ms(
            V_mV,
            gate_approx[gate_name]["tau_min"],
            gate_approx[gate_name]["tau_max"],
            gate_approx[gate_name]["Vtau"],
            gate_approx[gate_name]["ktau"],
        )
        curves[gate_name] = (x_inf, taus)
        
    return curves
