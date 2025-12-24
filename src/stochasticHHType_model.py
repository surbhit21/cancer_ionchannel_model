import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple
from Utilities import *




if __name__ == "__main__":
    p = Params()
    gate_approx = DEFAULT_GATE_APPROX

    Vm0 = -60.0
    m0 = sigmoid_x_inf(Vm0, gate_approx["m"]["Vhalf"], gate_approx["m"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h"]["Vhalf"], gate_approx["h"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["n"]["Vhalf"], gate_approx["n"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["s"]["Vhalf"], gate_approx["s"]["k"])
    r0 = sigmoid_x_inf(Vm0, gate_approx["r"]["Vhalf"], gate_approx["r"]["k"])
    ci0 = 0.0002
    
    cER0 = p.cER_rest
    y0 = np.array([Vm0, m0, h0, n0, s0, r0, ci0, cER0], dtype=float)

    t_end = 30000  # ms
    t_span = (0.0, t_end)  # ms (give it time to settle into a limit cycle)
    t_eval = np.linspace(*t_span, t_end+1)
    dt = 0.02

    t, Y = simulate_sde_euler_maruyama(y0, (0.0, t_end), dt, p, gate_approx)

    Vm = Y[0]
    ci = Y[6]
# 
    # print("Integration success:", sol.success)
    # print("Initial Vm (mV):", sol.y[0, 0])
    # print("Initial ci (mM):", sol.y[6, 0])
    # print("Final Vm (mV):", sol.y[0, -1])
    # print("Final ci (mM):", sol.y[6, -1])

    
    w = (ci**p.n_KCa) / (ci**p.n_KCa + p.Kd_KCa**p.n_KCa)
    print(w[-2000:].min(), w[-2000:].max())

    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t*1e-3, Vm,c="blue")
    ax[1].plot(t*1e-3, ci,c="red")
    ax[0].set_xlabel('Time (ms)')
    ax[1].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Membrane voltage (Vm in mV)')
    ax[1].set_ylabel('Cytosolic calcium (ci in mM)')
    plt.legend()
    plt.savefig('./HHType_with_KCa.png',dpi=300)
    plt.show()
    
    # t = sol.t
    # Vm = sol.y[0]
    # ci = sol.y[6]

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
    ax1.plot(t*1e-3, Vm_norm, color="blue", label="Vm (normalised)")
    ax1.set_ylabel("Normalised $V_m$", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot Ca (right axis)
    ax2.plot(t*1e-3, ci_norm, color="red", label="Ca (normalised)")
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