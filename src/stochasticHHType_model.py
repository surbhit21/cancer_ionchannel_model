import matplotlib.pyplot as plt
import numpy as np
import Plotting as cplot
from pathlib import Path
from scipy.integrate import solve_ivp
from typing import Dict, Tuple
from Utilities import *




if __name__ == "__main__":
    p = Params()
    gate_approx = DEFAULT_GATE_APPROX
    sim_type ='SEM'
    Vm0 = -65.0
    m0 = sigmoid_x_inf(Vm0, gate_approx["m_ca"]["Vhalf"], gate_approx["m_ca"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h_ca"]["Vhalf"], gate_approx["h_ca"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["m_k"]["Vhalf"], gate_approx["m_k"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["m_Ca"]["Vhalf"], gate_approx["m_Ca"]["k"])
    r0 = sigmoid_x_inf(Vm0, gate_approx["m_Cl"]["Vhalf"], gate_approx["m_Cl"]["k"])
    ci0 = 0.04
    # ci0 = 0.02
    
    cER0 = p.cER_rest
    y0 = np.array([Vm0, m0, h0, n0, s0, r0, ci0, cER0], dtype=float)

    t_end = 20000  # ms
    t_span = (0.0, t_end)  # ms (give it time to settle into a limit cycle)
    t_eval = np.linspace(*t_span, t_end+1)
    # dt = 0.01

    ts, Y, Currs, cafluxes = simulate_sde_euler_maruyama(y0, (0.0, t_end), p, gate_approx)

    Vm = Y[0]
    ci = Y[6]
#   
    IL, INa, ICa, IK, ICl, IKCa = Currs[0], Currs[1], Currs[2], Currs[3], Currs[4], Currs[5]
    J_mem, Jrel, Jserca, Jlk, Jexit = cafluxes[0], cafluxes[1], cafluxes[2], cafluxes[3], cafluxes[4]
    w = (ci**p.n_KCa) / (ci**p.n_KCa + p.Kd_KCa**p.n_KCa)
    breakpoint()
    current_dicts = {
        'IL': IL,
        'INa': INa,
        'ICa': ICa,
        'IK': IK,
        'ICl': ICl,
        'IKCa': IKCa
    }
    
    caflux_dicts = {
        'J_mem': J_mem,
        'Jrel': Jrel,
        'Jserca': Jserca,
        'Jlk': Jlk,
        'Jexit': Jexit
    }
   
    
   
    plt.plot(ts*p.dt*1e-3,w)
    plt.xlabel('Time (s)')
    plt.ylabel('KCa activation variable (w)')
    plt.title('KCa activation variable over time')
    plt.savefig(Path('./plots/SDE_HHType_KCa_activation_variable.png'), dpi=300)
    plt.show()
    
    cplot.plot_vm_and_ci(
        t_ms=ts*p.dt,
        Vm_mV=Vm,
        ci_mM=ci,
        savepath=Path('./plots/SDE_HHType_Vm_ci_with_KCa.png'),
        show=True,
        dpi=300,
    )
    
    cplot.plot_currents(
        t_ms=ts*p.dt,
        currents=current_dicts,
        ylabel=r"Current $(\mu A/cm^2)$",
        savepath=Path('./plots/SDE_HHType_currents_with_KCa.png'),
        show=True,
        dpi=300,
    )

    cplot.plot_currents(
        t_ms=ts*p.dt,
        currents=caflux_dicts,
        ylabel="Calcium Flux (nA)",
        savepath=Path('./plots/SDE_HHType_currents_with_KCa.png'),
        show=True,
        dpi=300,
    )

    cplot.plot_normalised_vm_and_ci(
        t_ms=ts*p.dt,
        Vm_mV=Vm,
        ci_mM=ci,
        savepath=Path('./plots/SDE_HHType_norm_Vm_ci_with_KCa.png'),
        show=True,
        dpi=300,
    )
    

    
    