import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import Plotting as cplot
from model.Utilities import *




if __name__ == "__main__":
    p = Params()
    gate_approx = DEFAULT_GATE_APPROX
    sim_type ='SEM'
    Vm0 = -65.0
    m0 = sigmoid_x_inf(Vm0, gate_approx["m_ca"]["Vhalf"], gate_approx["m_ca"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h_ca"]["Vhalf"], gate_approx["h_ca"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["m_k"]["Vhalf"], gate_approx["m_k"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["m_Ca"]["Vhalf"], gate_approx["m_Ca"]["k"])
    ci0 = p.crest
    
    cER0 = p.cER_rest
    y0 = np.array([Vm0, m0, h0, n0, s0, ci0, cER0], dtype=float)

    t_end = 20000  # ms
    t_span = (0.0, t_end)  # ms (give it time to settle into a limit cycle)
    t_eval = np.linspace(*t_span, t_end+1)
    # dt = 0.01

    ts, Y, Currs, cafluxes = simulate_sde_euler_maruyama(y0, (0.0, t_end), p, gate_approx)

    Vm = Y[0]
    ci = Y[5]

    IL, INa, ICa, IK, IKCa, I_TRPM4, I_ANO1 = Currs
    J_mem, Jrel, Jserca, Jlk, Jexit = cafluxes[0], cafluxes[1], cafluxes[2], cafluxes[3], cafluxes[4]
    w = (ci**p.n_KCa) / (ci**p.n_KCa + p.Kd_KCa**p.n_KCa)
    current_dicts = {
        'IL': IL,
        'INa': INa,
        'ICa': ICa,
        'IK': IK,
        'IKCa': IKCa,
        'I_TRPM4': I_TRPM4,
        'I_ANO1': I_ANO1
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
        savepath=Path('./plots/SDE_HHType_cafluxes_with_KCa.png'),
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
    

    
    
