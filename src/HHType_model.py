
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import Plotting as cplot
from scipy.integrate import solve_ivp
from Utilities import *
from Plotting import ensure_dir_exists
if __name__ == "__main__":
    p = Params()
    gate_approx = DEFAULT_GATE_APPROX

    Vm0 = -65.0
    m0 = sigmoid_x_inf(Vm0, gate_approx["m_ca"]["Vhalf"], gate_approx["m_ca"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h_ca"]["Vhalf"], gate_approx["h_ca"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["m_k"]["Vhalf"], gate_approx["m_k"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["m_Ca"]["Vhalf"], gate_approx["m_Ca"]["k"])
    r0 = sigmoid_x_inf(Vm0, gate_approx["m_Cl"]["Vhalf"], gate_approx["m_Cl"]["k"])
    ci0 = 0.04
    
    cER0 = p.cER_rest
    y0 = np.array([Vm0, m0, h0, n0, s0, r0, ci0, cER0, 0., 0., 0., 0., 0., 0.,0.,0.,0.,0.,0.], dtype=float)


    ensure_dir_exists('./plots/')
    Vms = np.arange(-120.0, 30.0, 0.1)
    ad_curves = get_activation_deactivation_curves(Vms, gate_approx)
    labs = {
        'm_ca': r'$m_{Ca,\infty}$',
        'h_ca': r'$h_{Ca,\infty}$',
        'm_k': r'$m_{K,\infty}$',
        'm_Ca': r'$m_{Ca,\infty}$',
        'm_Cl': r'$m_{Cl,\infty}$'
    }
    
    t_end = 100000  # ms
    t_span = (0.0, t_end)  # ms (give it time to settle into a limit cycle)
    t_eval = np.linspace(*t_span, t_end+1)

    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p, gate_approx),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    gating_dicts = {
        'm_ca': sol.y[1],
        'h_ca': sol.y[2],
        'm_k': sol.y[3],
        'm_Ca': sol.y[4],
        'm_Cl': sol.y[5]
    }

    ts = sol.t*p.dt
    Vm = sol.y[0]
    ci = sol.y[6]
    IL, INa, ICa, IK, ICl, IKCa = sol.y[8],sol.y[9],sol.y[10],sol.y[11],sol.y[12],sol.y[13]
    J_mem, Jrel, Jserca, Jlk, Jexit = sol.y[14],sol.y[15],sol.y[16],sol.y[17],sol.y[18]
    w = (sol.y[6]**p.n_KCa) / (sol.y[6]**p.n_KCa + p.Kd_KCa**p.n_KCa)
    current_dicts = {
        'IL': IL,
        'INa': INa,
        'ICa': ICa,
        'IK': IK,
        'ICl': ICl,
        'IKCa': IKCa
    }
    
    caflux_dicts = {
        # 'J_mem': J_mem,
        'Jrel': Jrel,
        'Jserca': Jserca,
        # 'Jlk': Jlk,
        # 'Jexit': Jexit
    }

    cplot.plot_gating_variables(Vms,ad_curves,ylabels = ["gating value","Time constants (ms)"], savepath=Path('./plots/HHType_gate_curves_with_KCa.png'), labs=labs,show=True, dpi=300)
    cplot.plot_currents(ts,gating_dicts,ylabel = "gating value",labs=labs,show=True, dpi=300, savepath=Path('./plots/HHType_gate_timeline_with_KCa.png'))
    # cplot.plot_currents(ts,gating_dicts,ylabel = "Time constants (ms)",labs=labs,show=True, dpi=300, savepath=Path('./plots/HHType_gate_timeconstant_with_KCa.png'))
    
    # breakpoint()
    # breakpoint()
    print("Integration success:", sol.success)
    print("Initial Vm (mV):", sol.y[0, 0])
    print("Initial ci (mM):", sol.y[6, 0])
    print("Final Vm (mV):", sol.y[0, -1])
    print("Final ci (mM):", sol.y[6, -1])

    
    plt.plot(ts*1e-3,w)
    plt.xlabel('Time (s)')
    plt.ylabel('KCa activation variable (w)')
    plt.title('KCa activation variable over time')
    plt.savefig(Path('./plots/HHType_KCa_activation_variable.png'), dpi=300)
    plt.show()
    
    cplot.plot_vm_and_ci(
        t_ms=ts,
        Vm_mV=Vm,
        ci_mM=ci,
        savepath=Path('./plots/HHType_Vm_ci_with_KCa.png'),
        show=True,
        dpi=300,
    )
    
    cplot.plot_currents(
        t_ms=ts,
        currents=current_dicts,
        ylabel=r"Current $(\mu A/cm^2)$",
        savepath=Path('./plots/HHType_currents_with_KCa.png'),
        show=True,
        dpi=300,
    )

    cplot.plot_currents(
        t_ms=ts,
        currents=caflux_dicts,
        ylabel=r"Calcium Flux $(\mu M/ms)$",
        savepath=Path('./plots/HHType_currents_with_KCa.png'),
        show=True,
        dpi=300,
    )

    cplot.plot_normalised_vm_and_ci(
        t_ms=ts,
        Vm_mV=Vm,
        ci_mM=ci,
        savepath=Path('./plots/HHType_norm_Vm_ci_with_KCa.png'),
        show=True,
        dpi=300,
    )
    
    