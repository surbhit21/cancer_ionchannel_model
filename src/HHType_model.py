
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import Plotting as cplot
from scipy.integrate import solve_ivp
from Utilities import *
    
if __name__ == "__main__":
    p = Params()
    gate_approx = DEFAULT_GATE_APPROX

    Vm0 = -65.0
    m0 = sigmoid_x_inf(Vm0, gate_approx["m"]["Vhalf"], gate_approx["m"]["k"])
    h0 = sigmoid_x_inf(Vm0, gate_approx["h"]["Vhalf"], gate_approx["h"]["k"])
    n0 = sigmoid_x_inf(Vm0, gate_approx["n"]["Vhalf"], gate_approx["n"]["k"])
    s0 = sigmoid_x_inf(Vm0, gate_approx["s"]["Vhalf"], gate_approx["s"]["k"])
    r0 = sigmoid_x_inf(Vm0, gate_approx["r"]["Vhalf"], gate_approx["r"]["k"])
    ci0 = 4.1e-5
    
    cER0 = p.cER_rest
    y0 = np.array([Vm0, m0, h0, n0, s0, r0, ci0, cER0, 0., 0., 0., 0., 0., 0.,0.,0.,0.,0.,0.], dtype=float)


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
    # breakpoint()
    print("Integration success:", sol.success)
    print("Initial Vm (mV):", sol.y[0, 0])
    print("Initial ci (mM):", sol.y[6, 0])
    print("Final Vm (mV):", sol.y[0, -1])
    print("Final ci (mM):", sol.y[6, -1])

    
    ts = sol.t
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
   
    
   
    plt.plot(ts,w)
    plt.xlabel('Time (ms)')
    plt.ylabel('KCa activation variable (w)')
    plt.title('KCa activation variable over time')
    plt.savefig('./HHType_KCa_activation_variable.png', dpi=300)
    plt.show()
    
    cplot.plot_vm_and_ci(
        t_ms=ts,
        Vm_mV=Vm,
        ci_mM=ci,
        savepath=Path('./HHType_Vm_ci_with_KCa.png'),
        show=True,
        dpi=300,
    )
    
    cplot.plot_currents(
        t_ms=ts,
        currents=current_dicts,
        ylabel=r"Current $(\mu A/cm^2)$",
        savepath=Path('./HHType_currents_with_KCa.png'),
        show=True,
        dpi=300,
    )

    cplot.plot_currents(
        t_ms=ts,
        currents=caflux_dicts,
        ylabel=r"Calcium Flux $(\mu M/ms)$",
        savepath=Path('./HHType_currents_with_KCa.png'),
        show=True,
        dpi=300,
    )

    cplot.plot_normalised_vm_and_ci(
        t_ms=ts,
        Vm_mV=Vm,
        ci_mM=ci,
        savepath=Path('./HHType_norm_Vm_ci_with_KCa.png'),
        show=True,
        dpi=300,
    )
    
    