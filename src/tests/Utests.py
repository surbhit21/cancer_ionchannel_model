# tests/test_ode_smoke.py
import numpy as np

# Adjust this import to match your project layout.
# Example: from cancerode.model import Params, ode_system, DEFAULT_GATE_APPROX
from model import Params, ode_system, DEFAULT_GATE_APPROX, sigmoid_x_inf


def test_rhs_shape_and_finite():
    p = Params()
    gate = DEFAULT_GATE_APPROX

    Vm0 = -60.0
    y0 = np.array(
        [
            Vm0,
            sigmoid_x_inf(Vm0, gate["m"]["Vhalf"], gate["m"]["k"]),
            sigmoid_x_inf(Vm0, gate["h"]["Vhalf"], gate["h"]["k"]),
            sigmoid_x_inf(Vm0, gate["n"]["Vhalf"], gate["n"]["k"]),
            sigmoid_x_inf(Vm0, gate["s"]["Vhalf"], gate["s"]["k"]),
            sigmoid_x_inf(Vm0, gate["r"]["Vhalf"], gate["r"]["k"]),
            p.crest,
        ],
        dtype=float,
    )

    dy = ode_system(t_ms=0.0, y=y0, p=p, gate_approx=gate, Iapp_uAcm2=0.0)

    assert dy.shape == y0.shape
    assert np.all(np.isfinite(dy))


def test_short_integration_stays_finite():
    # Import solve_ivp only inside the test so it’s clear this is an integration test.
    from scipy.integrate import solve_ivp

    p = Params()
    gate = DEFAULT_GATE_APPROX
    Vm0 = -60.0
    y0 = np.array(
        [
            Vm0,
            sigmoid_x_inf(Vm0, gate["m"]["Vhalf"], gate["m"]["k"]),
            sigmoid_x_inf(Vm0, gate["h"]["Vhalf"], gate["h"]["k"]),
            sigmoid_x_inf(Vm0, gate["n"]["Vhalf"], gate["n"]["k"]),
            sigmoid_x_inf(Vm0, gate["s"]["Vhalf"], gate["s"]["k"]),
            sigmoid_x_inf(Vm0, gate["r"]["Vhalf"], gate["r"]["k"]),
            p.crest,
        ],
        dtype=float,
    )

    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p, gate, Iapp_uAcm2=0.2),
        t_span=(0.0, 50.0),   # ms
        y0=y0,
        t_eval=np.linspace(0.0, 50.0, 501),
        rtol=1e-6,
        atol=1e-9,
    )

    assert sol.success
    assert sol.y.shape[0] == 7
    assert np.all(np.isfinite(sol.y))
