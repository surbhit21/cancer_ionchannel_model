# tests/test_ode_smoke.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Utilities import Params, ode_system, DEFAULT_GATE_APPROX, sigmoid_x_inf, trace_diagnostics


def initial_state(p, gate, Vm0=-60.0):
    return np.array(
        [
            Vm0,
            sigmoid_x_inf(Vm0, gate["m_ca"]["Vhalf"], gate["m_ca"]["k"]),
            sigmoid_x_inf(Vm0, gate["h_ca"]["Vhalf"], gate["h_ca"]["k"]),
            sigmoid_x_inf(Vm0, gate["m_k"]["Vhalf"], gate["m_k"]["k"]),
            sigmoid_x_inf(Vm0, gate["m_Ca"]["Vhalf"], gate["m_Ca"]["k"]),
            p.crest,
            p.cER_rest,
        ],
        dtype=float,
    )


def test_rhs_shape_and_finite():
    p = Params()
    gate = DEFAULT_GATE_APPROX

    y0 = initial_state(p, gate)

    dy = ode_system(t_ms=0.0, y=y0, p=p, gate_approx=gate)

    assert dy.shape == y0.shape
    assert np.all(np.isfinite(dy))


def test_short_integration_stays_finite():
    # Import solve_ivp only inside the test so it’s clear this is an integration test.
    from scipy.integrate import solve_ivp

    p = Params()
    p.I_ext = 0.2
    gate = DEFAULT_GATE_APPROX
    y0 = initial_state(p, gate)

    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p, gate),
        t_span=(0.0, 50.0),   # ms
        y0=y0,
        t_eval=np.linspace(0.0, 50.0, 501),
        rtol=1e-6,
        atol=1e-9,
    )

    assert sol.success
    assert sol.y.shape[0] == 7
    assert np.all(np.isfinite(sol.y))


def test_diagnostics_shape_and_finite():
    p = Params()
    gate = DEFAULT_GATE_APPROX
    y0 = initial_state(p, gate)
    t = np.array([0.0, 1.0])
    y = np.column_stack([y0, y0])

    currents, ca_fluxes = trace_diagnostics(t, y, p, gate)

    assert currents.shape == (7, 2)
    assert ca_fluxes.shape == (5, 2)
    assert np.all(np.isfinite(currents))
    assert np.all(np.isfinite(ca_fluxes))
