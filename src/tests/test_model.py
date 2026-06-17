import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Utilities import (
    DEFAULT_GATE_APPROX,
    Params,
    ode_system,
    sigmoid_x_inf,
    simulate_sde_euler_maruyama,
    trace_diagnostics,
    w_kca,
)


def initial_state(p: Params, gate=DEFAULT_GATE_APPROX, Vm0=-60.0) -> np.ndarray:
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


def test_initial_state_valid():
    p = Params()
    y0 = initial_state(p)

    assert y0.shape == (7,)
    assert np.all(np.isfinite(y0))
    assert np.all((y0[1:5] >= 0.0) & (y0[1:5] <= 1.0))
    assert y0[5] >= 0.0
    assert y0[6] >= 0.0


def test_ode_system_returns_7_finite_derivatives():
    p = Params()
    y0 = initial_state(p)

    dy = ode_system(0.0, y0, p, DEFAULT_GATE_APPROX)

    assert dy.shape == (7,)
    assert np.all(np.isfinite(dy))


def test_short_solve_ivp_integration_stays_finite():
    p = Params()
    y0 = initial_state(p)

    sol = solve_ivp(
        lambda t, y: ode_system(t, y, p, DEFAULT_GATE_APPROX),
        (0.0, 100.0),
        y0,
        t_eval=np.linspace(0.0, 100.0, 101),
        rtol=1e-6,
        atol=1e-9,
    )

    assert sol.success
    assert sol.y.shape == (7, 101)
    assert np.all(np.isfinite(sol.y))


def test_trace_diagnostics_shapes_and_finite_values():
    p = Params()
    y0 = initial_state(p)
    t = np.array([0.0, 1.0, 2.0])
    y = np.column_stack([y0, y0, y0])

    currents, fluxes = trace_diagnostics(t, y, p, DEFAULT_GATE_APPROX)

    assert currents.shape == (7, 3)
    assert fluxes.shape == (5, 3)
    assert np.all(np.isfinite(currents))
    assert np.all(np.isfinite(fluxes))


@pytest.mark.parametrize("ci", [0.0, 1e-6, 1e-4, 1e-3, 1e-2])
def test_ca_dependent_activation_bounded(ci):
    values = w_kca(ci, Params())

    assert len(values) == 3
    assert all(0.0 <= value <= 1.0 for value in values)


def test_sde_reproducible_with_seed():
    p = Params()
    p.dt = 1
    p.seed = 42
    y0 = initial_state(p)

    t1, y1, currents1, fluxes1 = simulate_sde_euler_maruyama(y0, (0.0, 10.0), p, DEFAULT_GATE_APPROX)
    t2, y2, currents2, fluxes2 = simulate_sde_euler_maruyama(y0, (0.0, 10.0), p, DEFAULT_GATE_APPROX)

    assert np.allclose(t1, t2)
    assert np.allclose(y1, y2)
    assert np.allclose(currents1, currents2)
    assert np.allclose(fluxes1, fluxes2)
    assert np.all(np.isfinite(y1))
