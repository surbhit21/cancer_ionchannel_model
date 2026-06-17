# Cancer Ion Channel Model

This repository explores a Hodgkin-Huxley-type ion-channel model for cancer cell electrophysiology. The model couples membrane voltage dynamics, voltage-gated channel gating, cytosolic calcium, ER calcium release/leak, SERCA uptake, and calcium-activated KCa/TRPM4/ANO1 conductances.

The codebase now has three user-facing entry points:

- Deterministic and stochastic simulation scripts in `src/model`
- A PyQt desktop GUI in `src/gui`
- A FastAPI browser app in `src/web`

## Features

- Hodgkin-Huxley-type gating and membrane voltage dynamics
- Cytosolic and ER calcium dynamics
- Calcium-dependent KCa, TRPM4, and ANO1 conductances
- Deterministic ODE integration with `scipy.integrate.solve_ivp`
- Stochastic Euler-Maruyama simulation
- Plot generation with Matplotlib
- Desktop parameter tuning with PyQt5
- Browser-based parameter tuning with FastAPI
- Pytest model and API smoke tests
- GitHub Actions CI for push and pull request checks

## Repository Structure

```text
.
├── .github
│   └── workflows
│       └── ci.yml                 # GitHub Actions test workflow
├── src
│   ├── gui
│   │   ├── __init__.py
│   │   └── GUI_HH.py              # PyQt desktop GUI
│   ├── model
│   │   ├── __init__.py
│   │   ├── HHType_model.py        # Deterministic ODE script
│   │   ├── Plotting.py            # Plotting helpers
│   │   ├── stochasticHHType_model.py
│   │   └── Utilities.py           # Params, currents, ODE/SDE helpers
│   ├── tests
│   │   ├── Utests.py              # Legacy smoke tests
│   │   ├── test_model.py          # Pytest model tests
│   │   └── test_web.py            # Pytest FastAPI tests
│   └── web
│       ├── __init__.py
│       └── app.py                 # FastAPI browser app
├── LICENSE
├── Readme.md
└── requirements.txt
```

Generated plots are written to `plots/` by the model scripts.

## Setup

Clone the repository:

```bash
git clone https://github.com/surbhit21/cancer_ionchannel_model.git
cd cancer_ionchannel_model
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bat
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Simulations

Run the deterministic ODE simulation and generate plots:

```bash
python src/model/HHType_model.py
```

Run the stochastic Euler-Maruyama simulation:

```bash
python src/model/stochasticHHType_model.py
```

## Desktop GUI

Launch the PyQt parameter tuner:

```bash
python src/gui/GUI_HH.py
```

The GUI reruns the ODE model and plots:

- Membrane voltage, `V_m`
- Cytosolic calcium, `c_i`

The current GUI exposes these parameters in `src/gui/GUI_HH.py`:

```python
self.slider_specs = [
    ("Kd_KCa", "Kd_KCa", 5e-3, 5e-2),
    ("v_rel", "v_rel", 0.0, 0.10),
    ("v_serca", "v_serca", 0.0, 0.10),
]
```

## Web App

Launch the FastAPI web app:

```bash
uvicorn src.web.app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The web app provides:

- Browser sliders for `Kd_KCa`, `v_rel`, and `v_serca`
- A `/api/simulate` JSON endpoint
- Canvas plots for `V_m` and `c_i`
- Interactive API docs at `http://127.0.0.1:8000/docs`

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"Kd_KCa": 0.0003, "v_rel": 0.01, "v_serca": 0.02, "t_end_ms": 1000, "dt_ms": 20}'
```

## Tests

Run the pytest suite:

```bash
python -m pytest -q
```

The tests cover:

- 7-variable model state validity
- ODE RHS shape and finite values
- Short deterministic integration
- Diagnostic current/flux traces
- Calcium-dependent activation bounds
- SDE reproducibility with fixed seed
- FastAPI UI and `/api/simulate` endpoint smoke tests

## Continuous Integration

GitHub Actions is configured in `.github/workflows/ci.yml`.

On every push and pull request, CI:

- sets up Python 3.11
- installs `requirements.txt`
- runs `python -m pytest -q`

## Notes

The current integrated model state is:

```text
[Vm, m, h, n, s, ci, cER]
```

Currents and calcium fluxes are computed separately as diagnostics rather than being integrated as state variables.
