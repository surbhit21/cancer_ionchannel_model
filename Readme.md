This is a repository for exploring a Hodgkin–Huxley–type model with calcium dynamics and a calcium-activated potassium (KCa) current to study voltage fluctuations in cancer cells. 

## Features

-  Hodgkin–Huxley gating + calcium dynamics
- Uses `solve_ivp` with configurable tolerances
- Clean separation between model code (`src/model`), desktop GUI code (`src/gui`), and web app code (`src/web`)


## Repository Structure

```text
.
├── README.md              # This file
├── src
│   ├── gui
│   │   ├── __init__.py
│   │   └── GUI_HH.py      # GUI for parameter tuning
│   ├── model
│   │   ├── __init__.py
│   │   ├── HHType_model.py              # Deterministic ODE integration
│   │   ├── Plotting.py                  # Plotting helpers
│   │   ├── stochasticHHType_model.py    # Stochastic SDE integration
│   │   └── Utilities.py                 # Params, ODEs, gating functions
│   ├── web
│   │   ├── __init__.py
│   │   └── app.py        # FastAPI browser app
│   └── tests
│       └── Utests.py
├── plots                  # to save plots
├── LICENSE                # License file
└── requirements.txt


```

To generate plots

## 1. clone the repository:

``` 
git clone https://github.com/surbhit21/cancer_ionchannel_model.git
```

## 2. (Recommended) Create a virtual environment

```
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

## 4. Run the ODE integrator:

```
python src/model/HHType_model.py
```

To run the GUI:

```
python src/gui/GUI_HH.py
```

To run the web app:

```
uvicorn src.web.app:app --reload
```

Then open `http://127.0.0.1:8000` in a browser.


## 5. GUI for parameter tuning

This repository also provides an interactive **PyQt-based graphical user interface (GUI)** for exploring a Hodgkin–Huxley–type neuron model with calcium dynamics and a calcium-activated potassium (KCa) current.

The GUI allows you to **adjust up to 5 model parameters using sliders**, re-run the numerical integration, and immediately visualise:
- Membrane voltage (`V_m`)
- Cytosolic calcium concentration (`c_i`)

Plots are embedded using Matplotlib, and the system of ODEs is solved using `scipy.integrate.solve_ivp`.

### GUI Features
---

- PyQt5 GUI with embedded Matplotlib plots
- Up to **5 tunable parameters** exposed as sliders
- Explicit **Run / Update** button (no recomputation while dragging sliders)

---
### Adjusting Model Parameters

The parameters exposed in the GUI are defined in `src/gui/GUI_HH.py`: 
```
self.slider_specs = [
     ("Kd_KCa", "Kd_KCa", 1e-4, 5e-2),
      ("n_KCa", "n_KCa", 1.0, 8.0),
      ("v_rel", "v_rel", 0.0, 0.1),
      ("v_serca", "v_serca", 0.0, 0.10),
]
```

## 6. Web app for browser-based parameter tuning

The FastAPI app in `src/web/app.py` provides:

- A browser interface with sliders for `Kd_KCa`, `v_rel`, and `v_serca`
- A `/api/simulate` JSON endpoint that runs the ODE model
- Canvas plots for membrane voltage (`V_m`) and cytosolic calcium (`c_i`)
- Automatic OpenAPI documentation at `http://127.0.0.1:8000/docs`
