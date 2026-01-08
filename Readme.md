This is a repository for exploring a Hodgkin–Huxley–type model with calcium dynamics and a calcium-activated potassium (KCa) current to study voltage fluctuations in cancer cells. 

## Features

-  Hodgkin–Huxley gating + calcium dynamics
- Uses `solve_ivp` with configurable tolerances
- Clean separation between model (`Utilities.py`), execution (`HHType_model.py` and `stochasticHHType_model.py`) and GUI (`GUI_HH.py`)


## Repository Structure

```text
.
├── Utilities.py           # Model definitions (Params, ODEs, gating functions)
├── README.md              # This file
├── src
    ├── GUI_HH.py          # GUI file to for parameter tuning
    ├── HHType_model.py    # This file
    ├── Plotting.py        # function to plot 
    ├── stochasticHHType_model.py    # Integration of SDEs
    ├── Utilities.py      # Parameter definition and helper functions
├── plots                 # to save plots
├── LICENSE                # License file
└── requirements.txt                 # to save plots


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
python HHType_model.py 
```


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

The parameters exposed in the GUI are defined in `GUI_HH.py`: 
```
self.slider_specs = [
     ("Kd_KCa", "Kd_KCa", 1e-4, 5e-2),
      ("n_KCa", "n_KCa", 1.0, 8.0),
      ("v_rel", "v_rel", 0.0, 0.1),
      ("v_serca", "v_serca", 0.0, 0.10),
]
```
