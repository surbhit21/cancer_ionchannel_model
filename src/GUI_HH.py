# gui_hh_kca_pyqt.py
import sys
import numpy as np
from scipy.integrate import solve_ivp

from Utilities import *

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QSlider, QGridLayout, QMessageBox, QSizePolicy
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def fmt_value(x: float) -> str:
    # Nice compact formatting
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.4g}"
    return f"{x:.6g}"


class ParamSlider(QWidget):
    """
    A slider that maps an integer slider position -> float value in [vmin, vmax].
    """
    def __init__(self, name: str, vmin: float, vmax: float, initial: float, steps: int = 1000, parent=None):
        super().__init__(parent)
        self.name = name
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.steps = int(steps)

        self.label = QLabel(name)
        self.value_label = QLabel("")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.steps)

        # Layout
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.value_label, 0, 1, alignment=Qt.AlignRight)
        layout.addWidget(self.slider, 1, 0, 1, 2)
        self.setLayout(layout)

        # Init
        self.set_value(initial)
        self.slider.valueChanged.connect(self._update_value_label)
        self._update_value_label()

    def _pos_to_value(self, pos: int) -> float:
        a = pos / self.steps
        return self.vmin + a * (self.vmax - self.vmin)

    def _value_to_pos(self, value: float) -> int:
        if self.vmax == self.vmin:
            return 0
        a = (value - self.vmin) / (self.vmax - self.vmin)
        a = min(1.0, max(0.0, a))
        return int(round(a * self.steps))

    def value(self) -> float:
        return float(self._pos_to_value(self.slider.value()))

    def set_value(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(self._value_to_pos(float(value)))
        self.slider.blockSignals(False)
        self._update_value_label()

    def _update_value_label(self):
        self.value_label.setText(fmt_value(self.value()))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_vm = fig.add_subplot(2, 1, 1)
        self.ax_ci = fig.add_subplot(2, 1, 2, sharex=self.ax_vm)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HH + KCa Parameter Tuner (PyQt)")
        self.resize(1200, 750)

        # Model state
        self.p = Params()
        self.gate_approx = DEFAULT_GATE_APPROX

        # Sim settings
        self.t_end_ms = 30000
        self.dt_ms = 1.0

        # Choose <= 5 Params attributes to expose
        # Format: (attr, label, min, max)
        self.slider_specs = [
            ("Kd_KCa", "Kd_KCa", 1e-4, 5e-2),
            ("n_KCa", "n_KCa", 1.0, 8.0),
            ("cER_rest", "cER_rest", 1e-3, 5e-1),
            # Add up to 2 more (must exist in Params):
            # ("gKCa", "gKCa", 0.0, 50.0),
            # ("tau_Ca", "tau_Ca", 1.0, 1000.0),
        ]

        self.param_sliders = {}

        # UI
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        # Controls panel
        controls = QVBoxLayout()
        main_layout.addLayout(controls, 0)

        box = QGroupBox("Parameters")
        box_layout = QVBoxLayout(box)

        # Build sliders
        for attr, label, vmin, vmax in self.slider_specs:
            if not hasattr(self.p, attr):
                raise AttributeError(
                    f"Params() has no attribute '{attr}'. "
                    f"Edit slider_specs to use valid fields from your Params class."
                )
            initial = float(getattr(self.p, attr))
            ps = ParamSlider(label, vmin, vmax, initial, steps=2000)
            box_layout.addWidget(ps)
            self.param_sliders[attr] = ps

        controls.addWidget(box)

        # Run button + status
        self.run_btn = QPushButton("Run / Update")
        self.run_btn.clicked.connect(self.run_and_plot)
        controls.addWidget(self.run_btn)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)

        controls.addStretch(1)

        # Plot area
        self.canvas = MplCanvas(self)
        main_layout.addWidget(self.canvas, 1)

        # Initial plot
        self.run_and_plot()

    def apply_params(self):
        for attr, slider in self.param_sliders.items():
            setattr(self.p, attr, float(slider.value()))

    def initial_state(self):
        Vm0 = -65.0
        g = self.gate_approx
        m0 = sigmoid_x_inf(Vm0, g["m"]["Vhalf"], g["m"]["k"])
        h0 = sigmoid_x_inf(Vm0, g["h"]["Vhalf"], g["h"]["k"])
        n0 = sigmoid_x_inf(Vm0, g["n"]["Vhalf"], g["n"]["k"])
        s0 = sigmoid_x_inf(Vm0, g["s"]["Vhalf"], g["s"]["k"])
        r0 = sigmoid_x_inf(Vm0, g["r"]["Vhalf"], g["r"]["k"])
        ci0 = 0.002

        cER0 = float(self.p.cER_rest)
        return np.array([Vm0, m0, h0, n0, s0, r0, ci0, cER0], dtype=float)

    def run_and_plot(self):
        self.run_btn.setEnabled(False)
        self.status.setText("Running simulation...")
        QApplication.processEvents()

        try:
            self.apply_params()

            t_end = int(self.t_end_ms)
            t_span = (0.0, float(t_end))
            t_eval = np.arange(0.0, float(t_end) + self.dt_ms, self.dt_ms)

            y0 = self.initial_state()

            sol = solve_ivp(
                fun=lambda t, y: ode_system(t, y, self.p, self.gate_approx),
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
            )

            if not sol.success:
                self.status.setText(f"Integration failed: {sol.message}")
                return

            t = sol.t * 1e-3  # seconds
            Vm = sol.y[0]
            ci = sol.y[6]

            # Plot
            self.canvas.ax_vm.clear()
            self.canvas.ax_ci.clear()

            self.canvas.ax_vm.plot(t, Vm)
            self.canvas.ax_ci.plot(t, ci)

            self.canvas.ax_vm.set_ylabel("Vm (mV)")
            self.canvas.ax_ci.set_ylabel("ci (mM)")
            self.canvas.ax_ci.set_xlabel("Time (s)")

            # status summary
            w = (ci ** self.p.n_KCa) / (ci ** self.p.n_KCa + self.p.Kd_KCa ** self.p.n_KCa)
            w_tail = w[-2000:] if w.size >= 2000 else w
            self.status.setText(
                f"Success. Vm_final={Vm[-1]:.2f} mV, ci_final={ci[-1]:.4g} mM, "
                f"w_tail=[{w_tail.min():.3g}, {w_tail.max():.3g}]"
            )

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.status.setText(f"Error: {e}")
        finally:
            self.run_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
