from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scipy.integrate import solve_ivp

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.Utilities import DEFAULT_GATE_APPROX, Params, ode_system, sigmoid_x_inf


app = FastAPI(title="Cancer Ion Channel Model")


class SimulationRequest(BaseModel):
    Kd_KCa: float = Field(default=3.0e-4, ge=1.0e-6, le=5.0e-2)
    v_rel: float = Field(default=0.01, ge=0.0, le=0.1)
    v_serca: float = Field(default=0.02, ge=0.0, le=0.1)
    t_end_ms: float = Field(default=20000.0, ge=100.0, le=100000.0)
    dt_ms: float = Field(default=10.0, ge=1.0, le=100.0)
    max_points: int = Field(default=2000, ge=200, le=5000)


def initial_state(p: Params) -> np.ndarray:
    gate = DEFAULT_GATE_APPROX
    Vm0 = -65.0
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


def run_simulation(request: SimulationRequest) -> dict:
    p = Params()
    p.Kd_KCa = request.Kd_KCa
    p.v_rel = request.v_rel
    p.v_serca = request.v_serca

    t_eval = np.arange(0.0, request.t_end_ms, request.dt_ms)
    if t_eval.size == 0 or t_eval[-1] < request.t_end_ms:
        t_eval = np.append(t_eval, request.t_end_ms)
    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p, DEFAULT_GATE_APPROX),
        t_span=(0.0, request.t_end_ms),
        y0=initial_state(p),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise HTTPException(status_code=422, detail=sol.message)

    stride = max(1, int(np.ceil(sol.t.size / request.max_points)))
    t_s = sol.t[::stride] * 1e-3
    Vm = sol.y[0, ::stride]
    ci = sol.y[5, ::stride]
    w = (ci**p.n_KCa) / (ci**p.n_KCa + p.Kd_KCa**p.n_KCa)
    w_tail = w[-min(200, w.size):]

    return {
        "params": asdict(p),
        "summary": {
            "success": bool(sol.success),
            "points": int(t_s.size),
            "Vm_final": float(Vm[-1]),
            "ci_final": float(ci[-1]),
            "w_tail_min": float(np.min(w_tail)),
            "w_tail_max": float(np.max(w_tail)),
        },
        "series": {
            "t_s": t_s.tolist(),
            "Vm_mV": Vm.tolist(),
            "ci_mM": ci.tolist(),
        },
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/simulate")
def simulate(request: SimulationRequest) -> dict:
    return run_simulation(request)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HH + KCa Web Tuner</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #182026;
      --muted: #68727d;
      --line: #d9dee5;
      --blue: #1f66d1;
      --red: #c83c4a;
      --accent: #16815e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(260px, 340px) 1fr;
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 20px;
    }
    section {
      padding: 20px;
      min-width: 0;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 18px;
      letter-spacing: 0;
    }
    .control {
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
    }
    .control label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
    }
    .control strong {
      color: var(--ink);
      font-weight: 650;
    }
    input[type="range"] { width: 100%; }
    input[type="number"] {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: .62;
      cursor: wait;
    }
    .status {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fafbfc;
      color: var(--muted);
      min-height: 64px;
    }
    .plot {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 16px;
    }
    .plot h2 {
      font-size: 15px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }
    canvas {
      display: block;
      width: 100%;
      height: 320px;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      canvas { height: 260px; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>HH + KCa Web Tuner</h1>
      <div id="controls"></div>
      <div class="control">
        <label><strong>Duration</strong><span id="tEndLabel">20000 ms</span></label>
        <input id="t_end_ms" type="number" min="100" max="100000" step="100" value="20000" />
      </div>
      <button id="runBtn">Run / Update</button>
      <div id="status" class="status">Ready.</div>
    </aside>
    <section>
      <div class="plot">
        <h2>Membrane Voltage</h2>
        <canvas id="vmCanvas"></canvas>
      </div>
      <div class="plot">
        <h2>Cytosolic Calcium</h2>
        <canvas id="ciCanvas"></canvas>
      </div>
    </section>
  </main>
  <script>
    const specs = [
      { id: "Kd_KCa", label: "Kd_KCa", min: 1e-6, max: 5e-2, value: 3e-4, step: 1e-6 },
      { id: "v_rel", label: "v_rel", min: 0, max: 0.1, value: 0.01, step: 0.0005 },
      { id: "v_serca", label: "v_serca", min: 0, max: 0.1, value: 0.02, step: 0.0005 },
    ];

    const controls = document.getElementById("controls");
    const statusBox = document.getElementById("status");
    const runBtn = document.getElementById("runBtn");

    function fmt(value) {
      const abs = Math.abs(value);
      if ((abs > 0 && abs < 1e-3) || abs >= 1000) return Number(value).toExponential(3);
      return Number(value).toPrecision(5);
    }

    for (const spec of specs) {
      const wrap = document.createElement("div");
      wrap.className = "control";
      wrap.innerHTML = `
        <label><strong>${spec.label}</strong><span id="${spec.id}_label">${fmt(spec.value)}</span></label>
        <input id="${spec.id}" type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.value}" />
      `;
      controls.appendChild(wrap);
      const input = wrap.querySelector("input");
      const label = wrap.querySelector("span");
      input.addEventListener("input", () => { label.textContent = fmt(Number(input.value)); });
    }

    function values() {
      return {
        Kd_KCa: Number(document.getElementById("Kd_KCa").value),
        v_rel: Number(document.getElementById("v_rel").value),
        v_serca: Number(document.getElementById("v_serca").value),
        t_end_ms: Number(document.getElementById("t_end_ms").value),
        dt_ms: 10,
        max_points: 2000,
      };
    }

    function draw(canvas, x, y, color, yLabel) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const pad = { left: 58, right: 18, top: 18, bottom: 42 };
      const w = rect.width - pad.left - pad.right;
      const h = rect.height - pad.top - pad.bottom;
      const xmin = x[0], xmax = x[x.length - 1];
      let ymin = Math.min(...y), ymax = Math.max(...y);
      if (ymin === ymax) { ymin -= 1; ymax += 1; }
      const ypad = (ymax - ymin) * 0.08;
      ymin -= ypad; ymax += ypad;

      ctx.strokeStyle = "#d9dee5";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.left, pad.top, w, h);

      ctx.fillStyle = "#68727d";
      ctx.font = "12px system-ui";
      ctx.fillText("Time (s)", pad.left + w / 2 - 24, rect.height - 12);
      ctx.save();
      ctx.translate(14, pad.top + h / 2 + 28);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();

      ctx.fillText(ymax.toPrecision(4), 8, pad.top + 4);
      ctx.fillText(ymin.toPrecision(4), 8, pad.top + h);
      ctx.fillText(xmin.toPrecision(3), pad.left, rect.height - 24);
      ctx.fillText(xmax.toPrecision(3), pad.left + w - 34, rect.height - 24);

      ctx.beginPath();
      for (let i = 0; i < x.length; i++) {
        const px = pad.left + ((x[i] - xmin) / (xmax - xmin || 1)) * w;
        const py = pad.top + h - ((y[i] - ymin) / (ymax - ymin || 1)) * h;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }

    async function run() {
      runBtn.disabled = true;
      statusBox.textContent = "Running simulation...";
      try {
        const response = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(values()),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        const s = data.series;
        draw(document.getElementById("vmCanvas"), s.t_s, s.Vm_mV, "#1f66d1", "Vm (mV)");
        draw(document.getElementById("ciCanvas"), s.t_s, s.ci_mM, "#c83c4a", "ci (mM)");
        const summary = data.summary;
        statusBox.textContent =
          `Success. Vm_final=${summary.Vm_final.toFixed(2)} mV, ` +
          `ci_final=${summary.ci_final.toPrecision(4)} mM, ` +
          `w_tail=[${summary.w_tail_min.toPrecision(3)}, ${summary.w_tail_max.toPrecision(3)}], ` +
          `${summary.points} plotted points.`;
      } catch (error) {
        statusBox.textContent = `Error: ${error.message}`;
      } finally {
        runBtn.disabled = false;
      }
    }

    runBtn.addEventListener("click", run);
    window.addEventListener("resize", run);
    run();
  </script>
</body>
</html>
"""
