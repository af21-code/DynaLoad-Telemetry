# DynaLoad-Telemetry

**High-Frequency (100 Hz) Structural Load, Damper Kinematics & Thermo-Hydraulic Anomaly Detection Platform for Aerospace MRO (ATA Chapter 32) and Formula 1 Vehicle Dynamics**

---

## 1. Executive Summary

`DynaLoad-Telemetry` is a full-stack structural dynamics and predictive health monitoring platform engineered to acquire, differentiate, classify, and audit $100\text{ Hz}$ ($\Delta t = 10\text{ ms}$) suspension and shock-strut telemetry across two safety- and performance-critical domains:

* **Profile A — Aerospace MRO (ATA Chapter 32 Landing Gear):** Processes high-rate aircraft Quick Access Recorder (QAR) and Landing Gear Health Monitoring (LGHM) touchdown telemetry. Evaluates main-gear oleo-pneumatic strut compression, tire spin-up, bank-angle roll coupling, vertical load factor ($n_z$) envelopes, nitrogen ($N_2$) pre-charge leakage, and **EASA CS 25.473 / AMM 05-51-11 Phase 1 & Phase 2 Hard-Landing** inspection triggers.
* **Profile B — Motorsport Formula 1 Trackside (Suspension & Kerb Loads):** Processes trackside ATLAS-equivalent $100\text{ Hz}$ damper potentiometer, pushrod load, and internal gas-reservoir telemetry across high-speed ground-effect straights, heavy braking zones, and sawtooth chicanes. Evaluates **4-quadrant damper velocity histograms (Low/High-Speed Bump & Rebound)**, pushrod buckling margins, hydraulic cavitation, and **FIA Technical Regulation Article 3.5.9** titanium skid-block / plank bottoming strikes.

---

## 2. Unified Mathematical & Physical Core

Despite operating at different mass scales (a $78,000\text{ kg}$ narrow-body transport aircraft vs. a $798\text{ kg}$ Formula 1 car), **an aircraft oleo-pneumatic shock strut and an F1 pushrod-actuated hydropneumatic damper are governed by the exact same non-linear dynamical system.**

Both assemblies are velocity-dependent, polytropically pressurized hydro-mechanical energy absorbers connecting a sprung structural mass $m_s$ (airframe fuselage/wing box or F1 carbon monocoque) to an unsprung wheel assembly $m_u$ via a stroke coordinate $x(t) \in [0, x_{\max}]$:

$$m_s \ddot{z}_s(t) + F_{\text{gas}}\bigl(x(t), P_0, \gamma\bigr) + F_{\text{hyd}}\bigl(\dot{x}(t), c_{\text{LS}}, c_{\text{HS}}\bigr) + F_{\text{seal}}\bigl(\dot{x}(t)\bigr) = F_{\text{ext}}(t)$$

### 2.1 Isomorphism Across Aerospace MRO and Formula 1

| Physical & Mathematical Subsystem | Mathematical Formulation | Profile A: Aerospace MRO (ATA Chapter 32) | Profile B: Formula 1 Trackside (Suspension) |
| :--- | :--- | :--- | :--- |
| **Stroke Kinematics** | $x(t) \in [0, x_{\max}]$ | Main gear oleo chrome piston stroke ($0\text{--}450\text{ mm}$) via LVDT / RVDT | Corner/heave damper travel ($0\text{--}65\text{ mm}$) via linear potentiometer |
| **Numerical Velocity ($\dot{x}$)** | $v[n] = \dfrac{x[n+1] - x[n-1]}{2\,\Delta t}$ | Sink-rate absorption velocity ($+1200\text{ mm/s}$ impact peak) | Kerb-strike & platform rate ($\pm 850\text{ mm/s}$ peak) |
| **Polytropic Gas Spring** | $P(x) = P_0 \left(\dfrac{L_0}{L_0 - x}\right)^\gamma$ | Nitrogen ($N_2$) upper chamber ($P_0 \approx 30\text{ bar} \to 210\text{ bar}$, $\gamma \approx 1.35$) | Piggyback $N_2$ bladder reservoir ($P_0 \approx 16\text{ bar} \to 55\text{ bar}$, $\gamma \approx 1.38$) |
| **Hydraulic Damping** | $F_{\text{hyd}}(\dot{x}) = \begin{cases} c_{\text{LS}}\dot{x} & |\dot{x}| \le v_{\text{knee}} \\ c_{\text{HS}}\dot{x}\,\lvert\dot{x}\rvert^{\alpha-1} & |\dot{x}| > v_{\text{knee}} \end{cases}$ | Turbulent orifice plate & tapered metering pin ($\alpha = 2$, quadratic velocity squared) | Preloaded Belleville shim stack & high-speed blow-off valve ($\alpha \approx 1$, piecewise linear) |
| **Mechanical Bottoming** | $x(t) \ge x_{\text{bottom}} \to k_{\text{stop}} \gg k_{\text{gas}}$ | Metal-to-metal internal stop contact ($x \ge 430\text{ mm}$), risking trunnion/pintle fracture | Bump-rubber saturation & plank ground impact ($x \ge 60\text{ mm}$), risking FIA Art. 3.5.9 disqualification |
| **Degradation Manifold** | $r_P(x) = \dfrac{P_{\text{meas}}(t)}{P_{\text{expected}}(x)} \ll 1$ | Dynamic T-seal $N_2$ blow-by or MIL-PRF-87257 fluid leakage / foaming | Bladder permeability leak or high-frequency hydraulic cavitation (wheel-hop) |

---

### 2.2 Discrete-Time Numerical Differentiation & 4-Regime Classification

At $f_s = 100\text{ Hz}$ ($\Delta t = 0.01\text{ s}$), the kinematics engine (`kinematics_ml_engine/kinematics.py`) computes the instantaneous damper velocity $v[n] = \dot{x}(t_n)$ ($\text{mm/s}$) using second-order central finite differences on interior samples and second-order one-sided differences at boundary frames (with an optional Savitzky-Golay polynomial derivative filter for noisy analog potentiometers):

$$v[n] = \frac{x[n+1] - x[n-1]}{2\,\Delta t} + \mathcal{O}(\Delta t^2), \qquad a_{\text{strut}}[n] = \frac{v[n+1] - v[n-1]}{2\,\Delta t}$$

Each $10\text{ ms}$ frame is classified into five kinematic regimes parameterized by deadband $v_{\text{db}}$ and hydraulic knee velocities $(v_{\text{knee,bump}}, v_{\text{knee,reb}})$:

1. **High-Speed Bump (`HIGH_SPEED_BUMP`):** $v[n] > +v_{\text{knee,bump}}$ — Governed by high-speed orifice metering / shim blow-off. Absorbs kinetic impact energy during aircraft Weight-on-Wheels (WoW) touchdown or F1 sawtooth kerb strikes.
2. **Low-Speed Bump (`LOW_SPEED_BUMP`):** $+v_{\text{db}} < v[n] \le +v_{\text{knee,bump}}$ — Governed by low-speed bleed needles. Controls chassis/airframe pitch dive under braking, aerodynamic squat, and roll entry.
3. **Static / Deadband (`STATIC`):** $|v[n]| \le v_{\text{db}}$ — Quasi-static equilibrium or friction stiction band.
4. **Low-Speed Rebound (`LOW_SPEED_REBOUND`):** $-v_{\text{knee,reb}} \le v[n] < -v_{\text{db}}$ — Controls platform extension recovery after pitch/roll or flare derotation.
5. **High-Speed Rebound (`HIGH_SPEED_REBOUND`):** $v[n] < -v_{\text{knee,reb}}$ — Controls rapid wheel unloading off kerbs or post-impact oleo recoil.

---

### 2.3 Physics-Informed Isolation Forest Anomaly Detection

Traditional univariate threshold alarms fail when an oleo-strut or F1 damper suffers **gas pre-charge loss** or **hydraulic damping cavitation** during an otherwise moderate load event: peak vertical acceleration $n_z$ may remain below hard-landing/overload limits while the strut bottoms out prematurely or oscillates uncontrollably.

`StrutAnomalyDetector` (`kinematics_ml_engine/anomaly_detector.py`) projects each $100\text{ Hz}$ sample into a 6-dimensional physics-informed state space:

$$\mathbf{f}_n = \begin{bmatrix}
\dfrac{x_n}{x_{\max}}, &
\dfrac{|\dot{x}_n|}{v_{\text{knee}}}, &
n_{z,n}, &
\dfrac{P_n}{P_0 \left(\frac{L_0}{L_0 - x_n}\right)^\gamma}, &
\dfrac{|\dot{x}_n|}{v_{\text{knee}}\bigl(|n_{z,n} - 1| + 0.35\bigr)}, &
\sqrt{\dfrac{1}{W}\sum_{k=n-W+1}^{n} \left(\dfrac{\dot{x}_k}{v_{\text{knee}}}\right)^2}
\end{bmatrix}^T$$

An **Isolation Forest** ensemble of randomized binary partition trees isolates off-manifold states with shorter average path lengths $E[h(\mathbf{f}_n)]$, yielding normalized anomaly scores $s(\mathbf{f}_n, \psi) = 2^{-E[h(\mathbf{f}_n)]/c(\psi)}$ and isolating:
* **Polytropic Gas Loss (`ATA32_N2_PRECHARGE_LOSS_OR_SEAL_BLOWBY` / `F1_RESERVOIR_GAS_LOSS_BLADDER_LEAK`):** Depressed pressure ratio $P_n / P_{\text{expected}}(x_n) < r_{P,\min}$.
* **Hydraulic Damping Degradation (`ATA32_HYDRAULIC_ORIFICE_DAMPING_DEGRADATION` / `F1_DAMPER_HYDRAULIC_CAVITATION_SHIM_FATIGUE`):** Elevated specific damping admittance and $250\text{ ms}$ rolling high-frequency velocity RMS.
* **Mechanical Stop Bottoming (`ATA32_OLEO_MECHANICAL_STOP_BOTTOMING` / `F1_SKID_BLOCK_PLANK_BOTTOMING_STRIKE`):** Stroke excursions exceeding $x_{\text{bottom}}$.

---

## 3. Repository Structure

```text
DynaLoad-Telemetry/
├── README.md                               # Engineering documentation & execution guide
├── .github/
│   └── workflows/
│       └── tests.yml                       # CI pipeline (Python Pytest + .NET 8 Streamer E2E)
├── telemetry-streamer-dotnet/              # Module 1: C# .NET 8 100 Hz UDP Telemetry Streamer
│   ├── DynaLoad.TelemetryStreamer.csproj
│   ├── Program.cs
│   ├── Models/
│   │   └── TelemetryPacket.cs              # 7-channel 100 Hz packet schema (JSON & CSV)
│   ├── Simulation/
│   │   ├── ITelemetryGenerator.cs
│   │   ├── AerospaceTouchdownGenerator.cs  # Profile A (ATA 32 QAR Touchdown: Nominal/Overload/Fault)
│   │   └── F1TrackLapGenerator.cs          # Profile B (F1 Track Sector: Nominal/Overload/Fault)
│   └── Networking/
│       └── UdpTelemetryPublisher.cs        # 100 Hz PeriodicTimer UDP broadcaster & file exporter
├── kinematics-ml-engine/                   # Module 2: Python Kinematics, Loads & Isolation Forest Engine
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── kinematics_ml_engine/
│   │   ├── __init__.py
│   │   ├── models.py                       # Profile configs, batches, and report dataclasses
│   │   ├── kinematics.py                   # Central diff dx/dt, Savitzky-Golay, 4-regime histograms
│   │   ├── structural_loads.py             # EASA AMM 05-51-11 & FIA Art. 3.5.9 action report engine
│   │   ├── anomaly_detector.py             # Isolation Forest & polytropic feature extraction
│   │   ├── simulator.py                    # 100 Hz synthetic generator & CSV/JSON loaders
│   │   └── udp_listener.py                 # Live 100 Hz UDP socket capture
│   └── tests/
│       ├── conftest.py
│       ├── test_kinematics.py
│       ├── test_structural_loads.py
│       ├── test_anomaly_detector.py
│       └── test_end_to_end.py
└── dashboard/                              # Module 3: Streamlit Dual-Profile Interactive UI
    ├── app.py
    └── requirements.txt
```

---

## 4. Quick-Start & Usage Instructions

### 4.1 Run the C# .NET 8 100 Hz Telemetry Streamer

Verify all 6 operational combinations (`AerospaceAta32` & `Formula1Trackside` $\times$ `Nominal`, `Overload`, `Fault`):

```bash
dotnet run --project telemetry-streamer-dotnet/DynaLoad.TelemetryStreamer.csproj -- --self-test
```

Stream live $100\text{ Hz}$ UDP packets to `udp://127.0.0.1:5005` and export CSV/JSON artifacts:

```bash
# Stream Profile A (Aerospace ATA 32 Hard-Landing Overload) at 100 Hz
dotnet run --project telemetry-streamer-dotnet/DynaLoad.TelemetryStreamer.csproj -- \
  --profile aero --scenario overload --duration-sec 10 --port 5005 \
  --export-csv /tmp/aero_overload.csv

# Stream Profile B (Formula 1 Damper Gas-Loss & Cavitation Fault) at 100 Hz
dotnet run --project telemetry-streamer-dotnet/DynaLoad.TelemetryStreamer.csproj -- \
  --profile f1 --scenario fault --duration-sec 10 --port 5005 \
  --export-json /tmp/f1_fault.json
```

### 4.2 Run the Python Kinematics & ML Engine Test Suite

```bash
cd kinematics-ml-engine
pip install -r requirements.txt
pytest tests/ -v
```

### 4.3 Launch the Interactive Streamlit Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Use the sidebar selector to switch instantaneously between:
* **`Aerospace MRO Mode (ATA 32 Landing Gear)`**
* **`Formula 1 Trackside Mode (Suspension & Kerb Loads)`**
