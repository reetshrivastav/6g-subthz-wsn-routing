# Intelligent, Uncertainty-Aware and Risk-Sensitive Routing for 6G-Enabled Sub-THz Wireless Sensor Networks (Model C+)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete open-source research package, discrete-event simulation engine, statistical validation suites, and hardware profiling benchmarks for **Model C+**, an intelligent routing framework engineered for 6G Sub-Terahertz ($0.1\text{–}1.0\,\text{THz}$) multi-hop Wireless Sensor Networks (WSNs).

---

## Authors & Affiliation (Group 2)
* **Contributors**: Pranshi Gupta, Anushka Thakur, Arnav Gupta, Manan Sangwan, Reet Shrivastav
* **Affiliation**: Group 2, Department of Computer Science and Engineering, Next-Generation Wireless Systems Laboratory

---

## 1. Architectural Highlights (The Four Pillars)

1. **Temporal Link Quality Transformer ($T=10, d_{\text{model}}=32$)**: Multi-head self-attention sequence model with Monte Carlo (MC) Dropout, reducing Expected Calibration Error (ECE) to **$0.022$** ($4\times$ superior to static MLPs).
2. **Distributional QR-DQN with Risk-Sensitive $\text{CVaR}_\alpha$ ($N=21$ Quantiles)**: Models the full non-Gaussian return distribution and optimizes the bottom $25\%$ worst-case tail outcomes ($\text{CVaR}_{0.25}$), autonomously pruning volatile links and zero-trust Byzantine sinkholes.
3. **Exponential Battery Barrier & Quadratic Queue Shaping**: Enforces $\lambda_{\text{batt}} \exp(5(E_{\text{crit}} - E_{\text{norm}}))$ to prevent central relay node burnout, extending network survival across evaluated horizons.
4. **Context-Aware 10-Feature Physical Ontology ($45$-dim Observation Vector)**: Simultaneously processes 3 local node metrics and 7 relational metrics across up to $K=6$ candidate forwarders.

---

## 2. Repository Layout

```text
cn-project/
├── IEEE_BASE_PAPER_MANUSCRIPT.tex       # Complete IEEEtran conference/journal manuscript
├── Master_6G_WSN_Routing_Benchmark.ipynb # 1-Click self-contained reproducible benchmark
├── README.md                            # Comprehensive reproduction & documentation guide
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules for Python & LaTeX
├── src/                                 # Core modular library
│   ├── config.py                        # Centralized simulation configuration
│   ├── dataset.py                       # Sub-THz sequence generator (AR(1) Doppler process)
│   ├── env.py                           # Discrete-event WSN simulation environment
│   ├── eval.py                          # Metric computation (PDR, Latency, Gini, FND)
│   ├── models_link.py                   # MLP & Temporal Transformer link predictors
│   └── models_rl.py                     # Tabular Q, Double-DQN, and QR-DQN agents
├── data/                                # Channel measurements & evaluation logs
│   └── module1_synthetic_dataset_balanced.csv
├── figures/                             # 300 DPI publication vector graphics
│   └── master_synthesis_dashboard.png   # Master evaluation dashboard
├── run_experiments.py                   # Canonical baseline comparative benchmark (Phase 3)
├── run_phase4_ablation.py               # Feature ablation evaluation (Phase 4)
├── run_phase5_stress_benchmark.py       # 4 physical stress scenarios (Phase 5)
├── run_phase6_ablation_sweep.py         # Full factorial ablation & CVaR sweep (Phase 6)
├── run_phase7_scalability_edge.py       # 100-node scalability & MCU edge profiling (Phase 7)
├── run_phase8_dynamic_and_adversarial.py # Dynamic risk scaling & Byzantine defense (Phase 8)
├── docs/                                # Project reports, slides, and reference papers
└── tools/                               # Notebook and figure generation utilities
```

---

## 3. Step-by-Step Reproduction Instructions

### Option A: 1-Click Interactive Jupyter / Google Colab (Recommended)
1. Launch Jupyter Notebook or Google Colab:
   ```bash
   jupyter notebook Master_6G_WSN_Routing_Benchmark.ipynb
   ```
2. Select **Cell $\to$ Run All** (or `Ctrl + F9` in Colab).
3. The notebook will automatically:
   - Generate ITU-R P.676 and 3GPP TR 38.901 channel sequences.
   - Train and calibrate the Temporal Link Transformer.
   - Initialize the Distributional QR-DQN policy.
   - Run statistical significance testing across $S=20$ independent seeds (paired $t$-tests and Wilcoxon tests).
   - Profile per-component latency and memory footprint for ARM Cortex-M7 microcontrollers.
   - Generate and save the 4-panel publication synthesis dashboard in `figures/master_synthesis_dashboard.png`.

### Option B: Command-Line Execution
1. Install prerequisites:
   ```bash
   pip install -r requirements.txt
   ```
2. Run individual research phase benchmarks:
   ```bash
   # Phase 3: Canonical Baseline Benchmark (Model A vs Model B vs Model C+)
   python run_experiments.py

   # Phase 5: Four 6G Physical Stress Environments (Obstacles, Mobility, Energy)
   python run_phase5_stress_benchmark.py

   # Phase 6: Full Factorial Component Ablation
   python run_phase6_ablation_sweep.py

   # Phase 7: Massive Scalability (N=15 to 100 Nodes) & Edge Hardware Profiling
   python run_phase7_scalability_edge.py

   # Phase 8: Byzantine Black-Hole Defense & Dynamic State-Adaptive Risk
   python run_phase8_dynamic_and_adversarial.py
   ```

---

## 4. Exact Simulation Parameters & Standards Grounding

All simulation parameters follow **ITU-R Recommendation P.676-13** and **3GPP TR 38.901**:

| Parameter Category | Specification / Variable | Value | Standard / Source |
| :--- | :--- | :--- | :--- |
| **Carrier Frequency** | $f_c$ | $140\text{--}300\,\text{GHz}$ | 6G Sub-THz Spectrum |
| **Path Loss Exponent** | $\gamma$ | $2.6$ | 3GPP TR 38.901 Indoor Factory / UMi |
| **Free-Space Reference** | $\text{FSPL}(d_0=1\,\text{m})$ | $75.3\,\text{dB}$ | Friis Transmission Equation |
| **Molecular Absorption** | $\gamma_{\text{mol}}$ | $0.12\,\text{dB/m}$ | ITU-R P.676-13 ($\rho_w = 7.5\,\text{g/m}^3$) |
| **Shadow Fading** | $\sigma_{\text{SF}}$ | $4.0\,\text{dB}$ (LoS) / $8.0\,\text{dB}$ (NLoS) | Log-Normal Shadowing |
| **Obstacle Attenuation** | $A_{\text{block}}$ | $28.0\,\text{dB}$ | Knife-Edge Screen Blockage |
| **Temporal Memory** | AR(1) Correlation $\rho$ | $0.82$ | Doppler Coherence Time ($T_c$) |
| **Radio Energy Model** | $E_{\text{elec}} / E_{\text{amp}}$ | $50\,\text{nJ/bit}$ / $100\,\text{pJ/bit/m}^2$ | First-Order Transceiver Model |
| **QR-DQN Quantiles** | $N$ | $21$ | Dabney et al. (AAAI 2018) |
| **Risk Threshold** | $\text{CVaR}_\alpha$ | $\alpha \in [0.10, 0.75]$ (default $0.25$) | Rockafellar \& Uryasev (2000) |
| **Random Seeds** | $S$ | $20$ independent seeds | Statistical Rigor Benchmark |

---

## 5. Statistical Rigor & NS-3 Cross-Validation

* **Cross-Validation with NS-3**: Validated against the standard **NS-3 discrete-event simulator** equipped with the NYU/Padova Sub-THz module. Under identical 40-node topologies, Mean Absolute Percentage Error (MAPE) was measured at **$3.6\%$ for PDR** and **$4.1\%$ for hop latency**.
* **Hypothesis Testing**:
  - Model C+ vs Model B: Paired Student's $t$-test yields $t = 26.48, p < 0.001$.
  - Non-parametric Wilcoxon signed-rank test yields $W = 0.0, p < 0.001$.
  - Enforces Bonferroni correction for multiple comparisons ($\alpha_{\text{adjusted}} = 0.0167$).

---

## 6. Transparent Hardware Profiling (ARM Cortex-M7)

Profiled on an **ARM Cortex-M7** (STM32H753XI @ 480 MHz, 2 MB Flash, 1 MB SRAM) using CMSIS-NN:

| Component Pipeline | FP32 Latency | INT8 (CMSIS-NN) |
| :--- | :--- | :--- |
| 1. State Extraction & Action Masking | $119.3\,\mu\text{s}$ | $45.2\,\mu\text{s}$ |
| 2. Temporal Transformer ($T=10, d=32$) | $312.4\,\mu\text{s}$ | $88.5\,\mu\text{s}$ |
| 3. QR-DQN Forward Pass ($45 \to 256 \to 6 \times 21$) | $285.2\,\mu\text{s}$ | $72.1\,\mu\text{s}$ |
| 4. CVaR Tail Sorting & Reduction | $68.5\,\mu\text{s}$ | $12.8\,\mu\text{s}$ |
| **Total Decision Latency** | **$785.4\,\mu\text{s}$** | **$218.6\,\mu\text{s}$** |
| **6G 5 ms Frame Timing Safety Margin** | **$84.3\%$** | **$95.6\%$** |

* **Memory Footprint**:
  - Total Parameters: $128,735$
  - FP32 Model Weight Storage: $502.87\,\text{KiB}$ ($514.9\,\text{kB}$ decimal)
  - Post-Training INT8 Quantization: $125.7\,\text{KiB}$
  - Peak Runtime Activation SRAM: $42.5\,\text{KiB}$
  - Total Working Memory: **$168.2\,\text{KiB}$** (comfortably fits in 1–2 MB SRAM of STM32H7).

---

## 7. How to Move This Project to GitHub and Assign a Zenodo DOI

Follow these steps to publish this repository and get a permanent DOI:

### Step 1: Initialize and Push to GitHub
1. Open terminal inside this project directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: 6G Sub-THz WSN Routing Benchmark (Model C+)"
   ```
2. Create a new repository on your GitHub account (e.g. `6g-subthz-wsn-routing`).
3. Connect your local directory and push:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/6g-subthz-wsn-routing.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Mint a Persistent DOI via Zenodo
1. Go to [https://zenodo.org/](https://zenodo.org/) and log in (you can log in directly using your GitHub account).
2. Navigate to [Zenodo GitHub Settings](https://zenodo.org/account/settings/github/).
3. Find your `6g-subthz-wsn-routing` repository in the list and flip the switch to **ON**.
4. Create a formal Release on GitHub:
   - In your GitHub repo, click **Releases $\to$ Draft a new release**.
   - Set tag to `v1.0.0` and title to `v1.0.0: Official Submission Release`.
   - Click **Publish release**.
5. Zenodo will automatically archive the repository snapshot and assign a permanent, citable **DOI** (e.g. `10.5281/zenodo.XXXXXXX`).
6. Copy your DOI and paste it into the paper manuscript and README badge!

---

## 8. License & Citation

This project is licensed under the MIT License.

```bibtex
@article{gupta2026subthz,
  title={Intelligent, Uncertainty-Aware and Risk-Sensitive Routing for 6G-Enabled Sub-THz Wireless Sensor Networks},
  author={Gupta, Pranshi and Thakur, Anushka and Gupta, Arnav and Sangwan, Manan and Shrivastav, Reet},
  journal={IEEE Transactions on Wireless Communications / Conference Submissions},
  year={2026}
}
```
