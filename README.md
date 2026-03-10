# 🌊 QFlood Protectors
### Quantum-Enhanced Flood Risk Prediction & Sensor Placement Optimization

> **Pakistan's First Quantum Computing Hackathon (QCH 2026)**  
> Organized by PIEAS · NILOP · NCP | Supported by [Open Quantum Institute (OQI) at CERN](https://openquantuminstitute.org/) & GESDA  
> 📍 National Centre for Physics, Islamabad | 📅 February 6–8, 2026

---

## 🧭 Overview

Pakistan loses billions annually to catastrophic flooding — yet the two problems that matter most remain unsolved at scale:

1. **Where do you place sensors** to get the maximum early warning coverage across a river network?
2. **How do you predict floods accurately** when historical data is sparse and hydrological dependencies are complex?

Classical approaches hit a hard wall. Placing just **5 sensors across 150 candidate locations** yields **591 million possible combinations** — a search space that breaks greedy search, brute force, and simulated annealing at real-world scale. Classical ML struggles with the inter-dependent, sparse nature of flood data across river networks.

**QFlood Protectors** is a dual-module quantum computing solution that cracks both problems by mapping Pakistan's flood network directly onto quantum physics.

---

## ⚛️ Core Innovation — Ising Hamiltonian Flood Correlation

The central insight of this project is the formulation of the sensor placement problem as an **Ising Hamiltonian** — the same mathematical structure used to model interacting spins in quantum physics.

Each of Punjab's **16 critical river monitoring locations** is represented as a spin variable:

$$s_i \in \{+1, -1\}$$

where **+1 = sensor placed**, **−1 = no sensor** at location *i*.

The **coupling strengths** *J_ij* between every pair of nodes are derived directly from the **Punjab Flood Correlation Matrix** — a data-driven map of how flood danger propagates from one river station to another across the network. High correlation between sites → strong coupling; independent sites → weak coupling.

The cost Hamiltonian is:

$$H = -\sum_{i,j} J_{ij} \, s_i s_j$$

The **ground state of this Hamiltonian corresponds exactly to the optimal sensor placement** — the configuration that maximises correlated flood risk coverage across all nodes simultaneously. This is not a heuristic approximation; it is the physically correct minimum energy solution to the coverage problem.

This formulation is then handed directly to **QAOA** as its cost function, making the quantum advantage principled and measurable rather than superficial.

---

## 🔧 Architecture

```
Punjab Flood Data (16 River Stations)
           │
           ▼
  ┌─────────────────────┐
  │  Flood Correlation  │  ← Historical hydrological records
  │      Matrix         │     rainfall, river levels, flood events
  └────────┬────────────┘
           │  J_ij coupling strengths
           ▼
  ┌─────────────────────┐
  │  Ising Hamiltonian  │  ← Node inter-dependencies as spin interactions
  │   H = -Σ J_ij s_i s_j│    Ground state = optimal placement
  └────────┬────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│ Module 1│  │ Module 2 │
│  QAOA   │  │   QNN    │
└────┬────┘  └────┬─────┘
     │             │
     ▼             ▼
Optimal Sensor   Flood
 Placement     Probability
  Map          Prediction
```

### Module 1 — QAOA Sensor Placement

The Ising Hamiltonian is passed to the **Quantum Approximate Optimization Algorithm (QAOA)** as its cost function. QAOA explores the full 591M-combination search space via parameterised quantum circuits, converging on the placement that:

- ✅ Minimises total flood risk exposure across the network
- ✅ Maximises early warning coverage
- ✅ Respects cost and deployment constraints
- ✅ Captures inter-node dependencies that classical solvers ignore

### Module 2 — QNN Flood Forecasting

A **Quantum Neural Network (QNN)** trained on sparse historical flood records encodes multi-variate input features — rainfall, temperature, humidity, flood risk indices — into quantum feature space. The QNN:

- Predicts flood probability at each sensor location
- Leverages quantum feature encoding for superior generalisation on sparse data
- Is benchmarked against classical neural networks (CNNs, SVMs) to validate quantum advantage

---

## 📊 Key Results

| Metric | Result |
|--------|--------|
| Search space addressed | 591,000,000+ sensor placement combinations |
| River stations modelled | 16 critical Punjab monitoring locations |
| Flood correlation network | Full inter-node propagation matrix constructed |
| Classical → Quantum transformation | Validated end-to-end pipeline |
| QNN vs Classical benchmark | Competitive accuracy on sparse real-world datasets |

---

## 🗂️ Repository Structure

```
QFlood-Protectors/
│
├── 📁 datasets/
│   ├── Flood years.docx
│   ├── Pakistan River Inflow & Outflow (1950-2025).docx
│   ├── pakistan_weather_2000_2024.csv
│   └── punjab_updated.json             # Datasets for Punjab
│
├── 📁 Flood Prediction/
│   ├── Datasets
│   ├── QML VSC 
│   ├── punjab_qml_enhanced.json                     
│   ├── qml_flood_predictor.py
│   └── qml_training_data.json             
│
├── 📁 QUBO/
│   ├── Flood Problem.ipynb        # QAOA circuit for optimal sensor placement
│   └── qaoa_results_complete.json                   # Complete results
│
├── 📁 theory/
│   ├── Flood Problem.ipynb               
│   ├── Flood Problem.pdf                  
│   └── Quantum inspired Flood dynamics     
│
├── 📁 NDMA reports/
│   ├── A Comprehensive Study of Flood Events in Pakistan (1950-2025) NDMA Report.pdf     
│   ├── Assessing Impacts of Flood in Pakistan (2010-2024) NDMA Report.pdf
│   ├── Comprehensive Review of Monsoon 2024.pdf
│   ├── NDMA Annual Report (2007-8).pdf
│   ├── NDMA Annual Report (2009).pdf
│   ├── NDMA Annual Report (2010).pdf
│   ├── NDMA Annual Report (2012).pdf
│   ├── NDMA Annual Report (2013).pdf
│   ├── NDMA Annual Report (2014).pdf
│   ├── NDMA Annual Report (2015).pdf
│   ├── NDMA Annual Report (2016).pdf
│   ├── NDMA Annual Report (2017).pdf
│   ├── NDMA Annual Report (2018).pdf
│   ├── NDMA Annual Report (2019).pdf
│   ├── NDMA Annual Report (2020).pdf
│   ├── NDMA Annual Report (2021).pdf
│   ├── NDMA Annual Report (2022).pdf
│   ├── NDMA Annual Report (2024).pdf
│   └── NDMA Post Annual Report (2024).pdf
│
├── 📁 results/
│   ├── optimal_sensor_results.png          
│   └── qnn_benchmark_results.png            # QNN vs classical comparison plots
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

### Key Dependencies

```
qiskit >= 0.45
qiskit-optimization
pennylane >= 0.35
numpy
pandas
matplotlib
scikit-learn
```

### Run the Pipeline

```bash
# 1. Build the Punjab Flood Correlation Matrix
python ising/correlation_analysis.py

# 2. Construct the Ising Hamiltonian
python ising/hamiltonian.py

# 3. Run QAOA sensor placement optimisation
python qaoa/sensor_placement.py

# 4. Train and evaluate the QNN flood forecaster
python qnn/flood_forecasting.py
```

Or explore interactively via the notebooks in `/notebooks/`.

---

## 🌍 SDG Alignment

This project directly addresses three UN Sustainable Development Goals:

| SDG | Goal | Our Contribution |
|-----|------|-----------------|
| **SDG 13** | Climate Action | Quantum early warning system for flood-prone regions |
| **SDG 11** | Sustainable Cities | Optimal sensor infrastructure for urban flood resilience |
| **SDG 6** | Clean Water & Sanitation | River network monitoring for water safety |

---

## 👥 Team

| Name | Institution |
|------|-------------|
| Wajiha Masood | PIEAS, Islamabad |
| Muhammad Mudassar | GCU, Lahore |
| Laiba Saifullah | QAU, Islamabad |
| Falak Dinar | NILOP |
| Sidra Azam | University of Sargodha |
| M. Haroon Umar | NUST, Islamabad |

**Mentors**
- Dr. Muzamil Shah — QAU, Islamabad  
- Dr. Muhammad Kashif — NYUAD

---

## 🏆 Recognition

🥉 **3rd Place** — Quantum Computing Hackathon Pakistan (QCH 2026)  
Competing against 42 finalists selected from 950+ nationwide applications.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- 🌐 [QCH 2026 Official Website](https://quantum-hackathon-chi.vercel.app/)
- 🤝 [Open Quantum Institute at CERN](https://openquantuminstitute.org/)
- 📸 [Instagram](https://www.instagram.com/quantum_computing_hackathon/)
- 💼 [LinkedIn](https://www.linkedin.com/company/quantum-computing-hackathon/)

---

*Built in 3 days at Pakistan's inaugural Quantum Computing Hackathon. Powered by quantum physics, driven by real-world impact.*
