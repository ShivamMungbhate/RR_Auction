---
# 🏏 SupeRR Selector — IPL Auction Intelligence

**Pressure-tested player intelligence for smarter IPL auction decisions.**

## 🚀 Live Demo

**Explore the deployed dashboard here:**

👉 [Open SupeRR Selector Dashboard](https://rr-auction-analysis-mjy92gl97pdgj5tyruvdjh.streamlit.app/)

SupeRR Selector is a data-driven cricket analytics project developed for **SupeRR Selector Season 2 — Rajasthan Royals × Unstop Hackathon**. It evaluates T20 players through performance under pressure, role-based analysis, and contextual metrics to help identify reliable player profiles for auction strategy.

Rather than relying solely on overall statistics, the project focuses on how players perform in high-pressure match situations and how those insights can support squad-building decisions.

---

## 🚀 Live Demo

**Explore the deployed dashboard here:**

👉 [https://rr-auction-analysis-mjy92gl97pdgj5tyruvdjh.streamlit.app/](https://rr-auction-analysis-mjy92gl97pdgj5tyruvdjh.streamlit.app/)

---


## 🎯 Project Objective

Build an analytical framework to identify players whose performance holds up under pressure and translate those insights into actionable auction intelligence for Rajasthan Royals.

The project aims to answer:

- Which players perform consistently in high-pressure situations?
- Who demonstrates reliability beyond raw averages and strike rates?
- Which player profiles offer value for squad construction?
- How can performance data help identify potential auction targets?

---

## ⚙️ Key Features

- **Pressure-Based Player Evaluation** — Analyse player performance in challenging match contexts rather than relying only on aggregate statistics.
- **Role-Aware Analysis** — Evaluate players according to their cricketing roles and responsibilities.
- **Contextual Performance Metrics** — Account for the situation in which performances occur.
- **Player Comparison** — Explore player profiles and compare their performance indicators.
- **Auction Intelligence** — Translate analytical findings into potential squad-building and recruitment insights.
- **Interactive Dashboard** — Explore results through an interactive Streamlit application.

---

## 🧠 Methodology

The project follows a data-driven pipeline:

1. **Data Preparation** — Clean, process, and organise ball-by-ball and player-related data.
2. **Match Context Identification** — Extract relevant game situations and pressure-related contexts.
3. **Performance Analysis** — Evaluate player contributions using contextual performance indicators.
4. **Player Assessment** — Generate player-level analytical outputs.
5. **Candidate Selection** — Build a player pool for further evaluation.
6. **Dashboard Visualisation** — Present the final analytical outputs through an interactive interface.

---

## 📊 Dashboard

The Streamlit dashboard provides a central interface to explore the final player analysis and auction-oriented insights.

### Dashboard capabilities

- Explore player ratings and analytical outputs
- Compare player profiles
- Examine role-based player information
- Identify potential auction candidates
- Navigate data-driven insights through interactive visualisations

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core analytics and data processing |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Plotly | Interactive visualisations |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Data exploration and experimentation |

---

## 📁 Project Structure

```text
SupeRR-Selector/
│
├── app.py
├── rr.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── .streamlit/
│
├── results_final2/
│   └── Final analytical outputs
│
├── matches.csv
├── playing_eleven.csv
├── competitions.csv
├── competition_team_player_mapping.csv
├── teams.csv
├── provisional_auction_pool.csv
├── ipl2027_unavailable.csv
├── ipl_2026_player_teams.csv
└── ipl_player_team_history.csv
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will open locally in your browser.

---

## 📦 Data & Outputs

The project uses structured cricket datasets and precomputed analytical outputs to power the dashboard.

The `results_final2/` directory contains the final outputs consumed by the application.

The dashboard uses these outputs to display player-level analysis without requiring the entire analytical pipeline to be rerun on every launch.

---

## 🏏 Use Case

The framework is designed to support:

- Identifying players who deliver in pressure situations
- Evaluating potential recruitment options
- Comparing players with similar roles
- Supporting auction shortlist preparation
- Building a balanced squad using contextual performance insights

The analysis is intended to support decision-making, not replace scouting, tactical judgement, or other recruitment considerations.

---

