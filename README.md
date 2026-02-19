# Does It Pay to Play Nice?

**The Effect of Toxic Behavior and Leadership-Style Communication on Win Rates in League of Legends**

Master's Seminar — Applied Sports Research, University of Zurich (UZH)  
Author: Gian Senpinar | February 2026

---

## Overview

This repository contains the research proposal and data collection pipeline for a quantitative study investigating whether toxic in-game behavior reduces win probability and whether leadership-style communication patterns — operationalized through Riot API ping data — independently predict match outcomes in League of Legends.

### Research Questions

1. **RQ1 (Toxicity):** Does higher team-level toxic behavior reduce win probability?
2. **RQ2 (Leadership):** Does leadership-style communication (structured ping usage, objective coordination) independently increase win probability?
3. **RQ3 (Moderation):** Is the leadership effect contingent on toxicity level?

### Theoretical Framework

- **Social Identity Theory** (Tajfel & Turner, 1979)
- **Transformational Leadership Theory** (Bass, 1985)
- **Team Conflict Theory** (De Dreu & Weingart, 2003)

---

## Repository Structure

```
├── proposal/                        # Research proposal
│   ├── research_proposal.tex        # LaTeX source
│   ├── research_proposal.pdf        # Compiled PDF (2 pages)
│   ├── research_proposal.docx       # Word version
│   └── references.bib               # Bibliography (APA)
│
├── Data Collection/                 # Riot API data pipeline
│   ├── collect_matches.py           # Phase 1: fetch ranked matches from EUW
│   ├── build_variables.py           # Phase 2: compute study variables → CSV
│   ├── generate_test_data.py        # Synthetic data for pipeline testing
│   └── requirements.txt             # Python dependencies
│
└── .gitignore
```

---

## Methodology

### Data Source

All data is collected from the **Riot Games Match-V5 API** (`developer.riotgames.com`). The public API provides:

- Per-player statistics (kills, deaths, assists, gold, CS, vision score)
- **Per-player ping counts by type** (onMyWay, command, assistMe, danger, push, getBack, enemyMissing, visionCleared)
- Match timeline events (objective kills, surrender votes)

In-game chat logs are **not** available via the public API. All constructs are operationalized through behavioral proxies.

### Variables

| Variable | Operationalization |
|---|---|
| **Win** (DV) | Binary: 1 = team won, 0 = lost |
| **Toxicity Score** (IV1) | Standardized composite: Feeding Index + Early Surrender + Vision Neglect |
| **Leadership Score** (IV2) | Standardized composite: Coordinating Ping Ratio + Objective Coordination + Vision Leadership |
| **Controls** | Rank tier, champion composition, team KDA, CS/min, patch |

### Analysis

Binary logistic regression with interaction term:

```
log(P(Win) / (1-P(Win))) = β₀ + β₁·Toxicity + β₂·Leadership + β₃·(Toxicity × Leadership) + βc·Controls
```

### Sample

2,000 ranked solo/duo matches from EUW, stratified across Gold and Platinum/Emerald tiers.

---

## Usage

### Prerequisites

- Python 3.11+
- Riot Games API key ([developer.riotgames.com](https://developer.riotgames.com))

### Setup

```bash
cd "Data Collection"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Collection

```bash
export RIOT_API_KEY="RGAPI-your-key-here"
python collect_matches.py       # ~1-2 hours (rate-limited)
python build_variables.py       # produces match_dataset.csv
```

### Pipeline Test (no API key needed)

```bash
python generate_test_data.py    # creates 10 synthetic matches
python build_variables.py       # validates pipeline end-to-end
```

---

## Key References

- Achterbosch, L., Pierce, M., & Simmons, C. (2021). Effects of conflicts on outcomes: The case of multiplayer online games. *Entertainment Computing, 37*, 100399.
- Bass, B. M. (1985). *Leadership and Performance Beyond Expectations*. Free Press.
- De Dreu, C. K. W., & Weingart, L. R. (2003). Task versus relationship conflict, team performance, and team member satisfaction: A meta-analysis. *Journal of Applied Psychology, 88*(4), 741–749.
- Kokkinakis, A. V. et al. (2020). Toxic behaviors in team-based competitive gaming. *CHI Play '20*, ACM.
- Kwak, H., Blackburn, J., & Han, S. (2015). Exploring cyberbullying and other toxic behavior in team competition online games. *CHI '15*, ACM.

---

## License

Academic use only. Data collected under Riot Games Developer API Terms of Service.
