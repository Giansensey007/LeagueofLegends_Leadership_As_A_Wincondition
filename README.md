# Does It Pay to Play Nice?

**Toxic Behaviour, Coordination Effort, and Win Probability in Ranked League of Legends**

Applied Sports Research Seminar, FS 2026 -- University of Zurich
Author: Gian Senpinar | April 2026

---

## Overview

This repository contains the data pipeline, replication script, and outputs for a seminar thesis investigating whether toxic behaviour and coordination effort independently predict win probability in ranked League of Legends, drawing on Lazear's (1989) tournament theory.

The final thesis is included as `Sport_Research_LoL (7)[82].pdf`.

### Research Questions

1. **RQ1 (Toxicity):** Is higher team-level toxic behaviour independently associated with lower win probability after controlling for player skill, farming efficiency, and early-game state?
2. **RQ2 (Coordination):** Is coordination effort independently associated with higher win probability?
3. **RQ3 (Interaction):** Is the beneficial effect of coordination attenuated in high-toxicity environments?

### Key Findings

- **Coordination** is a robust positive predictor of win probability across all four model specifications and all four regional subsamples (AME ~ +3.6 pp per SD, p < 0.001 in Model C).
- **Toxicity** shows no significant independent association with win probability in parsimonious or close-game specifications.
- **Interaction** is directionally consistent with the attenuation hypothesis but does not reach significance.

---

## Repository Structure

```
Sport_Research_LoL (7)[82].pdf      # Final thesis (17 pages)

Data Collection/
    collect_matches.py              # Phase 1: Riot API data collection (4 regions)
    build_variables.py              # Phase 2: variable construction from raw JSONs
    requirements.txt                # Python dependencies

Analysis/
    analysis.py                     # Complete replication script (Models A-D)
    generate_codebook.py            # Variable codebook generator
    figures/
        fig1_distributions.png      # Score distributions by outcome
        fig2_coefficients.png       # Coefficient forest plot (Models A-C)
        fig3_interaction.png        # Predicted win probability by toxicity level
        fig4_quartiles.png          # Win rates by score quartile
        fig5_regional.png           # Per-region coefficient plot
        fig6_vif.png                # VIF diagnostics
    tables/
        descriptive_stats.xlsx      # 9 sheets: descriptive stats, VIF, model results
        results_summary.txt         # Full regression output (all 4 models)

docs/
    data_collection_flowchart.pdf   # UML activity diagram
    data_collection_flowchart.png
```

---

## Methodology

### Data Source

All data collected from the **Riot Games Match-V5 API**. 19,520 team-level observations from 9,760 ranked matches across EUW, KR, NA, VN. Stratified by rank tier (Gold/Platinum/Emerald) and quarterly time window (Aug 2024 -- Feb 2026).

### Models

| Model | Description | N |
|---|---|---|
| **A (Baseline)** | Full controls: KDA, CS/min, Gold, Damage, Vision, First Blood, Duration + region FE | 19,520 |
| **B (Extended)** | + Early CS@10, Early Gold Advantage, Early Takedowns, Skill Index | 19,520 |
| **C (Parsimonious)** | Low-VIF controls only: KDA, CS/min, Vision, First Blood, Early Gold Adv, Skill Index (**primary**) | 19,520 |
| **D (Close Games)** | Model C on subsample with gold diff < 10,000 | 11,604 |

### Variables

| Variable | Operationalisation |
|---|---|
| **Win** (DV) | Binary: 1 = team won |
| **Toxicity Score** (IV1) | Z-scored composite: Feeding Index + Early Surrender + Vision Neglect |
| **Coordination Score** (IV2) | Z-scored composite: Coordinating Ping Ratio + Objectives + Vision Leadership |
| **Tox x Coord** | Interaction term |

---

## Replication

### Prerequisites

- Python 3.9+
- Dependencies: `pip install pandas numpy scipy matplotlib seaborn openpyxl`

### Run Analysis

```bash
cd Analysis/
python analysis.py
```

This produces all figures (in `figures/`) and tables (in `tables/`) reported in the thesis.

### Data Collection (optional, requires Riot API key)

```bash
cd "Data Collection"
pip install -r requirements.txt
export RIOT_API_KEY="RGAPI-your-key"
python collect_matches.py       # ~1-2 hours per region
python build_variables.py       # produces match_dataset.csv
```

---

## License

Academic use only. Data collected under Riot Games Developer API Terms of Service.
