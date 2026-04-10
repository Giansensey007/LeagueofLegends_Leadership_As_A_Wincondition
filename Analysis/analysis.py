#!/usr/bin/env python3
"""
Does It Pay to Play Nice?
Toxic Behaviour, Coordination Effort, and Win Probability in Ranked League of Legends
Gian Senpinar · Applied Sports Research Seminar, FS2026 · University of Zurich

COMPLETE REPLICATION SCRIPT
============================
Produces all tables and figures reported in the thesis:
  - Figures 1-6 (PNG) in figures/
  - results_summary.txt in tables/
  - descriptive_stats.xlsx (9 sheets) in tables/

Usage:
    cd Analysis/
    python analysis.py

Requirements:
    pip install pandas numpy scipy matplotlib seaborn openpyxl

Python 3.9+
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — relative paths (run from Analysis/ directory)
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

_BASE      = Path(__file__).resolve().parent
DATA_PATH  = str(_BASE.parent / "Data Collection" / "match_dataset.csv")
FIG_DIR    = str(_BASE / "figures")
TBL_DIR    = str(_BASE / "tables")
# ─────────────────────────────────────────────────────────────────────────────

import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH, keep_default_na=False)
df["region"] = df["region"].replace("", np.nan)

print("─" * 60)
print("DATASET LOADED")
print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Win rate       : {df['win'].mean():.1%}")
print(f"  Region counts  : {df['region'].value_counts().to_dict()}")
print("─" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
CONT_COLS = [
    "toxicity_score", "leadership_score", "avg_skill_index",
    "team_avg_kda", "team_avg_cs_min", "team_gold", "team_damage",
    "team_vision_score", "game_duration_min",
    "team_early_cs_10", "team_early_gold_adv", "team_early_takedowns",
]
for c in CONT_COLS:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

df["tox_x_coord_z"] = df["toxicity_score_z"] * df["leadership_score_z"]
df["early_win"]     = (df["team_early_gold_adv"] > 0).astype(float)

for r in ["KR", "NA", "VN"]:
    df[f"d_{r}"] = (df["region"] == r).astype(float)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  LOGISTIC REGRESSION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def logit_fit(X: np.ndarray, y: np.ndarray, names=None) -> dict:
    n, k = X.shape

    def neg_ll(b):
        mu = expit(np.clip(X @ b, -500, 500))
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    def grad(b):
        return -(X.T @ (y - expit(np.clip(X @ b, -500, 500))))

    def hess(b):
        mu = expit(np.clip(X @ b, -500, 500))
        return (X * (mu * (1 - mu))[:, None]).T @ X

    res  = minimize(neg_ll, np.zeros(k), jac=grad, method="BFGS",
                    options={"maxiter": 600, "gtol": 1e-7})
    beta = res.x
    mu   = np.clip(expit(X @ beta), 1e-15, 1 - 1e-15)
    ll   = -neg_ll(beta)

    H    = hess(beta)
    try:    cov = np.linalg.inv(H)
    except: cov = np.linalg.pinv(H)
    se   = np.sqrt(np.abs(np.diag(cov)))
    z    = beta / se
    pval = 2 * stats.norm.sf(np.abs(z))

    p0       = y.mean()
    ll_null  = n * (p0 * np.log(p0 + 1e-15) + (1 - p0) * np.log(1 - p0 + 1e-15))
    mcfadden = 1 - ll / ll_null
    acc      = float(np.mean((mu >= 0.5) == y.astype(bool)))
    ame      = np.mean(mu * (1 - mu)) * beta
    OR       = np.exp(beta)
    ci_lo    = np.exp(beta - 1.96 * se)
    ci_hi    = np.exp(beta + 1.96 * se)

    return dict(beta=beta, se=se, z=z, pval=pval, ll=ll,
                mcfadden=mcfadden, acc=acc, ame=ame, n=int(n),
                mu=mu, OR=OR, ci_lo=ci_lo, ci_hi=ci_hi,
                names=names or [f"x{i}" for i in range(k)])


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def print_model(label: str, m: dict) -> str:
    lines = []
    sep = "─" * 80
    lines.append(f"\n{sep}")
    lines.append(f"  {label}")
    lines.append(f"  McFadden R² = {m['mcfadden']:.4f}   Accuracy = {m['acc']:.3f}   "
                 f"Log-L = {m['ll']:.1f}   N = {m['n']:,}")
    lines.append(sep)
    hdr = f"  {'Variable':<26} {'β':>8} {'SE':>7} {'z':>8} {'p-value':>9} {'sig':4} {'OR':>8} {'AME':>8}"
    lines.append(hdr)
    lines.append("  " + "-" * 76)
    for nm, b, se, z, p, OR, ame in zip(
            m["names"], m["beta"], m["se"], m["z"], m["pval"], m["OR"], m["ame"]):
        lines.append(
            f"  {nm:<26} {b:8.4f} {se:7.4f} {z:8.2f} {p:9.4f} {sig_stars(p):4s} "
            f"{OR:8.4f} {ame:8.4f}"
        )
    block = "\n".join(lines)
    print(block)
    return block


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DEFINE CONTROL SETS
# ═════════════════════════════════════════════════════════════════════════════
y = df["win"].values.astype(float)

BASE_CTRL = [
    df["team_avg_kda_z"], df["team_avg_cs_min_z"],
    df["team_gold_z"], df["team_damage_z"],
    df["team_vision_score_z"], df["first_blood"].astype(float),
    df["game_duration_min_z"],
    df["d_KR"], df["d_NA"], df["d_VN"],
]
EARLY_CTRL = [
    df["team_early_cs_10_z"], df["team_early_gold_adv_z"],
    df["team_early_takedowns_z"], df["avg_skill_index_z"],
]
PARS_CTRL = [
    df["team_avg_kda_z"], df["team_avg_cs_min_z"],
    df["team_vision_score_z"], df["first_blood"].astype(float),
    df["team_early_gold_adv_z"], df["avg_skill_index_z"],
    df["d_KR"], df["d_NA"], df["d_VN"],
]

IV_COLS = [np.ones(len(df)), df["toxicity_score_z"],
           df["leadership_score_z"], df["tox_x_coord_z"]]
IV_NAMES = ["Intercept", "Toxicity", "Coordination", "Tox×Coord"]

NAME_A = IV_NAMES + ["KDA","CS/min","Gold","Damage","Vision Score",
                     "First Blood","Game Duration","d_KR","d_NA","d_VN"]
NAME_B = NAME_A + ["Early CS@10","Early Gold Adv","Early Takedowns","Skill Index"]
NAME_C = IV_NAMES + ["KDA","CS/min","Vision Score","First Blood",
                     "Early Gold Adv","Skill Index","d_KR","d_NA","d_VN"]


# ═════════════════════════════════════════════════════════════════════════════
# 5.  ESTIMATE MODELS
# ═════════════════════════════════════════════════════════════════════════════
print("\nFitting Model A (Baseline full controls)…")
XA = np.column_stack(IV_COLS + BASE_CTRL)
mA = logit_fit(XA, y, NAME_A)

print("Fitting Model B (Extended: +early-game +skill)…")
XB = np.column_stack(IV_COLS + BASE_CTRL + EARLY_CTRL)
mB = logit_fit(XB, y, NAME_B)

print("Fitting Model C (Parsimonious, low-VIF) — PRIMARY MODEL…")
XC = np.column_stack(IV_COLS + PARS_CTRL)
mC = logit_fit(XC, y, NAME_C)

print("Fitting Model D (Close games: gold diff < 10,000)…")
match_gold = df.groupby("match_id")["team_gold"].agg(["min","max"])
match_gold["gold_diff"] = match_gold["max"] - match_gold["min"]
df_close  = df.merge(match_gold[["gold_diff"]], on="match_id")
df_close  = df_close[df_close["gold_diff"] < 10000].reset_index(drop=True)
y_close   = df_close["win"].values.astype(float)
XD = np.column_stack([
    np.ones(len(df_close)),
    df_close["toxicity_score_z"], df_close["leadership_score_z"], df_close["tox_x_coord_z"],
    df_close["team_avg_kda_z"], df_close["team_avg_cs_min_z"],
    df_close["team_vision_score_z"], df_close["first_blood"].astype(float),
    df_close["team_early_gold_adv_z"], df_close["avg_skill_index_z"],
    (df_close["region"]=="KR").astype(float),
    (df_close["region"]=="NA").astype(float),
    (df_close["region"]=="VN").astype(float),
])
mD = logit_fit(XD, y_close, NAME_C)
print(f"  Close-games sample: N={len(df_close):,} ({100*len(df_close)/len(df):.1f}% of full sample)")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  VIF CALCULATION
# ═════════════════════════════════════════════════════════════════════════════
def calc_vif(X_df: pd.DataFrame) -> dict:
    vifs = {}
    cols = list(X_df.columns)
    X_arr = X_df.values.astype(float)
    for i, c in enumerate(cols):
        y_  = X_arr[:, i]
        X_  = np.column_stack([np.ones(len(y_)),
                                X_arr[:, [j for j in range(len(cols)) if j != i]]])
        try:
            coef = np.linalg.lstsq(X_, y_, rcond=None)[0]
            ss_res = np.sum((y_ - X_ @ coef) ** 2)
            ss_tot = np.sum((y_ - y_.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vifs[c] = 1 / (1 - r2) if r2 < 0.9999 else 1e9
        except Exception:
            vifs[c] = np.nan
    return vifs

VIF_COLS   = ["toxicity_score_z","leadership_score_z","tox_x_coord_z",
              "team_avg_kda_z","team_avg_cs_min_z","team_gold_z","team_damage_z",
              "team_vision_score_z","game_duration_min_z",
              "team_early_cs_10_z","team_early_gold_adv_z","team_early_takedowns_z",
              "avg_skill_index_z"]
VIF_LABELS = ["Toxicity","Coordination","Tox×Coord","KDA","CS/min","Gold",
              "Damage","Vision Score","Game Duration",
              "Early CS@10","Early Gold Adv","Early Takedowns","Skill Index"]
vif_df = df[VIF_COLS].rename(columns=dict(zip(VIF_COLS, VIF_LABELS)))
vifs   = calc_vif(vif_df)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PER-REGION ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════════════
region_results = {}
for reg in ["EUW", "KR", "NA", "VN"]:
    sub = df[df["region"] == reg].reset_index(drop=True)
    y_r = sub["win"].values.astype(float)
    X_r = np.column_stack([
        np.ones(len(sub)),
        sub["toxicity_score_z"], sub["leadership_score_z"], sub["tox_x_coord_z"],
        sub["team_avg_kda_z"], sub["team_avg_cs_min_z"], sub["team_vision_score_z"],
        sub["first_blood"].astype(float), sub["team_early_gold_adv_z"], sub["avg_skill_index_z"],
    ])
    nm_r = ["Intercept","Toxicity","Coordination","Tox×Coord",
            "KDA","CS/min","Vision","First Blood","Early Gold Adv","Skill Index"]
    try:
        region_results[reg] = logit_fit(X_r, y_r, nm_r)
    except Exception as e:
        print(f"  {reg}: model failed — {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 8.  SAVE RESULTS SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
all_output_lines = []
all_output_lines.append("=" * 80)
all_output_lines.append("RESULTS SUMMARY — Does It Pay to Play Nice?")
all_output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
all_output_lines.append("=" * 80)

for label, m in [("MODEL A — Baseline (full controls)", mA),
                  ("MODEL B — Extended (+early-game +skill)", mB),
                  ("MODEL C — Parsimonious / low-VIF  [PRIMARY]", mC),
                  ("MODEL D — Close-Games Robustness", mD)]:
    block = print_model(label, m)
    all_output_lines.append(block)

all_output_lines.append("\n" + "─" * 80)
all_output_lines.append("  MULTICOLLINEARITY — Variance Inflation Factors")
all_output_lines.append("─" * 80)
all_output_lines.append(f"  {'Variable':<22} {'VIF':>8}   {'Assessment'}")
all_output_lines.append("  " + "-" * 55)
for nm, v in vifs.items():
    flag = "  ⚠ HIGH (excluded from Model C)" if v > 10 else \
           "  ⚠ Elevated (excluded from C)" if v > 5 else "  ✓ Acceptable"
    all_output_lines.append(f"  {nm:<22} {v:8.2f}   {flag}")

all_output_lines.append("\n" + "─" * 80)
all_output_lines.append("  PER-REGION ROBUSTNESS (Model C specification)")
all_output_lines.append("─" * 80)
hdr2 = f"  {'Region':<6} {'N':>6}  {'McF.R²':>8}  {'Tox β':>8} {'(p)':>8}  {'Coord β':>8} {'(p)':>8}  {'Inter. β':>9} {'(p)':>8}"
all_output_lines.append(hdr2)
all_output_lines.append("  " + "-" * 80)
for reg in ["EUW","KR","NA","VN"]:
    m = region_results[reg]
    ti = m["names"].index("Toxicity"); ci = m["names"].index("Coordination"); xi = m["names"].index("Tox×Coord")
    all_output_lines.append(
        f"  {reg:<6} {m['n']:>6}  {m['mcfadden']:>8.3f}  "
        f"{m['beta'][ti]:>8.3f} {m['pval'][ti]:>8.4f}  "
        f"{m['beta'][ci]:>8.3f} {m['pval'][ci]:>8.4f}  "
        f"{m['beta'][xi]:>9.3f} {m['pval'][xi]:>8.4f}"
    )

full_output = "\n".join(all_output_lines)
txt_path = os.path.join(TBL_DIR, "results_summary.txt")
with open(txt_path, "w", encoding="utf-8") as fh:
    fh.write(full_output)
print(f"\nResults saved → {txt_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  EXCEL OUTPUT — DESCRIPTIVE STATS + TABLES
# ═════════════════════════════════════════════════════════════════════════════
xl_path = os.path.join(TBL_DIR, "descriptive_stats.xlsx")
with pd.ExcelWriter(xl_path, engine="openpyxl") as xw:

    desc_cols = [
        "toxicity_score", "leadership_score", "avg_skill_index",
        "team_avg_kda", "team_avg_cs_min", "team_gold", "team_damage",
        "team_vision_score", "game_duration_min",
        "team_early_cs_10", "team_early_gold_adv", "team_early_takedowns",
        "first_blood", "win",
    ]
    desc_labels = [
        "Toxicity Score", "Coordination Score", "Avg Skill Index",
        "Team Avg KDA", "Team Avg CS/min", "Team Total Gold", "Team Damage",
        "Team Vision Score", "Game Duration (min)",
        "Early CS @ 10", "Early Gold Advantage", "Early Takedowns",
        "First Blood (0/1)", "Win (0/1)",
    ]
    desc = df[desc_cols].describe(percentiles=[.25,.5,.75]).T
    desc.index = desc_labels
    skew  = df[desc_cols].skew();  skew.index = desc_labels
    kurt  = df[desc_cols].kurt();  kurt.index = desc_labels
    desc["skewness"] = skew; desc["kurtosis"] = kurt
    desc = desc.round(4)
    desc.to_excel(xw, sheet_name="Descriptive Statistics")

    wl = df.groupby("win")[desc_cols].mean().T
    wl.index = desc_labels
    wl.columns = ["Loss (win=0)", "Win (win=1)"]
    wl["Difference"] = wl["Win (win=1)"] - wl["Loss (win=0)"]
    wl = wl.round(4)
    wl.to_excel(xw, sheet_name="Means by Outcome")

    reg_grp = df.groupby("region")[desc_cols + ["win"]].agg(["mean","std","count"])
    reg_grp.columns = ["_".join(c) for c in reg_grp.columns]
    reg_grp = reg_grp.round(4)
    reg_grp.to_excel(xw, sheet_name="Region Breakdown")

    reg_win = df.groupby(["region","win"])[["toxicity_score","leadership_score","avg_skill_index","team_avg_cs_min"]].mean().round(4)
    reg_win.to_excel(xw, sheet_name="Region x Outcome Means")

    vif_out = pd.DataFrame({
        "Variable": list(vifs.keys()),
        "VIF": [round(v,3) for v in vifs.values()],
        "Assessment": ["High — excluded from Model C" if v>10 else
                       "Elevated — excluded from Model C" if v>5 else
                       "Acceptable (< 5)" for v in vifs.values()]
    })
    vif_out.to_excel(xw, sheet_name="VIF Diagnostics", index=False)

    reg_res = pd.DataFrame({
        "Variable":    mC["names"],
        "Beta":        mC["beta"].round(4),
        "SE":          mC["se"].round(4),
        "z-statistic": mC["z"].round(3),
        "p-value":     mC["pval"].round(4),
        "Significance": [sig_stars(p) for p in mC["pval"]],
        "Odds Ratio":  mC["OR"].round(4),
        "OR 95% CI Lo": mC["ci_lo"].round(4),
        "OR 95% CI Hi": mC["ci_hi"].round(4),
        "AME":         mC["ame"].round(4),
    })
    reg_res.to_excel(xw, sheet_name="Model C Coefficients", index=False)

    key_vars = ["Toxicity","Coordination","Tox×Coord"]
    rows = []
    for v in key_vars:
        row = {"Variable": v}
        for label, m in [("A_Beta",mA),("B_Beta",mB),("C_Beta",mC),("D_Beta",mD)]:
            if v in m["names"]:
                i = m["names"].index(v)
                pfx = label.split("_")[0]
                row[f"{pfx}_Beta"] = round(m["beta"][i],4)
                row[f"{pfx}_SE"]   = round(m["se"][i],4)
                row[f"{pfx}_p"]    = round(m["pval"][i],4)
                row[f"{pfx}_OR"]   = round(m["OR"][i],4)
            else:
                for sfx in ["Beta","SE","p","OR"]: row[f"{label.split('_')[0]}_{sfx}"] = None
        rows.append(row)
    for pfx_label, m in [("A",mA),("B",mB),("C",mC),("D",mD)]:
        pass
    fit_row = {"Variable": "McFadden R²"}
    for pfx, m in [("A",mA),("B",mB),("C",mC),("D",mD)]:
        fit_row[f"{pfx}_Beta"] = round(m["mcfadden"],4)
    rows.append(fit_row)
    acc_row = {"Variable": "Accuracy"}
    for pfx, m in [("A",mA),("B",mB),("C",mC),("D",mD)]:
        acc_row[f"{pfx}_Beta"] = round(m["acc"],4)
    rows.append(acc_row)
    n_row = {"Variable": "N"}
    for pfx, m in [("A",mA),("B",mB),("C",mC),("D",mD)]:
        n_row[f"{pfx}_Beta"] = m["n"]
    rows.append(n_row)
    pd.DataFrame(rows).to_excel(xw, sheet_name="Model Comparison", index=False)

    rr_rows = []
    for reg in ["EUW","KR","NA","VN"]:
        m = region_results[reg]
        for v in ["Toxicity","Coordination","Tox×Coord"]:
            i = m["names"].index(v)
            rr_rows.append({"Region":reg,"Variable":v,
                            "Beta":round(m["beta"][i],4),"SE":round(m["se"][i],4),
                            "z":round(m["z"][i],3),"p-value":round(m["pval"][i],4),
                            "Sig":sig_stars(m["pval"][i]),"OR":round(m["OR"][i],4),
                            "McFadden R2":round(m["mcfadden"],4),"N":m["n"]})
    pd.DataFrame(rr_rows).to_excel(xw, sheet_name="Regional Robustness", index=False)

    qrows = []
    for col, lbl in [("toxicity_score","Toxicity"),("leadership_score","Coordination")]:
        q = pd.qcut(df[col], 4, labels=["Q1 Low","Q2","Q3","Q4 High"])
        grp = df.groupby(q, observed=True)["win"].agg(["mean","count"]).reset_index()
        grp.columns = ["Quartile","Win Rate","N"]
        grp.insert(0,"Variable",lbl)
        qrows.append(grp)
    pd.concat(qrows).round(4).to_excel(xw, sheet_name="Win Rates by Quartile", index=False)

print(f"Excel tables saved → {xl_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 10.  FIGURES
# ═════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})
CLR_WIN  = "#2166AC"
CLR_LOSS = "#D6604D"

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {path}")


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle("Figure 1: Behavioural Score and Skill Distributions by Match Outcome",
             fontsize=12, fontweight="bold")
for ax, (col, lbl) in zip(axes, [
        ("toxicity_score", "Toxicity Score"),
        ("leadership_score", "Coordination Score"),
        ("avg_skill_index", "Skill Index")]):
    for v, clr, lab in [(0, CLR_LOSS, "Loss"), (1, CLR_WIN, "Win")]:
        ax.hist(df[df["win"] == v][col], bins=45, alpha=0.55,
                color=clr, label=lab, density=True)
    ax.set_xlabel(lbl); ax.set_ylabel("Density"); ax.legend(fontsize=8)
plt.tight_layout()
savefig("fig1_distributions.png")


fig, ax = plt.subplots(figsize=(9, 5.5))
KEY_VARS = ["Toxicity", "Coordination", "Tox×Coord"]
x_pos = np.arange(len(KEY_VARS)); W = 0.26
for i, (m, clr, lbl) in enumerate(zip(
        [mA, mB, mC], ["#1a9641","#0571b0","#ca0020"],
        ["Model A: Baseline","Model B: Extended","Model C: Parsimonious (primary)"])):
    bs, ses = [], []
    for v in KEY_VARS:
        if v in m["names"]:
            idx = m["names"].index(v); bs.append(m["beta"][idx]); ses.append(m["se"][idx])
        else:
            bs.append(np.nan); ses.append(np.nan)
    bs = np.array(bs); ses = np.array(ses); valid = ~np.isnan(bs)
    ax.bar(x_pos[valid] + i * W, bs[valid], W,
           yerr=1.96 * ses[valid], capsize=3, color=clr, alpha=0.8, label=lbl,
           error_kw={"linewidth": 1.2})
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x_pos + W); ax.set_xticklabels(KEY_VARS, fontsize=11)
ax.set_ylabel("Log-Odds Coefficient (β)"); ax.legend(fontsize=9)
ax.set_title("Figure 2: Key IV Coefficients with 95% CI Across Model Specifications",
             fontweight="bold")
plt.tight_layout()
savefig("fig2_coefficients.png")


fig, ax = plt.subplots(figsize=(7, 5))
coord_r = np.linspace(-2.5, 2.5, 200)
b  = mC["beta"]; nm = mC["names"]
b0 = b[nm.index("Intercept")]; bt = b[nm.index("Toxicity")]
bc = b[nm.index("Coordination")]; bi = b[nm.index("Tox×Coord")]
for (lbl, tv), clr in zip(
        [("Low Toxicity (−1 SD)", -1), ("Mean Toxicity (0)", 0),
         ("High Toxicity (+1 SD)", 1)],
        ["#2166AC","#4DAC26","#D01C8B"]):
    ax.plot(coord_r, expit(b0 + bt*tv + bc*coord_r + bi*tv*coord_r),
            color=clr, linewidth=2.2, label=lbl)
ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
ax.set_xlabel("Coordination Score (standardised)"); ax.set_ylabel("P(Win)")
ax.set_title("Figure 3: Predicted Win Probability by Toxicity Level\n"
             "(Model C, all other controls at mean)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
savefig("fig3_interaction.png")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (col, lbl) in zip(axes, [
        ("toxicity_score", "Toxicity Score"), ("leadership_score", "Coordination Score")]):
    df["_q"] = pd.qcut(df[col], 4, labels=["Q1\n(Low)","Q2","Q3","Q4\n(High)"])
    means = df.groupby("_q", observed=True)["win"].mean()
    bars  = ax.bar(means.index, means.values,
                   color=[CLR_WIN if v > 0.5 else CLR_LOSS for v in means.values],
                   alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--")
    ax.set_ylim(0, 1); ax.set_xlabel(f"{lbl} Quartile"); ax.set_ylabel("Win Rate")
    ax.set_title(f"Win Rate by {lbl} Quartile", fontweight="bold")
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.suptitle("Figure 4: Empirical Win Rates Across Score Quartiles", fontweight="bold")
plt.tight_layout()
savefig("fig4_quartiles.png")


fig, ax = plt.subplots(figsize=(8, 4.5))
regs = ["EUW","KR","NA","VN"]
y_p  = np.arange(len(regs))
for var, clr, offset, lbl in [
        ("Coordination","#2166AC", 0.2,"Coordination"),
        ("Toxicity","#D6604D",-0.2,"Toxicity")]:
    bs  = [region_results[r]["beta"][region_results[r]["names"].index(var)] for r in regs]
    ses = [region_results[r]["se"][region_results[r]["names"].index(var)]   for r in regs]
    ax.barh(y_p + offset, bs, 0.35, xerr=1.96*np.array(ses), capsize=3,
            color=clr, alpha=0.85, label=lbl, error_kw={"linewidth":1.1})
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(y_p); ax.set_yticklabels(regs, fontsize=11)
ax.set_xlabel("Log-Odds Coefficient (β)"); ax.legend(fontsize=9)
ax.set_title("Figure 5: Per-Region Coefficients with 95% CI (Model C)",
             fontweight="bold")
plt.tight_layout()
savefig("fig5_regional.png")


fig, ax = plt.subplots(figsize=(9, 4.5))
sorted_vif = sorted(vifs.items(), key=lambda x: -x[1])
nms_v = [x[0] for x in sorted_vif]; vals_v = [x[1] for x in sorted_vif]
clrs_v = ["#D73027" if v>10 else "#FC8D59" if v>5 else "#91BFDB" for v in vals_v]
ax.barh(nms_v, vals_v, color=clrs_v, edgecolor="white", alpha=0.9)
ax.axvline(5,  color="orange", linewidth=1, linestyle="--", label="VIF = 5")
ax.axvline(10, color="red",    linewidth=1, linestyle="--", label="VIF = 10")
ax.set_xlabel("Variance Inflation Factor"); ax.legend(fontsize=8)
ax.set_title("Figure 6: Multicollinearity Diagnostics (VIF)", fontweight="bold")
plt.tight_layout()
savefig("fig6_vif.png")


# ═════════════════════════════════════════════════════════════════════════════
# 11.  FINAL CONSOLE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"  → {os.path.join(TBL_DIR, 'results_summary.txt')}")
print(f"  → {os.path.join(TBL_DIR, 'descriptive_stats.xlsx')}")
for i in range(1, 7):
    print(f"  → {os.path.join(FIG_DIR, f'fig{i}_*.png')}")
print()
print("KEY FINDINGS (Model C — Primary Specification):")
tox_i  = mC["names"].index("Toxicity")
crd_i  = mC["names"].index("Coordination")
int_i  = mC["names"].index("Tox×Coord")
print(f"  Toxicity   : β={mC['beta'][tox_i]:.4f}  SE={mC['se'][tox_i]:.4f}"
      f"  p={mC['pval'][tox_i]:.4f} {sig_stars(mC['pval'][tox_i])}")
print(f"  Coordination: β={mC['beta'][crd_i]:.4f}  SE={mC['se'][crd_i]:.4f}"
      f"  p={mC['pval'][crd_i]:.4f} {sig_stars(mC['pval'][crd_i])}")
print(f"  Tox×Coord  : β={mC['beta'][int_i]:.4f}  SE={mC['se'][int_i]:.4f}"
      f"  p={mC['pval'][int_i]:.4f} {sig_stars(mC['pval'][int_i])}")
print(f"  McFadden R²: {mC['mcfadden']:.4f}  |  Accuracy: {mC['acc']:.3f}")
print()
