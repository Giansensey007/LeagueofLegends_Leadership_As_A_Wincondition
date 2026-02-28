#!/usr/bin/env python3
"""
=============================================================================
  Statistical Analysis — "Does It Pay to Play Nice?"
  Applied Sports Research — League of Legends

  Full pipeline:
    B. Diagnostics  (descriptive stats, distributions, outliers, correlation, VIF)
    C. Modeling      (baseline + extended logistic regression, robustness checks)
    D. Visualizations (paper-ready charts)

  Usage:
      cd Analysis/
      python analysis.py

  Input :  ../Data Collection/match_dataset.csv
  Output:  figures/*.png  +  tables/*.xlsx  +  console summary
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, classification_report

# ─── Configuration ──────────────────────────────────────────────────────────

BASE     = Path(__file__).parent
DATA_CSV = BASE.parent / "Data Collection" / "match_dataset.csv"
FIG_DIR  = BASE / "figures"
TBL_DIR  = BASE / "tables"

FIG_DIR.mkdir(exist_ok=True)
TBL_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font="Helvetica", font_scale=1.1)
PALETTE = {"EUW": "#3498DB", "NA": "#E74C3C", "KR": "#2ECC71", "VN": "#F39C12"}

IV_COMPONENTS = [
    "max_feeding_index", "avg_feeding_index", "early_surrender",
    "vision_neglect_score",
    "coord_ping_ratio", "team_coord_ratio", "objectives_taken",
    "vision_leadership",
]
IV_COMPOSITE = ["toxicity_score", "leadership_score", "toxicity_x_leadership"]

CONTROLS_BASE = [
    "team_avg_kda", "team_avg_cs_min", "team_gold", "team_damage",
    "team_vision_score", "first_blood", "game_duration_min",
]
CONTROLS_EARLY = [
    "team_early_cs_10", "team_early_gold_adv", "team_early_takedowns",
    "avg_skill_index",
]
CONTROLS_ALL = CONTROLS_BASE + CONTROLS_EARLY

print("=" * 70)
print("  ANALYSIS — Does It Pay to Play Nice?")
print("=" * 70)

# ─── Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_CSV, keep_default_na=False)
print(f"\nLoaded {len(df)} rows  ({len(df)//2} matches x 2 teams)")
print(f"Regions: {sorted(df['region'].unique().tolist())}")
print(f"Columns: {len(df.columns)}")

# =============================================================================
#  PHASE B — DIAGNOSTICS
# =============================================================================

print("\n" + "=" * 70)
print("  PHASE B — DIAGNOSTICS")
print("=" * 70)

# ── B1: Descriptive Statistics ──────────────────────────────────────────────

print("\n--- B1: Descriptive Statistics ---")

desc_cols = IV_COMPONENTS + IV_COMPOSITE + CONTROLS_ALL + ["win"]
desc = df[desc_cols].describe().T
desc["skewness"] = df[desc_cols].skew()
desc["kurtosis"] = df[desc_cols].kurtosis()
desc.to_excel(TBL_DIR / "B1_descriptive_stats.xlsx")
print(desc[["mean", "std", "min", "max", "skewness", "kurtosis"]].round(3).to_string())

print("\n--- B1b: Key stats by region ---")
for region in sorted(df["region"].unique()):
    rdf = df[df["region"] == region]
    print(f"\n  [{region}]  n={len(rdf)//2} matches")
    print(f"    Avg toxicity:        {rdf['toxicity_score'].mean():.3f} +/- {rdf['toxicity_score'].std():.3f}")
    print(f"    Avg leadership:      {rdf['leadership_score'].mean():.3f} +/- {rdf['leadership_score'].std():.3f}")
    print(f"    Avg early CS @10:    {rdf['team_early_cs_10'].mean():.1f}")
    print(f"    Avg skill index:     {rdf['avg_skill_index'].mean():.1f}")
    print(f"    Avg duration:        {rdf['game_duration_min'].mean():.1f} min")

desc_by_region = df.groupby("region")[desc_cols].describe()
desc_by_region.to_excel(TBL_DIR / "B1_descriptive_by_region.xlsx")

desc_by_quarter = df.groupby(["game_year", "game_quarter"])[
    ["toxicity_score", "leadership_score", "win"]].agg(["mean", "std", "count"])
desc_by_quarter.to_excel(TBL_DIR / "B1_descriptive_by_quarter.xlsx")

# ── B2: Distribution Plots ─────────────────────────────────────────────────

print("\n--- B2: Distribution Plots ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax_i, (col, title, color) in enumerate([
    ("toxicity_score", "Toxicity Score Distribution", "#E74C3C"),
    ("leadership_score", "Leadership Score Distribution", "#2ECC71"),
]):
    ax = axes[ax_i]
    for region in sorted(df["region"].unique()):
        rdf = df[df["region"] == region]
        ax.hist(rdf[col], bins=50, alpha=0.5, label=region,
                color=PALETTE[region], density=True)
    ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    sample = df[col].dropna().sample(min(5000, len(df)), random_state=42)
    stat, p = stats.shapiro(sample)
    ax.text(0.02, 0.95, f"Shapiro p={p:.4f}", transform=ax.transAxes,
            fontsize=9, va="top", fontstyle="italic",
            color="red" if p < 0.05 else "green")
fig.tight_layout()
fig.savefig(FIG_DIR / "B2_score_distributions.png", dpi=200)
plt.close()
print("  Saved B2_score_distributions.png")

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, col in enumerate(IV_COMPONENTS):
    ax = axes[i // 4][i % 4]
    df[col].hist(bins=40, ax=ax, color="#7FB3D8", edgecolor="white")
    ax.set_title(col.replace("_", " ").title(), fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("IV Component Distributions", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "B2_component_distributions.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved B2_component_distributions.png")

# ── B3: Outlier Detection ──────────────────────────────────────────────────

print("\n--- B3: Outlier Detection ---")

outlier_cols = ["toxicity_score", "leadership_score", "max_feeding_index",
                "team_total_pings", "team_gold", "avg_skill_index"]
outlier_report = []
for col in outlier_cols:
    z = np.abs(stats.zscore(df[col].dropna()))
    n_outliers = (z > 3).sum()
    pct = n_outliers / len(z) * 100
    outlier_report.append({"variable": col, "n_outliers_z3": n_outliers,
                           "pct": round(pct, 2)})
    print(f"  {col}: {n_outliers} outliers (|z|>3) = {pct:.2f}%")

pd.DataFrame(outlier_report).to_excel(TBL_DIR / "B3_outliers.xlsx", index=False)

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(22, 5))
for i, col in enumerate(outlier_cols):
    sns.boxplot(data=df, x="region", y=col, ax=axes[i], palette=PALETTE,
                order=["EUW", "NA", "KR", "VN"])
    axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
fig.suptitle("Outlier Box Plots by Region", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "B3_outlier_boxplots.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved B3_outlier_boxplots.png")

# ── B4: Correlation Matrix ─────────────────────────────────────────────────

print("\n--- B4: Correlation Matrix ---")

corr_cols = IV_COMPOSITE + CONTROLS_ALL + ["win"]
corr = df[corr_cols].corr()
corr.to_excel(TBL_DIR / "B4_correlation_matrix.xlsx")

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={"size": 7},
            xticklabels=[c.replace("_", "\n") for c in corr_cols],
            yticklabels=[c.replace("_", " ") for c in corr_cols])
ax.set_title("Correlation Matrix — IVs, Controls (incl. Early-Game) & DV",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "B4_correlation_heatmap.png", dpi=200)
plt.close()
print("  Saved B4_correlation_heatmap.png")

# ── B5: VIF Multicollinearity ──────────────────────────────────────────────

print("\n--- B5: VIF Multicollinearity Check ---")

vif_cols = IV_COMPOSITE + CONTROLS_ALL
vif_data = df[vif_cols].dropna()
vif_data = sm.add_constant(vif_data)
vif_results = []
for i, col in enumerate(vif_data.columns):
    if col == "const":
        continue
    vif_val = variance_inflation_factor(vif_data.values, i)
    vif_results.append({"variable": col, "VIF": round(vif_val, 2)})
    flag = " !!!" if vif_val > 5 else ""
    print(f"  {col:30s}  VIF = {vif_val:.2f}{flag}")

pd.DataFrame(vif_results).to_excel(TBL_DIR / "B5_vif.xlsx", index=False)

# =============================================================================
#  PHASE C — STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("  PHASE C — LOGISTIC REGRESSION")
print("=" * 70)

# ── C1: Restructure to 1 row per match (blue-side perspective) ─────────

print("\n--- C1: Restructuring to 1 row per match (blue side = team 100) ---")

df_blue = df[df["team_id"] == 100].copy()
df_blue["side"] = "blue"
print(f"  Analysis dataset: {len(df_blue)} rows (1 per match, blue-side perspective)")
print(f"  Win rate: {df_blue['win'].mean():.3f}")

region_dummies = pd.get_dummies(df_blue["region"], prefix="region", drop_first=True)
region_dummies = region_dummies.astype(int)


def fit_logit(X_df, y_series, label="model"):
    """Fit logistic regression and return (model, results_df)."""
    mask = X_df.notna().all(axis=1) & y_series.notna()
    X_clean = X_df[mask]
    y_clean = y_series[mask]
    m = sm.Logit(y_clean, X_clean).fit(disp=0)
    rdf = pd.DataFrame({
        "Coefficient": m.params,
        "Std Error": m.bse,
        "z-value": m.tvalues,
        "p-value": m.pvalues,
        "Odds Ratio": np.exp(m.params),
        "CI 2.5%": np.exp(m.conf_int()[0]),
        "CI 97.5%": np.exp(m.conf_int()[1]),
    }).round(4)
    print(f"\n  [{label}]  n={int(m.nobs)}  pseudo-R2={m.prsquared:.4f}  "
          f"AIC={m.aic:.1f}  BIC={m.bic:.1f}")
    return m, rdf, X_clean, y_clean


# ── C2a: Baseline Model (original controls only) ─────────────────────────

print("\n--- C2a: Baseline Logistic Regression (original controls) ---")

X_base = df_blue[IV_COMPOSITE + CONTROLS_BASE].copy()
X_base = pd.concat([X_base, region_dummies], axis=1)
X_base = sm.add_constant(X_base)
y = df_blue["win"]

model_base, results_base, X_base_clean, y_base_clean = fit_logit(
    X_base, y, "Baseline")
print(model_base.summary2())
results_base.to_excel(TBL_DIR / "C2_logistic_baseline.xlsx")

auc_base = roc_auc_score(y_base_clean, model_base.predict(X_base_clean))
print(f"  AUC-ROC: {auc_base:.4f}")

# ── C2b: Extended Model (+ early-game controls + skill proxy) ────────────

print("\n--- C2b: Extended Logistic Regression (+early-game + skill) ---")

X_ext = df_blue[IV_COMPOSITE + CONTROLS_ALL].copy()
X_ext = pd.concat([X_ext, region_dummies], axis=1)
X_ext = sm.add_constant(X_ext)

model_ext, results_ext, X_ext_clean, y_ext_clean = fit_logit(
    X_ext, y, "Extended")
print(model_ext.summary2())
results_ext.to_excel(TBL_DIR / "C2_logistic_extended.xlsx")

auc_ext = roc_auc_score(y_ext_clean, model_ext.predict(X_ext_clean))
print(f"  AUC-ROC: {auc_ext:.4f}")

# ── C2c: Model comparison table ──────────────────────────────────────────

print("\n--- C2c: Model Comparison ---")

compare_vars = IV_COMPOSITE
comparison = pd.DataFrame(index=compare_vars)
for label, rdf, m in [("Baseline", results_base, model_base),
                       ("Extended", results_ext, model_ext)]:
    for var in compare_vars:
        if var in rdf.index:
            comparison.loc[var, f"{label}_OR"] = rdf.loc[var, "Odds Ratio"]
            comparison.loc[var, f"{label}_p"] = rdf.loc[var, "p-value"]
            comparison.loc[var, f"{label}_CI_lo"] = rdf.loc[var, "CI 2.5%"]
            comparison.loc[var, f"{label}_CI_hi"] = rdf.loc[var, "CI 97.5%"]
    comparison.loc["_model_fit", f"{label}_OR"] = np.nan
    comparison.loc["_model_fit", f"{label}_p"] = np.nan
    comparison.loc["pseudo_R2", f"{label}_OR"] = round(m.prsquared, 4)
    comparison.loc["AIC", f"{label}_OR"] = round(m.aic, 1)
    comparison.loc["BIC", f"{label}_OR"] = round(m.bic, 1)
    comparison.loc["n", f"{label}_OR"] = int(m.nobs)

comparison.to_excel(TBL_DIR / "C2_model_comparison.xlsx")
print(comparison.to_string())

# ── C3: Classification Performance (extended model) ──────────────────────

print("\n--- C3: Classification Performance (Extended Model) ---")

y_pred_prob = model_ext.predict(X_ext_clean)
y_pred = (y_pred_prob >= 0.5).astype(int)
print(f"  AUC-ROC: {auc_ext:.4f}")
print(classification_report(y_ext_clean, y_pred, target_names=["Loss", "Win"]))

# ── C4: Robustness — per-region regressions (extended model) ────────────

print("\n--- C4: Robustness — Per-Region Sub-Regressions (Extended) ---")

robustness_rows = []
feature_cols_ext = IV_COMPOSITE + CONTROLS_ALL
for region in sorted(df_blue["region"].unique()):
    rdf = df_blue[df_blue["region"] == region]
    Xr = rdf[feature_cols_ext].copy()
    Xr = sm.add_constant(Xr)
    yr = rdf["win"]
    mask_r = Xr.notna().all(axis=1) & yr.notna()
    Xr, yr = Xr[mask_r], yr[mask_r]

    try:
        m = sm.Logit(yr, Xr).fit(disp=0)
        row = {"region": region, "n": int(m.nobs),
               "pseudo_R2": round(m.prsquared, 4)}
        for var in IV_COMPOSITE:
            row[f"{var}_coef"] = round(m.params.get(var, np.nan), 4)
            row[f"{var}_p"] = round(m.pvalues.get(var, np.nan), 4)
            row[f"{var}_OR"] = round(np.exp(m.params.get(var, 0)), 4)
        robustness_rows.append(row)
        print(f"\n  [{region}]  n={int(m.nobs)}  pseudo-R2={m.prsquared:.4f}")
        for var in IV_COMPOSITE:
            sig = ("***" if m.pvalues[var] < 0.001
                   else "**" if m.pvalues[var] < 0.01
                   else "*" if m.pvalues[var] < 0.05 else "")
            print(f"    {var:30s}  coef={m.params[var]:+.4f}  "
                  f"OR={np.exp(m.params[var]):.4f}  p={m.pvalues[var]:.4f} {sig}")
    except Exception as e:
        print(f"  [{region}] Error: {e}")

pd.DataFrame(robustness_rows).to_excel(
    TBL_DIR / "C4_robustness_by_region.xlsx", index=False)

# ── C5: Robustness — close games only (gold diff < 5000) ────────────────

print("\n--- C5: Robustness — Close Games (gold diff < 5000) ---")

df_red = df[df["team_id"] == 200][["match_id", "team_gold"]].rename(
    columns={"team_gold": "opp_gold"})
df_match = df_blue.merge(df_red, on="match_id", how="left")
df_match["gold_diff"] = abs(df_match["team_gold"] - df_match["opp_gold"])
df_close = df_match[df_match["gold_diff"] < 5000].copy()

print(f"  Close games: {len(df_close)} / {len(df_blue)} "
      f"({100*len(df_close)/len(df_blue):.1f}%)")

X_close = df_close[IV_COMPOSITE + CONTROLS_ALL].copy()
region_dum_close = pd.get_dummies(
    df_close["region"], prefix="region", drop_first=True).astype(int)
X_close = pd.concat([X_close, region_dum_close], axis=1)
X_close = sm.add_constant(X_close)
y_close = df_close["win"]

model_close, results_close, _, _ = fit_logit(X_close, y_close, "Close Games")
results_close.to_excel(TBL_DIR / "C5_close_games_robustness.xlsx")

print("\n  Key IV results (close games):")
for var in IV_COMPOSITE:
    if var in results_close.index:
        r = results_close.loc[var]
        sig = ("***" if r["p-value"] < 0.001
               else "**" if r["p-value"] < 0.01
               else "*" if r["p-value"] < 0.05 else "")
        print(f"    {var:30s}  OR={r['Odds Ratio']:.4f}  p={r['p-value']:.4f} {sig}")

# =============================================================================
#  PHASE D — VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("  PHASE D — VISUALIZATIONS")
print("=" * 70)

# ── D1: Win Rate by Toxicity / Leadership Quartiles ────────────────────

print("\n--- D1: Win Rate by Score Quartiles ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax_i, (score, color, title) in enumerate([
    ("toxicity_score", "#E74C3C", "Win Rate by Toxicity Quartile"),
    ("leadership_score", "#2ECC71", "Win Rate by Leadership Quartile"),
]):
    df_blue[f"{score}_q"] = pd.qcut(
        df_blue[score], 4, labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"])
    grouped = df_blue.groupby(f"{score}_q", observed=True)["win"].agg(
        ["mean", "count", "std"])
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])

    ax = axes[ax_i]
    ax.bar(range(4), grouped["mean"], yerr=grouped["se"] * 1.96,
           capsize=5, color=color, alpha=0.8, edgecolor="white", lw=1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(grouped.index)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(0.3, 0.7)
    for i, (val, n) in enumerate(zip(grouped["mean"], grouped["count"])):
        ax.text(i, val + 0.025, f"{val:.3f}\n(n={n})", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "D1_winrate_by_quartile.png", dpi=200)
plt.close()
print("  Saved D1_winrate_by_quartile.png")

# ── D2: Region Comparison (Box Plots) ──────────────────────────────────

print("\n--- D2: Region Comparison ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (col, title) in enumerate([
    ("toxicity_score", "Toxicity Score by Region"),
    ("leadership_score", "Leadership Score by Region"),
    ("team_total_pings", "Total Team Pings by Region"),
]):
    sns.boxplot(data=df_blue, x="region", y=col, ax=axes[i],
                palette=PALETTE, order=["EUW", "NA", "KR", "VN"],
                showfliers=False)
    axes[i].set_title(title, fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "D2_region_comparison.png", dpi=200)
plt.close()
print("  Saved D2_region_comparison.png")

# ── D3: Quarterly Trends ──────────────────────────────────────────────

print("\n--- D3: Quarterly Trends ---")

df_blue["period"] = (df_blue["game_year"].astype(str) + "-"
                     + df_blue["game_quarter"])
quarterly = df_blue.groupby("period").agg(
    toxicity_mean=("toxicity_score", "mean"),
    leadership_mean=("leadership_score", "mean"),
    win_rate=("win", "mean"),
    n=("win", "count"),
).reset_index().sort_values("period")
quarterly = quarterly[quarterly["n"] >= 50]

fig, ax1 = plt.subplots(figsize=(12, 5))
x = range(len(quarterly))
ax1.plot(x, quarterly["toxicity_mean"], "o-", color="#E74C3C", lw=2,
         markersize=7, label="Toxicity Score")
ax1.plot(x, quarterly["leadership_mean"], "s-", color="#2ECC71", lw=2,
         markersize=7, label="Leadership Score")
ax1.set_xticks(x)
ax1.set_xticklabels(quarterly["period"], rotation=45, ha="right")
ax1.set_ylabel("Score (standardized)", fontsize=12)
ax1.legend(loc="upper left")
ax1.set_title("Toxicity & Leadership Over Time", fontsize=14, fontweight="bold")

ax2 = ax1.twinx()
ax2.bar(x, quarterly["n"], alpha=0.15, color="gray", label="Sample size")
ax2.set_ylabel("n (matches)", fontsize=12, color="gray")

fig.tight_layout()
fig.savefig(FIG_DIR / "D3_quarterly_trends.png", dpi=200)
plt.close()
print("  Saved D3_quarterly_trends.png")

# ── D4: Odds Ratio Forest Plot (Extended Model) ─────────────────────────

print("\n--- D4: Odds Ratio Forest Plot (Extended Model) ---")

forest_vars = IV_COMPOSITE + ["team_avg_kda", "first_blood",
                               "team_vision_score", "avg_skill_index",
                               "team_early_cs_10"]
forest_df = results_ext.loc[results_ext.index.isin(forest_vars)].copy()
forest_df = forest_df.sort_values("Odds Ratio")

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = range(len(forest_df))
ax.barh(y_pos, forest_df["Odds Ratio"] - 1, left=1,
        color=[("#E74C3C" if v < 1 else "#2ECC71")
               for v in forest_df["Odds Ratio"]],
        alpha=0.7, height=0.6, edgecolor="white")
ax.errorbar(forest_df["Odds Ratio"], y_pos,
            xerr=[forest_df["Odds Ratio"] - forest_df["CI 2.5%"],
                  forest_df["CI 97.5%"] - forest_df["Odds Ratio"]],
            fmt="ko", capsize=4, markersize=5, lw=1.5)
ax.axvline(1, color="gray", ls="--", lw=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([v.replace("_", " ").title() for v in forest_df.index],
                   fontsize=10)
ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
ax.set_title("Odds Ratios — Extended Model (with Early-Game Controls)",
             fontsize=14, fontweight="bold")

for i, (idx, row) in enumerate(forest_df.iterrows()):
    sig = ("***" if row["p-value"] < 0.001
           else "**" if row["p-value"] < 0.01
           else "*" if row["p-value"] < 0.05 else "ns")
    ax.text(max(row["CI 97.5%"], 1) + 0.02, i,
            f"OR={row['Odds Ratio']:.3f} {sig}", fontsize=8, va="center")

fig.tight_layout()
fig.savefig(FIG_DIR / "D4_odds_ratio_forest.png", dpi=200)
plt.close()
print("  Saved D4_odds_ratio_forest.png")

# ── D5: Win Rate by Toxicity x Leadership (2D heatmap) ────────────────

print("\n--- D5: Toxicity x Leadership Interaction Heatmap ---")

df_blue["tox_q"] = pd.qcut(df_blue["toxicity_score"], 4,
                            labels=["Low", "Med-Low", "Med-High", "High"])
df_blue["lead_q"] = pd.qcut(df_blue["leadership_score"], 4,
                             labels=["Low", "Med-Low", "Med-High", "High"])
pivot = df_blue.groupby(
    ["tox_q", "lead_q"], observed=True)["win"].mean().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
            vmin=0.3, vmax=0.7, ax=ax, linewidths=1,
            cbar_kws={"label": "Win Rate"})
ax.set_xlabel("Leadership Score Quartile", fontsize=12)
ax.set_ylabel("Toxicity Score Quartile", fontsize=12)
ax.set_title("Win Rate: Toxicity x Leadership Interaction",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "D5_interaction_heatmap.png", dpi=200)
plt.close()
print("  Saved D5_interaction_heatmap.png")

# ── D6: Model Comparison Forest Plot (Baseline vs Extended) ─────────────

print("\n--- D6: Model Comparison Forest Plot ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax_i, var in enumerate(IV_COMPOSITE):
    ax = axes[ax_i]
    models_data = []
    for label, rdf, color in [("Baseline", results_base, "#3498DB"),
                               ("Extended", results_ext, "#E67E22"),
                               ("Close Games", results_close, "#9B59B6")]:
        if var in rdf.index:
            r = rdf.loc[var]
            models_data.append((label, r["Odds Ratio"],
                                r["CI 2.5%"], r["CI 97.5%"],
                                r["p-value"], color))

    for i, (lbl, or_val, ci_lo, ci_hi, pv, col) in enumerate(models_data):
        sig = ("***" if pv < 0.001 else "**" if pv < 0.01
               else "*" if pv < 0.05 else "ns")
        ax.errorbar(or_val, i, xerr=[[or_val - ci_lo], [ci_hi - or_val]],
                    fmt="o", color=col, capsize=6, markersize=8, lw=2,
                    label=f"{lbl} (OR={or_val:.3f} {sig})")

    ax.axvline(1, color="gray", ls="--", lw=1.5)
    ax.set_yticks(range(len(models_data)))
    ax.set_yticklabels([m[0] for m in models_data], fontsize=10)
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=11)
    ax.set_title(var.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="best")

fig.suptitle("Model Comparison: Baseline vs Extended vs Close Games",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "D6_model_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved D6_model_comparison.png")

# =============================================================================
#  SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("  COMPLETE — All outputs saved")
print("=" * 70)
print(f"\n  Tables:  {TBL_DIR}/")
for f in sorted(TBL_DIR.glob("*.xlsx")):
    print(f"    {f.name}")
print(f"\n  Figures: {FIG_DIR}/")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"    {f.name}")
print()
