#!/usr/bin/env python3
"""
=============================================================================
  Variable Construction — From Raw Match JSON to Study Dataset
  "Does It Pay to Play Nice?"
  Applied Sports Research — League of Legends

  Reads raw match JSON files produced by collect_matches.py (stored in
  raw_matches/{REGION}/ subdirectories) and outputs a single CSV
  (match_dataset.csv) with one row per team per match, containing all
  study variables ready for regression analysis.

  Columns added compared to single-region version:
    - region     : EUW | NA | KR | VN
    - game_date  : YYYY-MM-DD  (derived from gameCreation timestamp)

  Usage:
      python build_variables.py                    # all regions
      python build_variables.py --region EUW NA    # specific regions
=============================================================================
"""

import json
import logging
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
RAW_DIR    = BASE_DIR / "raw_matches"
OUTPUT_CSV = BASE_DIR / "match_dataset.csv"

ALL_REGIONS = ["EUW", "NA", "KR", "VN"]

# Ping fields classified as "coordinating" (leadership) vs all
COORDINATING_PINGS = [
    "onMyWayPings",
    "commandPings",
    "assistMePings",
    "pushPings",
]
ALL_PING_FIELDS = [
    "allInPings",
    "assistMePings",
    "commandPings",
    "dangerPings",
    "enemyMissingPings",
    "enemyVisionPings",
    "getBackPings",
    "holdPings",
    "needVisionPings",
    "onMyWayPings",
    "pushPings",
    "visionClearedPings",
]

RANK_MAP = {
    "IRON": 0, "BRONZE": 1, "SILVER": 2, "GOLD": 3,
    "PLATINUM": 4, "EMERALD": 5, "DIAMOND": 6,
    "MASTER": 7, "GRANDMASTER": 8, "CHALLENGER": 9,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build")

# ─── Helper Functions ───────────────────────────────────────────────────────

def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


def player_feeding_index(p: dict) -> float:
    """Deaths / (Kills + Assists + 1). Higher = worse."""
    deaths = p.get("deaths", 0)
    ka = p.get("kills", 0) + p.get("assists", 0)
    return deaths / max(ka, 1)


def player_total_pings(p: dict) -> int:
    return sum(p.get(f, 0) for f in ALL_PING_FIELDS)


def player_coord_pings(p: dict) -> int:
    return sum(p.get(f, 0) for f in COORDINATING_PINGS)


def classify_composition(participants: list[dict]) -> str:
    total_magic = sum(p.get("magicDamageDealtToChampions", 0)
                      for p in participants)
    total_phys  = sum(p.get("physicalDamageDealtToChampions", 0)
                      for p in participants)
    total_tank  = sum(p.get("totalDamageTaken", 0) for p in participants)
    total_dmg   = total_magic + total_phys + 1

    if total_tank / max(total_dmg, 1) > 1.2:
        return "engage_dominant"
    elif total_magic / max(total_dmg, 1) > 0.55:
        return "poke_dominant"
    else:
        return "standard"

# ─── Main Processing ───────────────────────────────────────────────────────

def process_match(data: dict, region: str) -> list[dict]:
    """
    Process one match JSON into two rows (one per team).
    Returns a list of 0 or 2 dicts.
    """
    info     = data.get("info", {})
    metadata = data.get("metadata", {})
    match_id = metadata.get("matchId", "unknown")

    game_duration     = info.get("gameDuration", 0)
    game_duration_min = game_duration / 60.0
    patch = info.get("gameVersion", "").rsplit(".", 1)[0]
    game_ended_in_surrender = info.get("gameEndedInSurrender", False)

    # Derive game date, year, quarter from gameCreation (ms epoch)
    game_creation_ms = info.get("gameCreation", 0)
    if game_creation_ms:
        dt = datetime.fromtimestamp(game_creation_ms / 1000, tz=timezone.utc)
        game_date = dt.strftime("%Y-%m-%d")
        game_year = dt.year
        game_quarter = f"Q{(dt.month - 1) // 3 + 1}"
    else:
        game_date = ""
        game_year = ""
        game_quarter = ""

    participants = info.get("participants", [])
    if len(participants) != 10:
        return []

    teams = {100: [], 200: []}
    for p in participants:
        tid = p.get("teamId")
        if tid in teams:
            teams[tid].append(p)

    if len(teams[100]) != 5 or len(teams[200]) != 5:
        return []

    # Team-level objectives
    team_objectives = {}
    for t in info.get("teams", []):
        tid = t.get("teamId")
        obj = t.get("objectives", {})
        team_objectives[tid] = {
            "dragons":     obj.get("dragon",     {}).get("kills", 0),
            "barons":      obj.get("baron",      {}).get("kills", 0),
            "heralds":     obj.get("riftHerald", {}).get("kills", 0),
            "towers":      obj.get("tower",      {}).get("kills", 0),
            "first_blood": obj.get("champion",   {}).get("first", False),
        }

    rows = []
    for team_id, players in teams.items():
        win = players[0].get("win", False)

        kills   = [p.get("kills", 0)   for p in players]
        deaths  = [p.get("deaths", 0)  for p in players]
        assists = [p.get("assists", 0) for p in players]
        gold    = [p.get("goldEarned", 0) for p in players]
        cs = [p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)
              for p in players]
        vision       = [p.get("visionScore", 0) for p in players]
        wards_placed = [p.get("wardsPlaced", 0) for p in players]
        damage       = [p.get("totalDamageDealtToChampions", 0) for p in players]

        team_kills   = sum(kills)
        team_deaths  = sum(deaths)
        team_assists = sum(assists)
        team_avg_kda = safe_div(team_kills + team_assists, max(team_deaths, 1))
        team_avg_cs_min = safe_div(sum(cs), game_duration_min * 5)
        team_gold = sum(gold)

        # ── IV1: TOXICITY SCORE ─────────────────────────────────
        feeding_indices   = [player_feeding_index(p) for p in players]
        max_feeding_index = max(feeding_indices)
        avg_feeding_index = sum(feeding_indices) / 5.0
        early_surrender   = 1 if (game_ended_in_surrender and not win) else 0
        vision_per_min    = [safe_div(v, game_duration_min) for v in vision]
        vision_neglect_count = sum(1 for vpm in vision_per_min if vpm < 0.3)
        vision_neglect_score = vision_neglect_count / 5.0

        # ── IV2: LEADERSHIP SCORE ───────────────────────────────
        total_pings_pp = [player_total_pings(p) for p in players]
        coord_pings_pp = [player_coord_pings(p) for p in players]
        max_ping_idx   = total_pings_pp.index(max(total_pings_pp))
        top_total      = total_pings_pp[max_ping_idx]
        top_coord      = coord_pings_pp[max_ping_idx]
        coord_ping_ratio = safe_div(top_coord, top_total)

        team_total_pings = sum(total_pings_pp)
        team_coord_pings = sum(coord_pings_pp)
        team_coord_ratio = safe_div(team_coord_pings, team_total_pings)

        obj = team_objectives.get(team_id, {})
        objectives_taken = (obj.get("dragons", 0) +
                            obj.get("barons", 0) +
                            obj.get("heralds", 0))

        vision_leaders = 0
        for p in players:
            pos = p.get("teamPosition", "")
            vpm = safe_div(p.get("visionScore", 0), game_duration_min)
            if pos == "UTILITY" and vpm >= 1.0:
                vision_leaders += 1
            elif pos == "JUNGLE" and vpm >= 0.6:
                vision_leaders += 1
        vision_leadership = 1 if vision_leaders >= 2 else 0

        comp_type = classify_composition(players)

        ping_breakdown = {}
        for field in ALL_PING_FIELDS:
            ping_breakdown[f"team_{field}"] = sum(
                p.get(field, 0) for p in players)

        row = {
            # ── identifiers / metadata ──
            "match_id":          match_id,
            "region":            region,
            "game_date":         game_date,
            "game_year":         game_year,
            "game_quarter":      game_quarter,
            "team_id":           team_id,
            "win":               int(win),
            "game_duration_min": round(game_duration_min, 2),
            "patch":             patch,

            # DV
            "team_gold": team_gold,

            # IV1: Toxicity
            "max_feeding_index":   round(max_feeding_index, 4),
            "avg_feeding_index":   round(avg_feeding_index, 4),
            "early_surrender":     early_surrender,
            "vision_neglect_score": round(vision_neglect_score, 4),

            # IV2: Leadership
            "coord_ping_ratio":  round(coord_ping_ratio, 4),
            "team_coord_ratio":  round(team_coord_ratio, 4),
            "objectives_taken":  objectives_taken,
            "vision_leadership": vision_leadership,
            "team_total_pings":  team_total_pings,
            "team_coord_pings":  team_coord_pings,

            # Controls
            "team_avg_kda":    round(team_avg_kda, 4),
            "team_avg_cs_min": round(team_avg_cs_min, 4),
            "team_kills":      team_kills,
            "team_deaths":     team_deaths,
            "team_assists":    team_assists,
            "comp_type":       comp_type,
            "first_blood":     int(obj.get("first_blood", False)),

            # Objectives detail
            "dragons":  obj.get("dragons", 0),
            "barons":   obj.get("barons", 0),
            "heralds":  obj.get("heralds", 0),
            "towers":   obj.get("towers", 0),

            # Vision
            "team_vision_score": sum(vision),
            "team_wards_placed": sum(wards_placed),

            # Damage
            "team_damage": sum(damage),
        }
        row.update(ping_breakdown)
        rows.append(row)

    return rows

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build study dataset from raw match JSONs")
    parser.add_argument(
        "--region", nargs="*", default=None,
        help="Region(s) to process. Default: auto-detect from raw_matches/")
    args = parser.parse_args()

    # Discover region subdirectories
    if args.region:
        regions = [r.upper() for r in args.region]
    else:
        regions = sorted([
            d.name for d in RAW_DIR.iterdir()
            if d.is_dir() and any(d.glob("*.json"))
        ])

    if not regions:
        log.error(f"No region subdirectories with JSON files in {RAW_DIR}/")
        log.error("Run collect_matches.py first.")
        sys.exit(1)

    log.info(f"Regions to process: {regions}")

    all_rows = []
    errors   = 0

    for region in regions:
        region_dir  = RAW_DIR / region
        match_files = sorted(region_dir.glob("*.json"))
        if not match_files:
            log.warning(f"[{region}] No JSON files in {region_dir}, skipping.")
            continue

        log.info(f"[{region}] Processing {len(match_files)} match files ...")

        for f in tqdm(match_files, desc=f"[{region}] Building vars"):
            try:
                data = json.loads(f.read_text())
                rows = process_match(data, region)
                all_rows.extend(rows)
            except Exception as e:
                errors += 1
                log.warning(f"Error processing {f.name}: {e}")

    if not all_rows:
        log.error("No valid rows produced. Check raw match files.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # ── Standardized composite scores (z-score per region) ───────────
    # Toxicity Score
    for col in ["max_feeding_index", "early_surrender", "vision_neglect_score"]:
        mean = df[col].mean()
        std  = df[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std if std > 0 else 0.0

    df["toxicity_score"] = (
        df["max_feeding_index_z"] +
        df["early_surrender_z"] +
        df["vision_neglect_score_z"]
    ).round(4)

    # Leadership Score
    for col in ["coord_ping_ratio", "objectives_taken", "vision_leadership"]:
        mean = df[col].mean()
        std  = df[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std if std > 0 else 0.0

    df["leadership_score"] = (
        df["coord_ping_ratio_z"] +
        df["objectives_taken_z"] +
        df["vision_leadership_z"]
    ).round(4)

    # Interaction term
    df["toxicity_x_leadership"] = (
        df["toxicity_score"] * df["leadership_score"]
    ).round(4)

    # Drop z-score intermediates
    z_cols = [c for c in df.columns if c.endswith("_z")]
    df.drop(columns=z_cols, inplace=True)

    # ── Save ──────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)

    log.info("=" * 64)
    log.info(f"  Dataset saved:  {OUTPUT_CSV}")
    log.info(f"  Rows:           {len(df)}  "
             f"({len(df)//2} matches × 2 teams)")
    log.info(f"  Columns:        {len(df.columns)}")
    log.info(f"  Regions:        {df['region'].unique().tolist()}")
    log.info(f"  Errors:         {errors}")
    log.info("=" * 64)

    # Quick summary
    log.info("\n--- Summary by region ---")
    for region in df["region"].unique():
        rdf = df[df["region"] == region]
        log.info(f"  [{region}]  matches={len(rdf)//2}  "
                 f"win_rate={rdf['win'].mean():.3f}  "
                 f"dates={rdf['game_date'].min()} → {rdf['game_date'].max()}")

    log.info("\n--- Overall ---")
    log.info(f"  Win rate:           {df['win'].mean():.3f}")
    log.info(f"  Avg Toxicity:       {df['toxicity_score'].mean():.3f}")
    log.info(f"  Avg Leadership:     {df['leadership_score'].mean():.3f}")
    log.info(f"  Avg duration (min): {df['game_duration_min'].mean():.1f}")
    log.info(f"  Avg team pings:     {df['team_total_pings'].mean():.0f}")


if __name__ == "__main__":
    main()
