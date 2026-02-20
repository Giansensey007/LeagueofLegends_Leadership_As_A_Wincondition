#!/usr/bin/env python3
"""Generate a variable codebook (data dictionary) as Excel."""

import pandas as pd
from pathlib import Path

CODEBOOK = [
    # (Variable, Type, Role, Description)
    ("match_id", "string", "Identifier", "Unique Riot match ID — links the two team rows belonging to the same game"),
    ("region", "categorical", "Grouping", "Server region: EUW, NA, KR, or VN"),
    ("game_date", "date", "Grouping", "Date the match was played (YYYY-MM-DD, UTC)"),
    ("game_year", "integer", "Grouping", "Calendar year of the match (2024, 2025, 2026)"),
    ("game_quarter", "categorical", "Grouping", "Calendar quarter: Q1 (Jan-Mar), Q2 (Apr-Jun), Q3 (Jul-Sep), Q4 (Oct-Dec)"),
    ("team_id", "integer", "Identifier", "100 = Blue side, 200 = Red side"),
    ("win", "binary", "DV", "1 = team won, 0 = team lost"),
    ("game_duration_min", "float", "Control", "Game length in minutes (>= 15 min after filtering)"),
    ("patch", "string", "Control", "Game patch version (e.g. 15.3)"),
    ("team_gold", "integer", "Control", "Total gold earned by all 5 team members"),
    # IV1: Toxicity
    ("max_feeding_index", "float", "IV1-Component", "Highest Deaths/(Kills+Assists+1) among team members — intentional feeding proxy"),
    ("avg_feeding_index", "float", "IV1-Component", "Team-average feeding index"),
    ("early_surrender", "binary", "IV1-Component", "1 if losing team in a surrendered game (tilt/give-up proxy)"),
    ("vision_neglect_score", "float [0-1]", "IV1-Component", "Proportion of team with < 0.3 vision/min (griefing/disengagement proxy)"),
    ("toxicity_score", "float (z)", "IV1-Composite", "Standardized sum of max_feeding_index_z + early_surrender_z + vision_neglect_score_z"),
    # IV2: Leadership
    ("coord_ping_ratio", "float [0-1]", "IV2-Component", "Top-pinging player's coordinating pings / total pings (shot-caller proxy)"),
    ("team_coord_ratio", "float [0-1]", "IV2-Component", "Team-wide coordinating pings / total pings"),
    ("objectives_taken", "integer", "IV2-Component", "Dragons + Barons + Rift Heralds taken (objective coordination)"),
    ("vision_leadership", "binary", "IV2-Component", "1 if both Support (>= 1.0 vis/min) and Jungle (>= 0.6 vis/min) meet vision thresholds"),
    ("leadership_score", "float (z)", "IV2-Composite", "Standardized sum of coord_ping_ratio_z + objectives_taken_z + vision_leadership_z"),
    # Interaction
    ("toxicity_x_leadership", "float", "Interaction", "toxicity_score * leadership_score (moderation effect)"),
    # Controls
    ("team_avg_kda", "float", "Control", "Team (Kills+Assists) / max(Deaths, 1)"),
    ("team_avg_cs_min", "float", "Control", "Team average CS per minute"),
    ("team_kills", "integer", "Control", "Total team kills"),
    ("team_deaths", "integer", "Control", "Total team deaths"),
    ("team_assists", "integer", "Control", "Total team assists"),
    ("comp_type", "categorical", "Control", "Team composition class: standard, poke_dominant, engage_dominant"),
    ("first_blood", "binary", "Control", "1 if team got first blood"),
    ("dragons", "integer", "Detail", "Dragon objectives taken"),
    ("barons", "integer", "Detail", "Baron objectives taken"),
    ("heralds", "integer", "Detail", "Rift Herald objectives taken"),
    ("towers", "integer", "Detail", "Towers destroyed"),
    ("team_vision_score", "integer", "Control", "Sum of vision scores across 5 players"),
    ("team_wards_placed", "integer", "Detail", "Sum of wards placed"),
    ("team_damage", "integer", "Control", "Sum of damage dealt to champions"),
    # Ping breakdown
    ("team_total_pings", "integer", "Detail", "Total pings (all types) by team"),
    ("team_coord_pings", "integer", "Detail", "Coordinating pings (onMyWay + command + assistMe + push)"),
    ("team_allInPings", "integer", "Ping Detail", "All-in pings total"),
    ("team_assistMePings", "integer", "Ping Detail", "Assist-me pings total"),
    ("team_commandPings", "integer", "Ping Detail", "Command pings total"),
    ("team_dangerPings", "integer", "Ping Detail", "Danger pings total"),
    ("team_enemyMissingPings", "integer", "Ping Detail", "Enemy-missing pings total"),
    ("team_enemyVisionPings", "integer", "Ping Detail", "Enemy-vision pings total"),
    ("team_getBackPings", "integer", "Ping Detail", "Get-back pings total"),
    ("team_holdPings", "integer", "Ping Detail", "Hold pings total"),
    ("team_needVisionPings", "integer", "Ping Detail", "Need-vision pings total"),
    ("team_onMyWayPings", "integer", "Ping Detail", "On-my-way pings total"),
    ("team_pushPings", "integer", "Ping Detail", "Push pings total"),
    ("team_visionClearedPings", "integer", "Ping Detail", "Vision-cleared pings total"),
]

out_dir = Path(__file__).parent.parent / "Multi-Region Dataset"
out_dir.mkdir(exist_ok=True)

df = pd.DataFrame(CODEBOOK, columns=["Variable", "Type", "Role", "Description"])
df.to_excel(out_dir / "variable_codebook.xlsx", index=False)
print(f"Codebook saved: {out_dir / 'variable_codebook.xlsx'}  ({len(df)} variables)")
