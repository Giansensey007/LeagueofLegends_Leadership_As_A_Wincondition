#!/usr/bin/env python3
"""
Generate 10 synthetic match JSON files that mirror the Riot Match-V5 API format.
Used to validate the build_variables.py pipeline without needing an API key.
"""

import json
import random
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_matches"
RAW_DIR.mkdir(exist_ok=True)

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
PING_FIELDS = [
    "allInPings", "assistMePings", "commandPings", "dangerPings",
    "enemyMissingPings", "enemyVisionPings", "getBackPings", "holdPings",
    "needVisionPings", "onMyWayPings", "pushPings", "visionClearedPings",
]

def make_participant(team_id: int, position: str, winning: bool) -> dict:
    """Generate a realistic participant stat block."""
    # Winners tend to have better stats
    bonus = 1.2 if winning else 0.85

    kills = max(0, int(random.gauss(5 * bonus, 3)))
    deaths = max(0, int(random.gauss(5 / bonus, 2.5)))
    assists = max(0, int(random.gauss(7 * bonus, 4)))

    cs = int(random.gauss(180, 40))
    neutral = int(random.gauss(30 if position == "JUNGLE" else 10, 8))
    gold = int((kills * 300 + assists * 150 + cs * 20) * random.uniform(0.8, 1.2))

    vision = int(random.gauss(25 if position == "UTILITY" else 12, 6))
    wards = int(random.gauss(12 if position == "UTILITY" else 5, 3))

    # Pings — shot-caller has more coordinating pings
    is_shotcaller = position in ("JUNGLE", "UTILITY") and random.random() > 0.5
    pings = {}
    for field in PING_FIELDS:
        if field in ("onMyWayPings", "commandPings", "assistMePings", "pushPings"):
            base = 15 if is_shotcaller else 5
        else:
            base = 8
        pings[field] = max(0, int(random.gauss(base, base * 0.5)))

    damage = int(random.gauss(18000 * bonus, 5000))
    magic_pct = random.uniform(0.2, 0.8)

    return {
        "teamId": team_id,
        "teamPosition": position,
        "win": winning,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "totalMinionsKilled": max(0, cs),
        "neutralMinionsKilled": max(0, neutral),
        "goldEarned": max(0, gold),
        "visionScore": max(0, vision),
        "wardsPlaced": max(0, wards),
        "totalDamageDealtToChampions": max(0, damage),
        "magicDamageDealtToChampions": max(0, int(damage * magic_pct)),
        "physicalDamageDealtToChampions": max(0, int(damage * (1 - magic_pct))),
        "totalDamageTaken": max(0, int(random.gauss(22000, 6000))),
        "timePlayed": 1800,
        **pings,
    }


def make_team_objectives(winning: bool) -> dict:
    bonus = 1.3 if winning else 0.7
    return {
        "objectives": {
            "dragon": {"kills": int(random.gauss(2.5 * bonus, 1)), "first": random.random() > 0.5},
            "baron": {"kills": 1 if (winning and random.random() > 0.4) else 0, "first": winning},
            "riftHerald": {"kills": int(random.gauss(1 * bonus, 0.5)), "first": random.random() > 0.5},
            "tower": {"kills": int(random.gauss(6 * bonus, 2)), "first": random.random() > 0.5},
            "champion": {"kills": 0, "first": random.random() > 0.5},
        }
    }


def make_match(match_idx: int) -> dict:
    match_id = f"EUW1_TEST_{match_idx:04d}"
    game_duration = int(random.gauss(1800, 300))  # ~30 min avg
    game_duration = max(960, game_duration)  # at least 16 min

    surrender = random.random() < 0.2
    blue_wins = random.random() > 0.5

    participants = []
    for pos in POSITIONS:
        participants.append(make_participant(100, pos, blue_wins))
        participants.append(make_participant(200, pos, not blue_wins))

    # Ensure teamId grouping is correct (API returns interleaved)
    random.shuffle(participants)

    teams = [
        {"teamId": 100, **make_team_objectives(blue_wins)},
        {"teamId": 200, **make_team_objectives(not blue_wins)},
    ]

    return {
        "metadata": {
            "matchId": match_id,
            "participants": [f"puuid_{i}" for i in range(10)],
        },
        "info": {
            "gameId": 7000000000 + match_idx,
            "queueId": 420,
            "gameDuration": game_duration,
            "gameVersion": "14.3.1",
            "gameEndedInSurrender": surrender,
            "gameEndedInEarlySurrender": False,
            "participants": participants,
            "teams": teams,
        },
    }


def main():
    print(f"Generating 10 test matches in {RAW_DIR}/ ...")
    for i in range(10):
        match = make_match(i)
        path = RAW_DIR / f"{match['metadata']['matchId']}.json"
        path.write_text(json.dumps(match, indent=2))
        print(f"  ✓ {path.name}")

    print(f"\nDone. Now run:  python build_variables.py")


if __name__ == "__main__":
    main()
