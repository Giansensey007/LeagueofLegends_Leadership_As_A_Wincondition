#!/usr/bin/env python3
"""
=============================================================================
  Data Collection — Riot Games Match-V5 API  (Multi-Region, Stratified)
  "Does It Pay to Play Nice?"
  Applied Sports Research — League of Legends

  Collects ranked solo/duo match data from four regions with STRATIFIED
  quarterly time-windows so the final dataset is roughly uniform over the
  past year (Mar 2025 – Feb 2026).

  Supported regions:
      EUW  (euw1  / europe)
      NA   (na1   / americas)
      KR   (kr    / asia)
      VN   (vn2   / sea)

  Usage:
      python collect_matches.py                       # all regions
      python collect_matches.py --region EUW          # single region
      python collect_matches.py --region EUW NA       # specific regions
      python collect_matches.py --target 5000         # custom per-region target

  The script is fully resumable: per-region checkpoints track every phase.
=============================================================================
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import requests
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────

# Riot Games API Key — registered app key (persistent)
API_KEY = os.environ.get(
    "RIOT_API_KEY",
    "RGAPI-64bdc0b2-c216-4bc0-8333-92db862b9cb7",
)

# Region definitions: { code: (platform_url, regional_url) }
REGIONS = {
    "EUW": ("https://euw1.api.riotgames.com", "https://europe.api.riotgames.com"),
    "NA":  ("https://na1.api.riotgames.com",  "https://americas.api.riotgames.com"),
    "KR":  ("https://kr.api.riotgames.com",   "https://asia.api.riotgames.com"),
    "VN":  ("https://vn2.api.riotgames.com",  "https://sea.api.riotgames.com"),
}

QUEUE = "RANKED_SOLO_5x5"
TIERS = ["GOLD", "PLATINUM", "EMERALD"]
DIVISIONS = ["I", "II", "III", "IV"]

TARGET_PER_REGION = 2500           # matches per region  (total ≈ 10 000)
SUMMONERS_PER_DIVISION = 15        # pages of league entries to sample
MIN_GAME_DURATION = 900            # exclude remakes (< 15 min)

# ── Time window & quarterly buckets for stratified sampling ──────────────
TIME_START = datetime(2025, 3, 1, tzinfo=timezone.utc)
TIME_END   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

QUARTERS = [
    ("Q1_2025", datetime(2025, 3,  1, tzinfo=timezone.utc),
                datetime(2025, 5, 31, 23, 59, 59, tzinfo=timezone.utc)),
    ("Q2_2025", datetime(2025, 6,  1, tzinfo=timezone.utc),
                datetime(2025, 8, 31, 23, 59, 59, tzinfo=timezone.utc)),
    ("Q3_2025", datetime(2025, 9,  1, tzinfo=timezone.utc),
                datetime(2025, 11, 30, 23, 59, 59, tzinfo=timezone.utc)),
    ("Q4_2025", datetime(2025, 12, 1, tzinfo=timezone.utc),
                datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)),
]

# ── Rate limiting — Riot: 20 req/1 s, 100 req/2 min ─────────────────────
RATE_SHORT_LIMIT = 20
RATE_LONG_LIMIT  = 100
RATE_LONG_WINDOW = 120   # seconds

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "raw_matches"
RAW_DIR.mkdir(exist_ok=True)

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect")

# ─── Rate Limiter ───────────────────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter for Riot API."""

    def __init__(self):
        self.timestamps: list[float] = []

    def wait_if_needed(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < RATE_LONG_WINDOW]

        # 2-min window
        if len(self.timestamps) >= RATE_LONG_LIMIT:
            sleep_until = self.timestamps[0] + RATE_LONG_WINDOW
            wait = sleep_until - now + 0.5
            if wait > 0:
                log.info(f"Rate limit (2-min): sleeping {wait:.1f}s ...")
                time.sleep(wait)

        # 1-sec burst
        recent = [t for t in self.timestamps if now - t < 1.0]
        if len(recent) >= RATE_SHORT_LIMIT:
            time.sleep(1.05)

        self.timestamps.append(time.time())


limiter = RateLimiter()

# ─── API Helpers ────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> dict | list | None:
    """GET with rate limiting, retry on 429, and error handling."""
    headers = {"X-Riot-Token": API_KEY}
    for attempt in range(5):
        limiter.wait_if_needed()
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.RequestException as e:
            log.warning(f"Request error: {e}. Retrying in 5s ...")
            time.sleep(5)
            continue

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            log.warning(f"429 rate limited. Waiting {retry_after}s ...")
            time.sleep(retry_after + 1)
        elif resp.status_code == 403:
            log.error("403 Forbidden — API key is invalid or expired.")
            sys.exit(1)
        elif resp.status_code == 404:
            return None
        else:
            log.warning(f"HTTP {resp.status_code} for {url}. Retrying ...")
            time.sleep(3)
    log.error(f"Failed after 5 attempts: {url}")
    return None


def get_league_entries(platform_url: str, tier: str, division: str,
                       page: int = 1) -> list[dict]:
    url = f"{platform_url}/lol/league/v4/entries/{QUEUE}/{tier}/{division}"
    result = _get(url, params={"page": page})
    return result if result else []


def get_match_ids(regional_url: str, puuid: str,
                  start_epoch: int, end_epoch: int,
                  count: int = 100) -> list[str]:
    url = f"{regional_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    result = _get(url, params={
        "queue": 420,
        "type": "ranked",
        "startTime": start_epoch,
        "endTime": end_epoch,
        "count": count,
    })
    return result if result else []


def get_match(regional_url: str, match_id: str) -> dict | None:
    url = f"{regional_url}/lol/match/v5/matches/{match_id}"
    return _get(url)

# ─── Checkpoint ─────────────────────────────────────────────────────────────

def _cp_path(region: str) -> Path:
    return BASE_DIR / f"checkpoint_{region}.json"


def load_checkpoint(region: str) -> dict:
    p = _cp_path(region)
    if p.exists():
        return json.loads(p.read_text())
    return {
        "puuids": [],
        "match_ids_by_quarter": {},
        "matches_fetched_by_quarter": {},
    }


def save_checkpoint(region: str, cp: dict):
    _cp_path(region).write_text(json.dumps(cp, indent=2))

# ─── Phase 1: Collect PUUIDs ───────────────────────────────────────────────

def collect_puuids(platform_url: str, region: str, cp: dict) -> list[str]:
    """Gather PUUIDs from league entries (Gold / Plat / Emerald)."""
    existing = set(cp.get("puuids", []))
    if len(existing) >= 200:
        log.info(f"[{region}] Phase 1 — {len(existing)} PUUIDs cached, skipping.")
        return list(existing)

    log.info(f"[{region}] Phase 1 — Collecting PUUIDs ...")
    puuids = set(existing)

    for tier in TIERS:
        for division in DIVISIONS:
            for page in range(1, SUMMONERS_PER_DIVISION + 1):
                entries = get_league_entries(platform_url, tier, division, page)
                if not entries:
                    break
                for entry in entries:
                    puuid = entry.get("puuid")
                    if puuid and puuid not in puuids:
                        puuids.add(puuid)
                if len(puuids) % 50 == 0:
                    cp["puuids"] = list(puuids)
                    save_checkpoint(region, cp)
                log.info(f"  [{region}] {tier} {division} p{page}: "
                         f"{len(puuids)} PUUIDs")
                if len(puuids) >= 400:
                    break
            if len(puuids) >= 400:
                break
        if len(puuids) >= 400:
            break

    cp["puuids"] = list(puuids)
    save_checkpoint(region, cp)
    log.info(f"[{region}] Phase 1 done — {len(puuids)} PUUIDs.")
    return list(puuids)

# ─── Phase 2: Collect match IDs (stratified by quarter) ────────────────────

def collect_match_ids_stratified(
    regional_url: str,
    puuids: list[str],
    region: str,
    cp: dict,
    target: int,
) -> dict[str, list[str]]:
    """For each quarterly window, collect match IDs from sampled players."""
    # Over-sample by 2× to account for invalid-match filtering
    target_ids_per_q = (target * 2) // len(QUARTERS)

    match_ids_by_q: dict[str, list[str]] = cp.get("match_ids_by_quarter", {})

    for q_name, q_start, q_end in QUARTERS:
        existing = set(match_ids_by_q.get(q_name, []))
        if len(existing) >= target_ids_per_q:
            log.info(f"[{region}] Phase 2 — {q_name}: {len(existing)} IDs "
                     f"(need {target_ids_per_q}), skipping.")
            continue

        log.info(f"[{region}] Phase 2 — {q_name}: collecting match IDs "
                 f"({len(existing)}/{target_ids_per_q}) ...")

        start_epoch = int(q_start.timestamp())
        end_epoch   = int(q_end.timestamp())

        shuffled = list(puuids)
        random.shuffle(shuffled)

        for puuid in tqdm(shuffled, desc=f"[{region}] {q_name} match-lists"):
            ids = get_match_ids(regional_url, puuid, start_epoch, end_epoch)
            for mid in ids:
                existing.add(mid)

            # periodic checkpoint
            if len(existing) % 200 == 0:
                match_ids_by_q[q_name] = list(existing)
                cp["match_ids_by_quarter"] = match_ids_by_q
                save_checkpoint(region, cp)

            if len(existing) >= target_ids_per_q:
                break

        match_ids_by_q[q_name] = list(existing)
        cp["match_ids_by_quarter"] = match_ids_by_q
        save_checkpoint(region, cp)
        log.info(f"[{region}] Phase 2 — {q_name}: {len(existing)} match IDs.")

    return match_ids_by_q

# ─── Phase 3: Fetch & validate match JSONs (stratified) ───────────────────

def is_valid_match(data: dict) -> bool:
    """Ranked 5v5, ≥ 15 min, no remakes, no early disconnects."""
    info = data.get("info", {})
    if info.get("queueId") != 420:
        return False
    if info.get("gameDuration", 0) < MIN_GAME_DURATION:
        return False
    if info.get("gameEndedInEarlySurrender", False):
        return False
    for p in info.get("participants", []):
        if p.get("timePlayed", 9999) < 300:
            return False
    return True


def fetch_matches_stratified(
    regional_url: str,
    match_ids_by_q: dict[str, list[str]],
    region: str,
    cp: dict,
    target: int,
) -> int:
    """Download match JSONs with equal quota per quarter."""
    region_dir = RAW_DIR / region
    region_dir.mkdir(exist_ok=True)

    target_per_q = target // len(QUARTERS)
    remainder    = target % len(QUARTERS)

    fetched_by_q: dict[str, list[str]] = cp.get("matches_fetched_by_quarter", {})
    total_valid = 0

    for i, (q_name, _, _) in enumerate(QUARTERS):
        q_target = target_per_q + (1 if i < remainder else 0)
        already = set(fetched_by_q.get(q_name, []))
        valid_count = len([m for m in already
                           if (region_dir / f"{m}.json").exists()])

        if valid_count >= q_target:
            log.info(f"[{region}] Phase 3 — {q_name}: "
                     f"{valid_count}/{q_target} done, skipping.")
            total_valid += valid_count
            continue

        candidates = match_ids_by_q.get(q_name, [])
        to_fetch = [m for m in candidates if m not in already]
        random.shuffle(to_fetch)

        log.info(f"[{region}] Phase 3 — {q_name}: fetching "
                 f"({valid_count}/{q_target}, {len(to_fetch)} candidates) ...")

        for mid in tqdm(to_fetch, desc=f"[{region}] {q_name} fetch"):
            if valid_count >= q_target:
                break

            out_path = region_dir / f"{mid}.json"
            if out_path.exists():
                already.add(mid)
                valid_count += 1
                continue

            data = get_match(regional_url, mid)
            if data is None:
                already.add(mid)
                continue

            if not is_valid_match(data):
                already.add(mid)
                continue

            out_path.write_text(json.dumps(data, indent=2))
            already.add(mid)
            valid_count += 1

            if valid_count % 25 == 0:
                fetched_by_q[q_name] = list(already)
                cp["matches_fetched_by_quarter"] = fetched_by_q
                save_checkpoint(region, cp)
                cur_total = sum(
                    len([m for m in fetched_by_q.get(q, [])
                         if (region_dir / f"{m}.json").exists()])
                    for q, _, _ in QUARTERS
                )
                log.info(f"  [{region}] {q_name}: {valid_count}/{q_target} | "
                         f"Region total ≈ {cur_total}/{target}")

        fetched_by_q[q_name] = list(already)
        cp["matches_fetched_by_quarter"] = fetched_by_q
        save_checkpoint(region, cp)
        total_valid += valid_count

    log.info(f"[{region}] Phase 3 done — {total_valid} valid matches "
             f"in {region_dir}/")
    return total_valid

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect LoL ranked match data (multi-region, stratified)")
    parser.add_argument(
        "--region", nargs="*", default=list(REGIONS.keys()),
        choices=list(REGIONS.keys()),
        help="Region(s) to collect. Default: all four.")
    parser.add_argument(
        "--target", type=int, default=TARGET_PER_REGION,
        help=f"Matches per region (default {TARGET_PER_REGION}).")
    args = parser.parse_args()

    if not API_KEY:
        log.error(
            "No API key found.\n"
            "  Set RIOT_API_KEY env var or hardcode in this script.\n")
        sys.exit(1)

    regions = args.region
    target  = args.target

    log.info("=" * 64)
    log.info("  LoL Data Collection — Multi-Region Stratified Sampling")
    log.info(f"  Regions:  {regions}")
    log.info(f"  Target:   {target} matches/region  "
             f"({target * len(regions)} total)")
    log.info(f"  Window:   {TIME_START.date()} → {TIME_END.date()}")
    log.info(f"  Quarters: {[q[0] for q in QUARTERS]}")
    log.info(f"  Tiers:    {TIERS}")
    log.info("=" * 64)

    for region in regions:
        platform_url, regional_url = REGIONS[region]

        log.info(f"\n{'─'*64}")
        log.info(f"  ▶ Region: {region}")
        log.info(f"{'─'*64}")

        cp = load_checkpoint(region)

        # Phase 1 — PUUIDs
        puuids = collect_puuids(platform_url, region, cp)

        # Phase 2 — Match IDs (stratified by quarter)
        match_ids_by_q = collect_match_ids_stratified(
            regional_url, puuids, region, cp, target)

        # Phase 3 — Fetch & save valid matches
        total = fetch_matches_stratified(
            regional_url, match_ids_by_q, region, cp, target)

        log.info(f"[{region}] ✓ {total} matches saved to "
                 f"{RAW_DIR / region}/")

    log.info("\n" + "=" * 64)
    log.info("  All regions complete.")
    log.info("  Next step:  python build_variables.py")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
