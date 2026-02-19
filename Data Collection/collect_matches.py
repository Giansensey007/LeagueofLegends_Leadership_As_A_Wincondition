#!/usr/bin/env python3
"""
=============================================================================
  Data Collection — Riot Games Match-V5 API
  "Does It Pay to Play Nice?"
  Applied Sports Research — League of Legends

  Collects ranked solo/duo match data from EUW and saves raw JSON per match.
  A separate script (build_variables.py) processes these into study variables.

  Usage:
      export RIOT_API_KEY="RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
      python collect_matches.py

  The script is resumable: it checkpoints progress and skips already-fetched
  matches on re-run.
=============================================================================
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import requests
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────

API_KEY = os.environ.get("RIOT_API_KEY", "")

PLATFORM_URL = "https://euw1.api.riotgames.com"   # league-v4, summoner-v4
REGIONAL_URL = "https://europe.api.riotgames.com"  # match-v5

QUEUE = "RANKED_SOLO_5x5"
TIERS = ["GOLD", "PLATINUM", "EMERALD"]
DIVISIONS = ["I", "II", "III", "IV"]

TARGET_MATCHES = 2000           # total unique matches to collect
MATCHES_PER_SUMMONER = 20       # recent ranked matches to pull per player
SUMMONERS_PER_DIVISION = 15     # pages of league entries to sample from
MIN_GAME_DURATION = 900         # exclude remakes (< 15 min = 900 sec)

# Rate limiting — Riot free-tier: 20/1s, 100/2min
RATE_SHORT_LIMIT = 20           # requests per 1-second window
RATE_LONG_LIMIT = 95            # stay under 100 per 2-min window (safety margin)
RATE_LONG_WINDOW = 120          # 2-minute window in seconds

# Paths
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw_matches"
CHECKPOINT_FILE = BASE_DIR / "checkpoint.json"

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
        # purge old timestamps
        self.timestamps = [t for t in self.timestamps if now - t < RATE_LONG_WINDOW]

        # check 2-min window
        if len(self.timestamps) >= RATE_LONG_LIMIT:
            sleep_until = self.timestamps[0] + RATE_LONG_WINDOW
            wait = sleep_until - now + 0.5
            if wait > 0:
                log.info(f"Rate limit (2-min window): sleeping {wait:.1f}s ...")
                time.sleep(wait)

        # check 1-sec burst
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


def get_league_entries(tier: str, division: str, page: int = 1) -> list[dict]:
    """Fetch a page of league entries for a tier/division."""
    url = f"{PLATFORM_URL}/lol/league/v4/entries/{QUEUE}/{tier}/{division}"
    result = _get(url, params={"page": page})
    return result if result else []


def get_puuid(summoner_id: str) -> str | None:
    """Resolve a summoner ID to a PUUID."""
    url = f"{PLATFORM_URL}/lol/summoner/v4/summoners/{summoner_id}"
    data = _get(url)
    return data["puuid"] if data else None


def get_match_ids(puuid: str, count: int = 20) -> list[str]:
    """Fetch recent ranked match IDs for a PUUID."""
    url = f"{REGIONAL_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    result = _get(url, params={"queue": 420, "type": "ranked", "count": count})
    return result if result else []


def get_match(match_id: str) -> dict | None:
    """Fetch full match data."""
    url = f"{REGIONAL_URL}/lol/match/v5/matches/{match_id}"
    return _get(url)

# ─── Checkpoint ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"puuids": [], "match_ids_seen": [], "matches_fetched": []}


def save_checkpoint(cp: dict):
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))

# ─── Phase 1: Collect summoner PUUIDs ───────────────────────────────────────

def collect_puuids(cp: dict) -> list[str]:
    """Gather PUUIDs from league entries across target tiers."""
    existing = set(cp.get("puuids", []))
    if len(existing) >= 200:
        log.info(f"Phase 1 — Already have {len(existing)} PUUIDs, skipping.")
        return list(existing)

    log.info("Phase 1 — Collecting summoner PUUIDs from league entries ...")
    puuids = set(existing)

    for tier in TIERS:
        for division in DIVISIONS:
            for page in range(1, SUMMONERS_PER_DIVISION + 1):
                entries = get_league_entries(tier, division, page)
                if not entries:
                    break
                for entry in entries:
                    sid = entry.get("summonerId")
                    if sid:
                        puuid = get_puuid(sid)
                        if puuid and puuid not in puuids:
                            puuids.add(puuid)
                # save progress periodically
                if len(puuids) % 50 == 0:
                    cp["puuids"] = list(puuids)
                    save_checkpoint(cp)

                log.info(f"  {tier} {division} p{page}: {len(puuids)} PUUIDs total")

                # stop early if we have plenty
                if len(puuids) >= 300:
                    break
            if len(puuids) >= 300:
                break
        if len(puuids) >= 300:
            break

    cp["puuids"] = list(puuids)
    save_checkpoint(cp)
    log.info(f"Phase 1 complete — {len(puuids)} PUUIDs collected.")
    return list(puuids)

# ─── Phase 2: Collect match IDs ────────────────────────────────────────────

def collect_match_ids(puuids: list[str], cp: dict) -> list[str]:
    """Fetch recent ranked match IDs from sampled summoners."""
    seen = set(cp.get("match_ids_seen", []))
    if len(seen) >= TARGET_MATCHES:
        log.info(f"Phase 2 — Already have {len(seen)} match IDs, skipping.")
        return list(seen)

    log.info("Phase 2 — Collecting match IDs ...")

    for puuid in tqdm(puuids, desc="Fetching match lists"):
        ids = get_match_ids(puuid, count=MATCHES_PER_SUMMONER)
        for mid in ids:
            seen.add(mid)
        # checkpoint every 20 summoners
        if len(seen) % 200 == 0:
            cp["match_ids_seen"] = list(seen)
            save_checkpoint(cp)

        if len(seen) >= TARGET_MATCHES * 2:
            # overshoot to allow filtering later
            break

    cp["match_ids_seen"] = list(seen)
    save_checkpoint(cp)
    log.info(f"Phase 2 complete — {len(seen)} unique match IDs.")
    return list(seen)

# ─── Phase 3: Fetch match data ─────────────────────────────────────────────

def is_valid_match(data: dict) -> bool:
    """Check exclusion criteria: ranked 5v5, >= 15 min, no remakes."""
    info = data.get("info", {})
    if info.get("queueId") != 420:
        return False
    if info.get("gameDuration", 0) < MIN_GAME_DURATION:
        return False
    if info.get("gameEndedInEarlySurrender", False):
        return False
    # check for early disconnects (any player with < 5 min play time)
    for p in info.get("participants", []):
        if p.get("timePlayed", 9999) < 300:
            return False
    return True


def fetch_matches(match_ids: list[str], cp: dict) -> int:
    """Download full match JSON for each match ID. Returns count of valid matches."""
    already_fetched = set(cp.get("matches_fetched", []))
    valid_count = len(already_fetched)

    # filter out already fetched
    to_fetch = [mid for mid in match_ids if mid not in already_fetched]
    log.info(f"Phase 3 — Fetching match details: {len(to_fetch)} remaining, "
             f"{valid_count} already done.")

    for mid in tqdm(to_fetch, desc="Fetching matches"):
        if valid_count >= TARGET_MATCHES:
            break

        out_path = RAW_DIR / f"{mid}.json"
        if out_path.exists():
            already_fetched.add(mid)
            valid_count += 1
            continue

        data = get_match(mid)
        if data is None:
            continue

        if not is_valid_match(data):
            already_fetched.add(mid)  # mark so we don't retry
            continue

        # save raw JSON
        out_path.write_text(json.dumps(data, indent=2))
        already_fetched.add(mid)
        valid_count += 1

        # checkpoint every 50 matches
        if valid_count % 50 == 0:
            cp["matches_fetched"] = list(already_fetched)
            save_checkpoint(cp)
            log.info(f"  ... {valid_count}/{TARGET_MATCHES} valid matches saved.")

    cp["matches_fetched"] = list(already_fetched)
    save_checkpoint(cp)
    log.info(f"Phase 3 complete — {valid_count} valid matches saved to {RAW_DIR}/")
    return valid_count

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        log.error(
            "No API key found.\n\n"
            "  1. Go to https://developer.riotgames.com and sign in.\n"
            "  2. Copy your Development API Key.\n"
            "  3. Run:  export RIOT_API_KEY=\"RGAPI-...\"\n"
            "  4. Then re-run this script.\n"
        )
        sys.exit(1)

    log.info("=" * 60)
    log.info("  LoL Data Collection — Does It Pay to Play Nice?")
    log.info(f"  Target: {TARGET_MATCHES} ranked matches, EUW, {TIERS}")
    log.info(f"  Output: {RAW_DIR}/")
    log.info("=" * 60)

    cp = load_checkpoint()

    # Phase 1: PUUIDs
    puuids = collect_puuids(cp)

    # Phase 2: Match IDs
    match_ids = collect_match_ids(puuids, cp)

    # Phase 3: Fetch & save valid matches
    total = fetch_matches(match_ids, cp)

    log.info("=" * 60)
    log.info(f"  Done. {total} raw match files in {RAW_DIR}/")
    log.info(f"  Next step: python build_variables.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
