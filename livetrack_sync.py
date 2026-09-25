"""Polls the kinetic-api livetrack feed for current vehicle positions.

    GET /fleet/trip/livetrack/{deviceId}?fromDate=<ms>&toDate=<ms>
    -> {"vehicleLiveTracks": [{latitude, longitude, speed, tsInMilliSeconds, ...}]}

The dashboard already calls this on demand, per event, to draw the GPS path
around a shock. This module does the same call on a timer with the Cognito
Bearer token attached, and keeps the newest fix per device in
`livetrack_latest.json` so a live position is always available without waiting
on the API.

Which devices get polled comes from `.env` (LIVETRACK_DEVICE_IDS); left empty,
the devices seen on the most recent events are used, so the poller follows
whatever the fleet is actually reporting.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

from auth import auth, authed_get
from event_sync import DATASETS, BASE_DIR, get_resilient_session

LIVETRACK_API_BASE = os.getenv(
    "LIVETRACK_API_BASE", "https://kinetic-api.dronaaim.ai/fleet/trip/livetrack"
).strip()

OUTPUT_FILE = os.path.join(
    BASE_DIR, os.getenv("LIVETRACK_OUTPUT_FILE", "livetrack_latest.json").strip()
)

# How far back each poll looks. The feed is queried by window, not "latest", so
# a window too short returns nothing at all for a parked or idle vehicle.
WINDOW_HOURS = float(os.getenv("LIVETRACK_WINDOW_HOURS", "24"))

# Cap on devices auto-discovered from recent events, so an unattended poller
# can't fan out into hundreds of requests every 10 seconds.
MAX_AUTO_DEVICES = int(os.getenv("LIVETRACK_MAX_DEVICES", "10"))

_SESSION = get_resilient_session()
_LOCK = threading.Lock()
_LATEST = {}


def configured_device_ids():
    raw = os.getenv("LIVETRACK_DEVICE_IDS", "").strip()
    if not raw:
        return []
    return [d.strip() for d in raw.split(",") if d.strip()]


def _devices_from_recent_events():
    """Falls back to the device IDs on the newest events of both datasets."""
    seen = []
    for cfg in DATASETS.values():
        summary_file = cfg["summary_file"]
        if not os.path.exists(summary_file):
            continue
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            continue
        # Records are stored newest-first, so a short prefix is the recent fleet.
        for record in records[:MAX_AUTO_DEVICES * 2]:
            device_id = record.get("deviceId")
            if device_id and device_id not in seen:
                seen.append(device_id)
            if len(seen) >= MAX_AUTO_DEVICES:
                return seen
    return seen


def resolve_device_ids():
    return configured_device_ids() or _devices_from_recent_events()


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def fetch_device(device_id, from_ms=None, to_ms=None):
    """One authenticated livetrack call. Returns the parsed position summary."""
    now_ms = int(time.time() * 1000)
    if to_ms is None:
        to_ms = now_ms
    if from_ms is None:
        from_ms = now_ms - int(WINDOW_HOURS * 3600 * 1000)

    entry = {
        "device_id": device_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "from_ms": from_ms,
        "to_ms": to_ms,
        "point_count": 0,
        "latest": None,
        "error": None,
    }

    try:
        response = authed_get(
            _SESSION,
            f"{LIVETRACK_API_BASE}/{device_id}",
            params={"fromDate": from_ms, "toDate": to_ms},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        points = payload.get("vehicleLiveTracks") or []
        entry["point_count"] = len(points)
        if points:
            newest = max(points, key=lambda p: p.get("tsInMilliSeconds") or 0)
            entry["latest"] = {
                "latitude": newest.get("latitude"),
                "longitude": newest.get("longitude"),
                "speed": newest.get("speed"),
                "heading": newest.get("heading"),
                "ts_ms": newest.get("tsInMilliSeconds"),
                "ts_iso": (
                    datetime.fromtimestamp(
                        newest["tsInMilliSeconds"] / 1000, tz=timezone.utc
                    ).isoformat()
                    if newest.get("tsInMilliSeconds")
                    else None
                ),
            }
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"

    return entry


def sync_livetrack():
    """One tick: every tracked device's newest position, written to disk.

    Never raises - a dead network for one tick should not stop the loop.
    """
    started = time.time()
    device_ids = resolve_device_ids()
    stats = {
        "devices": len(device_ids),
        "with_position": 0,
        "errors": 0,
        "auth": auth.status()["flow"],
        "source": "env" if configured_device_ids() else "recent-events",
    }

    results = {}
    for device_id in device_ids:
        entry = fetch_device(device_id)
        results[device_id] = entry
        if entry["error"]:
            stats["errors"] += 1
        elif entry["latest"]:
            stats["with_position"] += 1

    if results:
        with _LOCK:
            _LATEST.update(results)
            _atomic_write_json(
                OUTPUT_FILE,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "devices": _LATEST,
                },
            )

    stats["duration_s"] = round(time.time() - started, 2)
    return stats


def latest(device_id=None):
    with _LOCK:
        if device_id:
            return _LATEST.get(device_id)
        return dict(_LATEST)


if __name__ == "__main__":
    print("Devices:", resolve_device_ids())
    print(json.dumps(sync_livetrack(), indent=2))
    print(json.dumps(latest(), indent=2)[:1500])
