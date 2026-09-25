"""FastAPI service that keeps the event JSONs continuously fresh.

Every POLL_INTERVAL_SECONDS (10 by default) it:

  1. calls the kinetic-api event-with-gs feed for both decision=no and
     decision=yes (page 1, newest first - the same call main.py and
     main_decision_yes.py make by hand), downloads any new .gsdata file and folds
     the new events into events_gforce_summary.json / events_gforce_summary_yes.json;
  2. calls fleet/trip/livetrack for each tracked device with a Cognito Bearer
     token, and stores the newest lat/long in livetrack_latest.json.

view_gsensor.py --serve rebuilds its index from those files whenever they change,
so new events reach the dashboard on their own.

Credentials and device IDs come from .env - see .env.example.

Run:
    uvicorn sync_server:app --port 8010
    (or: python sync_server.py)
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import event_sync
import livetrack_sync
from auth import auth

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))

# On startup, compare the local record count against the API's totalRecords and
# reconcile the difference in the background. The 10s tail poll only ever looks
# at the newest few pages, so without this a dataset that fell behind (or was
# never fully backfilled) stays behind forever.
BACKFILL_ON_STARTUP = os.getenv("BACKFILL_ON_STARTUP", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Separately, some already-downloaded events are missing media/timestamp
# because their video hadn't finished processing upstream when first fetched
# (see is_fully_enriched's docstring). The 10s tail poll re-checks these
# automatically whenever they resurface on page 1, but an older event needs a
# full sweep to find - so this runs one at startup, then repeats periodically
# (media can keep lagging indefinitely, not just once), independent of backfill.
ENRICH_ON_STARTUP = os.getenv("ENRICH_ON_STARTUP", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ENRICH_RECHECK_INTERVAL_SECONDS = int(os.getenv("ENRICH_RECHECK_INTERVAL_SECONDS", "1800"))

# Last tick's outcome per dataset, plus counters - exposed at /status so the
# poller can be watched without reading the server log.
STATE = {
    "started_at": None,
    "poll_interval_seconds": POLL_INTERVAL_SECONDS,
    "ticks": 0,
    "last_tick_at": None,
    "last_error": None,
    "total_new_events": 0,
    "datasets": {key: None for key in event_sync.DATASETS},
    "livetrack": None,
    "backfill": {key: None for key in event_sync.DATASETS},
    "enrich": {key: None for key in event_sync.DATASETS},
}

# Guards against two backfills (or two enrich sweeps) of the same dataset
# running at once (startup task racing a manual POST, or a periodic recheck
# firing while the previous one is still running).
_backfill_tasks = {}
_enrich_tasks = {}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _record_results(results):
    STATE["ticks"] += 1
    STATE["last_tick_at"] = _now_iso()
    for stats in results:
        stats["at"] = STATE["last_tick_at"]
        STATE["datasets"][stats["dataset"]] = stats
        if stats["new_events"]:
            STATE["total_new_events"] += stats["new_events"]
        if stats["error"]:
            STATE["last_error"] = f"[{stats['dataset']}] {stats['error']}"


def _tick():
    """One full poll: both event datasets, then every tracked vehicle's position.

    Runs in a worker thread - these are blocking `requests` calls, one of which
    may download a multi-megabyte .gsdata file.
    """
    results = event_sync.sync_all()
    livetrack_stats = livetrack_sync.sync_livetrack()
    return results, livetrack_stats


async def _run_sync():
    """Runs one tick off the event loop and folds the outcome into STATE."""
    results, livetrack_stats = await asyncio.to_thread(_tick)
    _record_results(results)
    livetrack_stats["at"] = STATE["last_tick_at"]
    STATE["livetrack"] = livetrack_stats
    return results


async def _poll_loop():
    while True:
        tick_started = time.monotonic()
        try:
            results = await _run_sync()
            for stats in results:
                if stats["error"]:
                    print(f"[sync] {stats['dataset']}: ERROR {stats['error']}")
                elif stats["new_events"]:
                    print(
                        f"[sync] {stats['dataset']}: +{stats['new_events']} new event(s) "
                        f"-> {stats['total_events']} total "
                        f"(ids: {stats['new_event_ids']})"
                    )
            lt = STATE["livetrack"]
            if lt and lt["errors"]:
                print(f"[livetrack] {lt['errors']}/{lt['devices']} device(s) failed this tick")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            STATE["last_error"] = f"{type(e).__name__}: {e}"
            print(f"[sync] tick failed: {e}")

        # Interval measured from the start of the tick, so a slow download does
        # not push every later poll further behind.
        await asyncio.sleep(max(0, POLL_INTERVAL_SECONDS - (time.monotonic() - tick_started)))


async def _run_backfill(key):
    """Reconciles one dataset in a worker thread, streaming progress into STATE.

    Long-running by nature - a 1,400-event gap means 1,400 .gsdata downloads -
    so it is never awaited by a request handler or the poll loop.
    """
    def progress(stats):
        stats["at"] = _now_iso()
        STATE["backfill"][key] = stats

    try:
        result = await asyncio.to_thread(event_sync.backfill_dataset, key, progress)
        if result.get("error"):
            print(f"[backfill] {key}: ERROR {result['error']}")
        else:
            print(
                f"[backfill] {key}: +{result['added']} event(s) over "
                f"{result['rounds']} round(s) -> {result['local_total']}/{result['remote_total']} "
                f"({'complete' if result.get('complete') else 'still short'})"
            )
        return result
    finally:
        _backfill_tasks.pop(key, None)


def _start_backfill(key):
    """Spawns a backfill unless one is already running for this dataset."""
    existing = _backfill_tasks.get(key)
    if existing and not existing.done():
        return False
    _backfill_tasks[key] = asyncio.create_task(_run_backfill(key))
    return True


async def _startup_backfill():
    """Checks each dataset for a gap and reconciles only the ones that need it."""
    try:
        gaps = await asyncio.to_thread(event_sync.gap_report)
    except Exception as e:
        print(f"[backfill] gap check failed: {e}")
        return

    for key, entry in gaps.items():
        STATE["backfill"][key] = {**entry, "running": False, "added": 0}
        if entry.get("error"):
            print(f"[backfill] {key}: gap check error - {entry['error']}")
        elif entry["missing"] > 0:
            print(
                f"[backfill] {key}: local {entry['local_total']} vs API "
                f"{entry['remote_total']} - {entry['missing']} missing, reconciling..."
            )
            _start_backfill(key)
        else:
            print(f"[backfill] {key}: in sync ({entry['local_total']} events)")


async def _run_enrich(key):
    """Backfills timestamp/media for already-known-but-incomplete events.

    Same shape as _run_backfill(): a worker thread, streamed progress, never
    awaited by a request handler.
    """
    def progress(stats):
        stats["at"] = _now_iso()
        STATE["enrich"][key] = stats

    try:
        result = await asyncio.to_thread(event_sync.enrich_dataset, key, progress)
        if result.get("error"):
            print(f"[enrich] {key}: ERROR {result['error']}")
        elif result["enriched"] > 0 or not result.get("complete"):
            print(
                f"[enrich] {key}: filled in timestamp/media for {result['enriched']} event(s) "
                f"over {result['rounds']} round(s) - {result['still_unenriched']} still missing it "
                f"({'complete' if result.get('complete') else 'giving up for now, will retry later'})"
            )
        return result
    finally:
        _enrich_tasks.pop(key, None)


def _start_enrich(key):
    """Spawns an enrich sweep unless one is already running for this dataset."""
    existing = _enrich_tasks.get(key)
    if existing and not existing.done():
        return False
    _enrich_tasks[key] = asyncio.create_task(_run_enrich(key))
    return True


async def _startup_enrich():
    """Checks each dataset for events missing timestamp/media and fixes them."""
    for key in event_sync.DATASETS:
        try:
            gap = len(await asyncio.to_thread(event_sync.unenriched_ids, key))
        except Exception as e:
            print(f"[enrich] {key}: gap check failed - {e}")
            continue

        if gap > 0:
            print(f"[enrich] {key}: {gap} event(s) missing timestamp/media, enriching...")
            _start_enrich(key)
        else:
            print(f"[enrich] {key}: every local event already has timestamp/media.")


async def _enrich_recheck_loop():
    """Periodic safety net: media can keep lagging indefinitely upstream, not
    just at startup, so this doesn't just run once. Runs alongside the 10s tail
    poll, which already re-checks anything still on the newest pages - this
    catches whatever has since aged off them."""
    while True:
        await asyncio.sleep(ENRICH_RECHECK_INTERVAL_SECONDS)
        await _startup_enrich()


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["started_at"] = _now_iso()
    task = asyncio.create_task(_poll_loop())
    print(
        f"[sync] polling {event_sync.API_URL} every {POLL_INTERVAL_SECONDS}s "
        f"for decision=no and decision=yes"
    )
    print(f"[sync] livetrack devices: {livetrack_sync.resolve_device_ids() or '(none resolved yet)'}")
    status = auth.status()
    if status["enabled"]:
        print(f"[auth] flow={status['flow']} token_kind={status['token_kind']}")
    else:
        print("[auth] disabled - set the COGNITO_* values in .env to authenticate")

    backfill_task = asyncio.create_task(_startup_backfill()) if BACKFILL_ON_STARTUP else None
    if not BACKFILL_ON_STARTUP:
        print("[backfill] disabled (BACKFILL_ON_STARTUP=false) - POST /backfill to run one")

    enrich_task = asyncio.create_task(_startup_enrich()) if ENRICH_ON_STARTUP else None
    if not ENRICH_ON_STARTUP:
        print("[enrich] disabled (ENRICH_ON_STARTUP=false) - POST /enrich to run one")
    enrich_recheck_task = asyncio.create_task(_enrich_recheck_loop())

    try:
        yield
    finally:
        # Cancel the poll loop, the startup gap checks, and any backfill/enrich
        # sweep still in flight - fine to abandon mid-sweep, since the next
        # startup (or the periodic recheck) picks up where this left off.
        pending = [task, enrich_recheck_task]
        if backfill_task:
            pending.append(backfill_task)
        if enrich_task:
            pending.append(enrich_task)
        pending.extend(t for t in _backfill_tasks.values() if not t.done())
        pending.extend(t for t in _enrich_tasks.values() if not t.done())

        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        print("[sync] poller stopped.")


app = FastAPI(
    title="FNOL Event Sync",
    description="Polls the kinetic-api G-Sensor event feed and keeps the dashboard JSONs fresh.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "FNOL Event Sync",
        "polling": f"every {POLL_INTERVAL_SECONDS}s",
        "endpoints": [
            "/status",
            "/health",
            "/auth",
            "/events/{dataset}",
            "/livetrack",
            "/livetrack/{device_id}",
            "/gap",
            "POST /sync",
            "POST /backfill",
            "POST /enrich",
        ],
        "datasets": list(event_sync.DATASETS),
    }


@app.get("/auth")
def auth_status():
    """Auth state without exposing any token or credential."""
    return auth.status()


@app.get("/livetrack")
def livetrack_all():
    """Newest known position per tracked device, as of the last tick."""
    return {
        "devices": livetrack_sync.latest(),
        "tracked": livetrack_sync.resolve_device_ids(),
        "last_tick": STATE["livetrack"],
    }


@app.get("/livetrack/{device_id}")
def livetrack_device(device_id: str, refresh: bool = False):
    """One device's position. `?refresh=true` calls the API now instead of
    serving what the last tick stored."""
    if refresh:
        return livetrack_sync.fetch_device(device_id)
    entry = livetrack_sync.latest(device_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No position stored for {device_id} yet - try ?refresh=true",
        )
    return entry


@app.get("/health")
def health():
    return {"ok": STATE["last_error"] is None, "ticks": STATE["ticks"], "last_error": STATE["last_error"]}


@app.get("/status")
def status():
    """Everything the last tick did, per dataset."""
    return STATE


@app.get("/events/{dataset}")
def events(dataset: str, limit: int = 20):
    """The newest records as currently stored - handy for confirming the poller
    picked something up without opening the JSON file."""
    if dataset not in event_sync.DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")
    state = event_sync._STATES[dataset]
    state.ensure_loaded()
    return JSONResponse(
        {
            "dataset": dataset,
            "total": len(state.records),
            "events": state.records[:limit],
        }
    )


@app.get("/gap")
async def gap():
    """Local record count vs the API's totalRecords, per dataset.

    One tiny API call each - the response carries totalRecords directly, so this
    does not walk any pages.
    """
    return await asyncio.to_thread(event_sync.gap_report)


@app.post("/backfill")
async def backfill(dataset: str = None):
    """Starts a reconciliation sweep in the background and returns immediately.

    Progress shows up under /status -> backfill. Optionally scoped to one
    dataset; otherwise every dataset is reconciled.
    """
    keys = [dataset] if dataset else list(event_sync.DATASETS)
    for key in keys:
        if key not in event_sync.DATASETS:
            raise HTTPException(status_code=404, detail=f"Unknown dataset: {key}")

    started, already_running = [], []
    for key in keys:
        (started if _start_backfill(key) else already_running).append(key)

    return {
        "started": started,
        "already_running": already_running,
        "note": "Runs in the background - poll /status for progress.",
    }


@app.post("/enrich")
async def enrich(dataset: str = None):
    """Starts a timestamp/media reconciliation sweep in the background.

    Fixes already-downloaded events that are still missing media (and so have
    no timestamp) because their video hadn't finished processing upstream when
    first fetched. Progress shows up under /status -> enrich.
    """
    keys = [dataset] if dataset else list(event_sync.DATASETS)
    for key in keys:
        if key not in event_sync.DATASETS:
            raise HTTPException(status_code=404, detail=f"Unknown dataset: {key}")

    started, already_running = [], []
    for key in keys:
        (started if _start_enrich(key) else already_running).append(key)

    return {
        "started": started,
        "already_running": already_running,
        "note": "Runs in the background - poll /status for progress.",
    }


@app.post("/sync")
async def sync_now():
    """Forces a tick immediately instead of waiting for the timer."""
    results = await _run_sync()
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("SYNC_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SYNC_SERVER_PORT", "8010")),
    )
