"""Shared extraction core for the kinetic-api G-Sensor event feed.

main.py (decision=no) and main_decision_yes.py (decision=yes) are two copies of
the same pipeline, each run by hand. That logic lives here once so the polling
server (sync_server.py) can run both datasets on a timer and keep the summary
JSONs - the files view_gsensor.py builds its dashboard index from - continuously
fresh, without anyone re-running a script.

Nothing here changes the shape of the records written: the JSON stays exactly
what main.py produced, so view_gsensor.py keeps working unchanged.
"""

import json
import os
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from auth import authed_get

# Paths are anchored to this file, not the current working directory, so the
# server writes to the repo's JSONs no matter where uvicorn is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The event feed currently answers unauthenticated, so the Bearer token is only
# attached when .env asks for it - switching it on by mistake would otherwise
# break a working call with a token the endpoint never wanted.
APPLY_AUTH_TO_EVENT_API = os.getenv("APPLY_AUTH_TO_EVENT_API", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

API_BASE_URL = "https://kinetic-api.dronaaim.ai"
FILE_BASE_URL = "https://sv.smartwitness.co"  # SmartWitness S3-backed CDN (actual file server)
API_URL = f"{API_BASE_URL}/ai-model/event-with-gs"

LIMIT = 10
SORT_KEY = "dateTime"
SORT_ORDER = "DESC"

# Newest-first polling: page 1 alone covers a normal tick. Deeper pages are only
# walked while a page keeps yielding unseen events, so a burst that lands more
# than LIMIT events between ticks is still caught - without ever re-walking the
# whole archive the way main.py does.
#
# NOTE: this tail poll is a best-effort "catch what just arrived" pass, NOT a
# guarantee of completeness - see backfill_dataset() for why that needs its own
# strategy.
MAX_POLL_PAGES = 5

# --- Backfill -------------------------------------------------------------
#
# The API's pagination is non-deterministic: sortKey/sortOrder are ignored
# (every key tested comes back unordered, ASC and DESC return the same first
# record), and two identical requests return different pages - fetching pages
# 1-3 twice yields 276 unique IDs out of 300, i.e. records drift between pages
# between calls. Walking 1..N once therefore silently misses records, which is
# how the local copy ended up ~45% short of the API's totalRecords.
#
# Within a single snapshot, though, pages do NOT overlap. So the strategy is to
# make enumeration as close to instantaneous as possible, then do the slow work
# separately:
#
#   phase 1  walk every page collecting metadata only, no downloads (~33 fast
#            requests at limit=100) - little time for the order to drift
#   phase 2  download .gsdata for whatever phase 1 found missing, at leisure
#   repeat   re-run phase 1 (cheap) until the local count reaches totalRecords
#            or a round finds nothing new
BACKFILL_PAGE_SIZE = int(os.getenv("BACKFILL_PAGE_SIZE", "100"))

# Rounds of enumerate+ingest before giving up on converging. Each round after
# the first exists purely to catch records that shifted pages mid-sweep - and
# because a single sweep only sees ~87% of records (measured), several rounds
# are routinely needed even when nothing is actually wrong.
BACKFILL_MAX_ROUNDS = int(os.getenv("BACKFILL_MAX_ROUNDS", "15"))

# A sweep finding zero *new* IDs does NOT mean the dataset is complete - it may
# just be this sweep's ~87% coverage missing the same records again. Only give
# up early (before remote_total is reached) after this many CONSECUTIVE
# fruitless sweeps, so a run of bad luck doesn't look like convergence.
BACKFILL_STAGNANT_LIMIT = int(os.getenv("BACKFILL_STAGNANT_LIMIT", "4"))

# Events ingested between releases of the dataset lock. Keeps the 10s tail poll
# from being starved for the whole (potentially very long) backfill.
BACKFILL_BATCH_SIZE = int(os.getenv("BACKFILL_BATCH_SIZE", "25"))

# The same two datasets view_gsensor.py serves, pointed at the same folders and
# summary files, so the poller feeds exactly the dashboard's No Crash / Crash
# dropdown.
DATASETS = {
    "no_crash": {
        "decision": "no",
        "download_dir": os.path.join(BASE_DIR, "downloads_gsensor"),
        "summary_file": os.path.join(BASE_DIR, "events_gforce_summary.json"),
    },
    "crash": {
        "decision": "yes",
        "download_dir": os.path.join(BASE_DIR, "downloads_gsensor_yes"),
        "summary_file": os.path.join(BASE_DIR, "events_gforce_summary_yes.json"),
    },
}

# A record is "fully enriched" once it carries all of these.
REQUIRED_META_FIELDS = ("imei", "deviceId", "media")


def get_resilient_session():
    """Creates a requests session configured to auto-retry on connection issues."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,  # 2s, 4s, 8s, 16s, 32s backoff delays
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = get_resilient_session()


def build_media_details(item):
    """Full per-media metadata straight from the API (Speed, Camera, Heading,
    Altitude, Latitude, Longitude, FileFormat, ContentSize, Start/EndDateTime,
    URL, MDTURL, ...), plus convenience absolute URLs.
    """
    details = []
    for media in item.get("media", []):
        entry = dict(media)
        url = media.get("URL")
        if url:
            entry["full_url"] = f"{FILE_BASE_URL}{url}" if url.startswith("/") else url
        mdt_url = media.get("MDTURL")
        if mdt_url:
            entry["full_mdt_url"] = (
                f"{FILE_BASE_URL}{mdt_url}" if mdt_url.startswith("/") else mdt_url
            )
        details.append(entry)
    return details


def is_fully_enriched(record):
    """Whether a record has every field this sync adds - `media` checked for
    non-emptiness, not just presence.

    An event's video/GPS media is sometimes still processing upstream (at
    SmartWitness) when this sync first sees the event, so `media: []` at
    ingest time is NOT necessarily final - unlike `imei`/`deviceId`, which the
    API always returns from the first fetch. Timestamp data lives ONLY in
    `media[0].StartDateTime` (the event object itself carries no other
    timestamp field), so treating an empty list as "done forever" is exactly
    what leaves an event with no timestamp permanently: confirmed against the
    live API, every empty-media record checked (323/323) already has media
    available on a fresh call - our sync just never looked again.
    """
    return (
        "imei" in record
        and "deviceId" in record
        and bool(record.get("media"))
    )


def download_file(session, url_path, local_path):
    """Downloads a binary file safely, skipping if it already exists.

    Files are fetched from FILE_BASE_URL (sv.smartwitness.co), NOT the
    kinetic-api. The API only returns relative paths; the actual file content is
    served by the SmartWitness S3-backed CDN.
    """
    if os.path.exists(local_path):
        return True

    full_url = f"{FILE_BASE_URL}{url_path}" if url_path.startswith("/") else url_path
    for attempt in range(3):
        try:
            res = session.get(full_url, stream=True, timeout=30)
            if res.status_code == 200:
                # Written under a temp name first: the dashboard globs this
                # folder live, and a half-written .gsdata would fail to parse.
                tmp_path = f"{local_path}.part"
                with open(tmp_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
                os.replace(tmp_path, local_path)
                return True
            print(
                f"Download attempt {attempt + 1}/3 for {full_url} got HTTP {res.status_code}"
            )
        except Exception as e:
            print(f"Download retry {attempt + 1}/3 for {full_url} failed: {e}")
            time.sleep(3)

    return False


def _atomic_write_json(path, payload):
    """Writes a summary JSON via temp file + rename.

    view_gsensor.py re-reads these files while the poller writes them, so a plain
    open(..., "w") would expose a truncated file to the dashboard.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    # os.replace() on Windows can transiently fail with PermissionError
    # (WinError 5) if another process - antivirus, an indexer, even a brief
    # read from the dashboard - has the destination open at that exact
    # instant. Nothing here is a real conflict (observed live during an enrich
    # sweep), so back off briefly and retry rather than losing an entire
    # batch's worth of freshly-fetched data to a moment's file lock.
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


class _DatasetState:
    """In-memory mirror of one dataset's summary JSON.

    Held across ticks so a 10s poll does not re-parse a multi-megabyte file every
    time; reloaded only when the file changed underneath us (e.g. someone ran
    main.py by hand).
    """

    def __init__(self, key):
        cfg = DATASETS[key]
        self.key = key
        self.decision = cfg["decision"]
        self.download_dir = cfg["download_dir"]
        self.summary_file = cfg["summary_file"]
        self.records = []
        self.by_id = {}
        self.enriched_ids = set()
        self.loaded = False
        self.file_signature = None
        self.lock = threading.Lock()

    def _signature(self):
        try:
            stat = os.stat(self.summary_file)
            return (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None

    def ensure_loaded(self):
        if self.loaded and self._signature() == self.file_signature:
            return

        self.records = []
        self.by_id = {}
        self.enriched_ids = set()

        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                for item in self.records:
                    eid = item.get("eventId")
                    if eid is None:
                        continue
                    self.by_id[eid] = item
                    if is_fully_enriched(item):
                        self.enriched_ids.add(eid)
            except Exception as e:
                print(f"[{self.key}] Could not read {self.summary_file}: {e}. Starting fresh.")
                self.records = []
                self.by_id = {}
                self.enriched_ids = set()

        self.file_signature = self._signature()
        self.loaded = True

    def save(self):
        _atomic_write_json(self.summary_file, self.records)
        self.file_signature = self._signature()


_STATES = {key: _DatasetState(key) for key in DATASETS}


def _ingest_item(state, item):
    """Adds or backfills one API item. Returns True if the dataset changed."""
    event_id = item.get("eventId")
    if event_id is None or event_id in state.enriched_ids:
        return False

    existing = state.by_id.get(event_id)
    if existing is not None:
        # Legacy record already on disk - backfill only the metadata fields,
        # leaving anything already downloaded untouched.
        existing["imei"] = item.get("imei")
        existing["deviceId"] = item.get("deviceId")
        existing["media"] = build_media_details(item)
        state.enriched_ids.add(event_id)
        return True

    media_urls = []
    for media in item.get("media", []):
        mp4_path = media.get("URL")
        if mp4_path:
            media_urls.append(
                f"{FILE_BASE_URL}{mp4_path}" if mp4_path.startswith("/") else mp4_path
            )

    # Download the G-Sensor binary: the dashboard indexes the .gsdata files on
    # disk, so an event whose file is missing never shows up there.
    gs_url_path = item.get("url")
    local_gs_path = None
    if gs_url_path:
        filename = f"event_{event_id}_{os.path.basename(gs_url_path)}"
        save_path = os.path.join(state.download_dir, filename)
        if download_file(_SESSION, gs_url_path, save_path):
            local_gs_path = os.path.abspath(save_path)

    gs_status = local_gs_path or (
        "No GS URL in API response" if not gs_url_path else "Download failed"
    )
    new_record = {
        "eventId": event_id,
        "eventType": item.get("eventType"),
        "imei": item.get("imei"),
        "deviceId": item.get("deviceId"),
        "gsensor_file_location": gs_status,
        "video_urls": media_urls,
        "media": build_media_details(item),
    }
    # Newest first, matching the sortOrder=DESC feed and the order main.py's page
    # walk left the existing file in.
    state.records.insert(0, new_record)
    state.by_id[event_id] = new_record
    state.enriched_ids.add(event_id)
    return True


def sync_dataset(key):
    """Polls the newest pages of one dataset and folds anything new into its JSON.

    Returns a stats dict; never raises - a failed tick is reported and the next
    one simply tries again.
    """
    state = _STATES[key]
    started = time.time()
    stats = {
        "dataset": key,
        "decision": state.decision,
        "new_events": 0,
        "new_event_ids": [],
        "pages_scanned": 0,
        "total_events": 0,
        "changed": False,
        "error": None,
    }

    # Overlapping ticks would double-download and race on the same file; while
    # one is running, the next simply reports a skip.
    if not state.lock.acquire(blocking=False):
        stats["error"] = "sync already in progress"
        return stats

    try:
        os.makedirs(state.download_dir, exist_ok=True)
        state.ensure_loaded()

        changed = False
        new_ids = []
        for page in range(1, MAX_POLL_PAGES + 1):
            params = {
                "page": page,
                "limit": LIMIT,
                "sortKey": SORT_KEY,
                "sortOrder": SORT_ORDER,
                "decision": state.decision,
            }
            response = authed_get(
                _SESSION,
                API_URL,
                apply_auth=APPLY_AUTH_TO_EVENT_API,
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()

            stats["pages_scanned"] = page
            page_data = payload.get("data", [])
            if not page_data:
                break

            page_changed = False
            for item in page_data:
                if item.get("eventId") not in state.enriched_ids:
                    new_ids.append(item.get("eventId"))
                if _ingest_item(state, item):
                    page_changed = True
                    changed = True

            # Everything on this page was already known: with a newest-first
            # sort, nothing further back can be new either.
            if not page_changed:
                break
            if page >= payload.get("totalPages", page):
                break

        if changed:
            state.save()

        stats["changed"] = changed
        stats["new_events"] = len(new_ids)
        stats["new_event_ids"] = new_ids
        stats["total_events"] = len(state.records)
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
    finally:
        state.lock.release()

    stats["duration_s"] = round(time.time() - started, 2)
    return stats


def sync_all():
    """One full tick: both datasets, decision=no and decision=yes."""
    return [sync_dataset(key) for key in DATASETS]


# --- Backfill -------------------------------------------------------------


def remote_total(decision):
    """The API's own totalRecords for a decision, via one tiny request.

    The response carries the count directly, so checking for a gap costs a
    single call rather than a walk over every page.
    """
    response = authed_get(
        _SESSION,
        API_URL,
        apply_auth=APPLY_AUTH_TO_EVENT_API,
        params={"page": 1, "limit": 1, "decision": decision},
        timeout=20,
    )
    response.raise_for_status()
    return int(response.json().get("totalRecords", 0))


def enumerate_remote(decision, page_size=None):
    """Phase 1: every event the API will show us, metadata only, no downloads.

    Deliberately does no I/O beyond the page requests - the faster this
    completes, the less the API's unstable ordering can shuffle records between
    pages underneath us.

    Returns (items_by_id, total_records, pages_walked).
    """
    page_size = page_size or BACKFILL_PAGE_SIZE
    items = {}
    page = 1
    total_pages = 1
    total_records = 0

    while page <= total_pages:
        response = authed_get(
            _SESSION,
            API_URL,
            apply_auth=APPLY_AUTH_TO_EVENT_API,
            params={"page": page, "limit": page_size, "decision": decision},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        total_pages = payload.get("totalPages", total_pages)
        total_records = int(payload.get("totalRecords", total_records))

        for item in payload.get("data", []):
            event_id = item.get("eventId")
            if event_id is not None:
                items[event_id] = item

        page += 1

    return items, total_records, page - 1


def backfill_dataset(key, progress=None, max_rounds=None):
    """Reconciles one dataset against the API until it stops finding new events.

    Safe to run alongside the 10s tail poll: the dataset lock is taken per batch
    rather than for the whole run, so sync_dataset() still gets its turn.

    `progress` is an optional callable taking the stats dict, invoked after each
    batch and round so a server can report live progress.
    """
    state = _STATES[key]
    max_rounds = max_rounds or BACKFILL_MAX_ROUNDS
    started = time.time()

    stats = {
        "dataset": key,
        "decision": state.decision,
        "running": True,
        "rounds": 0,
        "stagnant_rounds": 0,
        "remote_total": 0,
        "local_total": 0,
        "missing": 0,
        "added": 0,
        "downloaded": 0,
        "failed": 0,
        "error": None,
    }

    def report():
        if progress:
            try:
                progress(dict(stats))
            except Exception:
                pass   # a broken progress callback must never kill the backfill

    try:
        os.makedirs(state.download_dir, exist_ok=True)
        stagnant_rounds = 0

        for round_number in range(1, max_rounds + 1):
            stats["rounds"] = round_number

            remote_items, total_records, _ = enumerate_remote(state.decision)
            stats["remote_total"] = total_records

            with state.lock:
                state.ensure_loaded()
                known = set(state.by_id)
                stats["local_total"] = len(state.records)

            # Converged: the API says we already have everything it knows about.
            # Checked BEFORE looking at this round's missing_ids, since a sweep
            # that (by the pagination instability) misses the tail can otherwise
            # look identical to genuine completion.
            if stats["local_total"] >= stats["remote_total"] > 0:
                break

            missing_ids = [eid for eid in remote_items if eid not in known]
            stats["missing"] = len(missing_ids)
            report()

            if not missing_ids:
                # This sweep's ~87%-coverage snapshot happened not to surface
                # anything new - not proof there's nothing left, since we know
                # local_total < remote_total. Try again; only give up after
                # several sweeps in a row come back equally empty-handed.
                stagnant_rounds += 1
                stats["stagnant_rounds"] = stagnant_rounds
                if stagnant_rounds >= BACKFILL_STAGNANT_LIMIT:
                    break
                continue

            # Ingest in batches, releasing the lock between them so the tail
            # poll can still run while a long backfill is in flight.
            added_this_round = 0
            for start in range(0, len(missing_ids), BACKFILL_BATCH_SIZE):
                batch = missing_ids[start:start + BACKFILL_BATCH_SIZE]
                with state.lock:
                    state.ensure_loaded()
                    changed = False
                    for event_id in batch:
                        item = remote_items[event_id]
                        before = len(state.records)
                        if _ingest_item(state, item):
                            changed = True
                            if len(state.records) > before:
                                added_this_round += 1
                                stats["added"] += 1
                    if changed:
                        state.save()
                    stats["local_total"] = len(state.records)
                report()

            if stats["local_total"] >= stats["remote_total"] > 0:
                break

            # Actual progress this round resets the stagnation counter - it's
            # only "give up" territory when several rounds IN A ROW add nothing.
            stagnant_rounds = 0 if added_this_round > 0 else stagnant_rounds + 1
            stats["stagnant_rounds"] = stagnant_rounds
            if stagnant_rounds >= BACKFILL_STAGNANT_LIMIT:
                break

        with state.lock:
            stats["local_total"] = len(state.records)
        stats["complete"] = (
            stats["remote_total"] > 0 and stats["local_total"] >= stats["remote_total"]
        )
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["complete"] = False
    finally:
        stats["running"] = False
        stats["duration_s"] = round(time.time() - started, 2)
        report()

    return stats


def _unenriched_ids_locked(state):
    """Same as unenriched_ids(), for a caller that already holds state.lock.

    state.lock is a plain threading.Lock, not reentrant - calling the public
    unenriched_ids() (which locks itself) from inside a block that already
    holds the lock deadlocks the thread instead of raising, since the second
    acquire just blocks forever waiting for a release that can't happen on the
    same thread. This is that call, minus the second lock.
    """
    return [r["eventId"] for r in state.records if not is_fully_enriched(r)]


def unenriched_ids(key):
    """Locally-known event IDs still missing imei/deviceId/media (so: no
    timestamp, since that lives only in media[0].StartDateTime).

    Only for a caller that does NOT already hold this dataset's lock - from
    inside one, use _unenriched_ids_locked(state) instead.
    """
    state = _STATES[key]
    with state.lock:
        state.ensure_loaded()
        return _unenriched_ids_locked(state)


def enrich_dataset(key, progress=None, max_rounds=None):
    """Re-checks already-known events that are still missing media/timestamp,
    and backfills them from a fresh API pull.

    Complements backfill_dataset(), which only handles events not yet known at
    all - this handles events that ARE known locally but were caught mid
    upstream-processing on first fetch (media/timestamp not attached yet). Same
    enumerate -> diff -> ingest -> converge shape, same per-batch lock release
    so the 10s tail poll keeps running alongside a long sweep.
    """
    state = _STATES[key]
    max_rounds = max_rounds or BACKFILL_MAX_ROUNDS
    started = time.time()

    stats = {
        "dataset": key,
        "decision": state.decision,
        "running": True,
        "rounds": 0,
        "stagnant_rounds": 0,
        "still_unenriched": 0,
        "enriched": 0,
        "error": None,
    }

    def report():
        if progress:
            try:
                progress(dict(stats))
            except Exception:
                pass

    try:
        stagnant_rounds = 0

        for round_number in range(1, max_rounds + 1):
            stats["rounds"] = round_number

            remote_items, _, _ = enumerate_remote(state.decision)

            with state.lock:
                state.ensure_loaded()
                pending = {eid for eid in _unenriched_ids_locked(state) if eid in remote_items}

            stats["still_unenriched"] = len(pending)
            report()

            if not pending:
                break

            enriched_this_round = 0
            pending_list = list(pending)
            for start in range(0, len(pending_list), BACKFILL_BATCH_SIZE):
                batch = pending_list[start:start + BACKFILL_BATCH_SIZE]
                with state.lock:
                    state.ensure_loaded()
                    changed = False
                    for event_id in batch:
                        record = state.by_id.get(event_id)
                        if record is None or is_fully_enriched(record):
                            continue   # already fixed by a concurrent tail poll
                        item = remote_items.get(event_id)
                        if item and item.get("media"):
                            if _ingest_item(state, item):
                                changed = True
                                enriched_this_round += 1
                                stats["enriched"] += 1
                    if changed:
                        state.save()
                report()

            # A sweep that fixed nothing may just have missed the same records
            # again (pagination is unstable) - only give up after several
            # fruitless sweeps in a row, same reasoning as backfill_dataset().
            stagnant_rounds = 0 if enriched_this_round > 0 else stagnant_rounds + 1
            stats["stagnant_rounds"] = stagnant_rounds
            if stagnant_rounds >= BACKFILL_STAGNANT_LIMIT:
                break

        stats["still_unenriched"] = len(unenriched_ids(key))
        stats["complete"] = stats["still_unenriched"] == 0
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["complete"] = False
    finally:
        stats["running"] = False
        stats["duration_s"] = round(time.time() - started, 2)
        report()

    return stats


def gap_report():
    """Per-dataset local-vs-API counts, for deciding whether a backfill/enrich
    pass is due. `missing` = events not downloaded at all; `unenriched` =
    events downloaded but still missing media/timestamp."""
    report = {}
    for key, state in _STATES.items():
        entry = {"dataset": key, "decision": state.decision, "error": None}
        try:
            with state.lock:
                state.ensure_loaded()
                entry["local_total"] = len(state.records)
                entry["unenriched"] = sum(1 for r in state.records if not is_fully_enriched(r))
            entry["remote_total"] = remote_total(state.decision)
            entry["missing"] = max(0, entry["remote_total"] - entry["local_total"])
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["missing"] = 0
            entry["unenriched"] = 0
        report[key] = entry
    return report


if __name__ == "__main__":
    import sys

    if "--backfill" in sys.argv:
        for dataset_key in DATASETS:
            print(f"Backfilling {dataset_key}...")
            print(" ", backfill_dataset(dataset_key))
    elif "--enrich" in sys.argv:
        # Standalone script: backfills timestamp/media for every currently-known
        # event that's missing it, using the same event-with-gs API - the fix
        # for the historical gap left by the is_fully_enriched bug above.
        for dataset_key in DATASETS:
            gap = len(unenriched_ids(dataset_key))
            if gap == 0:
                print(f"{dataset_key}: already has timestamp/media for every local event.")
                continue
            print(f"{dataset_key}: {gap} event(s) missing timestamp/media - enriching...")
            print(" ", enrich_dataset(dataset_key))
    elif "--gap" in sys.argv:
        print(json.dumps(gap_report(), indent=2))
    else:
        for result in sync_all():
            print(result)
