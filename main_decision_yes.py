import json
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE_URL = "https://kinetic-api.dronaaim.ai"
FILE_BASE_URL = "https://sv.smartwitness.co"  # SmartWitness S3-backed CDN (actual file server)
API_URL = f"{API_BASE_URL}/ai-model/event-with-gs"
LIMIT = 10
DECISION = "yes"

# Kept separate from the decision=no run (main.py) - different download folder and
# different summary JSON, so the two datasets never mix.
DOWNLOAD_DIR = "downloads_gsensor_yes"
OUTPUT_FILE = "events_gforce_summary_yes.json"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


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


def build_media_details(item):
    """Full per-media metadata straight from the API (Speed, Camera, Heading,
    Altitude, Latitude, Longitude, FileFormat, ContentSize, Start/EndDateTime,
    URL, MDTURL, ...), plus convenience absolute URLs.

    Kept alongside (not instead of) `video_urls`/`gsensor_file_location` so
    existing consumers (the dashboard) keep working unchanged.
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


# Fields this update adds on top of the original eventId/eventType/gsensor
# capture. A record is "fully enriched" once it carries all of them - checked
# by key presence, not truthiness, so a legitimately empty `media: []` isn't
# re-fetched forever.
REQUIRED_META_FIELDS = ("imei", "deviceId", "media")


def is_fully_enriched(record):
    return all(field in record for field in REQUIRED_META_FIELDS)


def download_file(session, url_path, local_path):
    """Downloads a binary file safely, skipping if it already exists.

    Files are fetched from FILE_BASE_URL (sv.smartwitness.co), NOT the
    kinetic-api.  The API only returns relative paths; the actual file
    content is served by the SmartWitness S3-backed CDN.
    """
    if os.path.exists(local_path):
        return True

    full_url = (
        f"{FILE_BASE_URL}{url_path}" if url_path.startswith("/") else url_path
    )
    for attempt in range(3):
        try:
            res = session.get(full_url, stream=True, timeout=30)
            if res.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(
                    f"\nDownload attempt {attempt + 1}/3 for {full_url} "
                    f"got HTTP {res.status_code}"
                )
        except Exception as e:
            print(
                f"\nDownload retry {attempt + 1}/3 for {full_url} failed: {e}"
            )
            time.sleep(3)

    return False


def process_all_pages():
    session = get_resilient_session()

    # 1. Load existing JSON progress if present
    all_events_summary = []
    by_id = {}
    fully_enriched_ids = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_events_summary = json.load(f)
                for item in all_events_summary:
                    eid = item.get("eventId")
                    if eid is None:
                        continue
                    by_id[eid] = item
                    if is_fully_enriched(item):
                        fully_enriched_ids.add(eid)
            print(
                f"Loaded {len(all_events_summary)} existing records from {OUTPUT_FILE} "
                f"({len(fully_enriched_ids)} already fully enriched)."
            )
        except Exception:
            print("Could not read existing summary file. Starting fresh.")

    current_page = 1
    total_pages = 1

    print("Starting data extraction (decision=yes)...")

    while current_page <= total_pages:
        params = {"page": current_page, "limit": LIMIT, "decision": DECISION}

        try:
            response = session.get(API_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            print(
                f"Failed to fetch Page {current_page}. Pausing for 10s before retry... ({e})"
            )
            time.sleep(10)
            continue  # Retry fetching this page

        total_pages = payload.get("totalPages", total_pages)
        page_data = payload.get("data", [])

        if not page_data:
            current_page += 1
            continue

        # 2. Check First and Last element of the page JSON
        first_event_id = page_data[0].get("eventId")
        last_event_id = page_data[-1].get("eventId")

        is_first_done = first_event_id in fully_enriched_ids
        is_last_done = last_event_id in fully_enriched_ids

        # If both are already fully enriched, skip entire page
        if is_first_done and is_last_done:
            print(
                f"Page {current_page}/{total_pages} already processed (First ID: {first_event_id}, Last ID: {last_event_id}). Skipping..."
            )
            current_page += 1
            continue

        # 3. Process items on this page
        print(
            f"Processing Page {current_page} of {total_pages} (Extracting new/incomplete events)..."
        )
        touched = False
        for item in page_data:
            event_id = item.get("eventId")
            if event_id in fully_enriched_ids:
                continue

            existing = by_id.get(event_id)
            if existing is not None:
                # Legacy record already on disk - backfill only the metadata
                # fields this update adds. Files already downloaded are left
                # untouched, so nothing already fetched gets fetched again.
                existing["imei"] = item.get("imei")
                existing["deviceId"] = item.get("deviceId")
                existing["media"] = build_media_details(item)
                fully_enriched_ids.add(event_id)
                touched = True
                continue

            # Brand-new event: full pipeline (download files + full record)
            media_urls = []
            for media in item.get("media", []):
                mp4_path = media.get("URL")
                if mp4_path:
                    full_mp4_url = (
                        f"{FILE_BASE_URL}{mp4_path}"
                        if mp4_path.startswith("/")
                        else mp4_path
                    )
                    media_urls.append(full_mp4_url)

            # Download G-Sensor binary file
            gs_url_path = item.get("url")
            local_gs_path = None

            if gs_url_path:
                filename = f"event_{event_id}_{os.path.basename(gs_url_path)}"
                save_path = os.path.join(DOWNLOAD_DIR, filename)

                if download_file(session, gs_url_path, save_path):
                    local_gs_path = os.path.abspath(save_path)

            gs_status = local_gs_path if local_gs_path else (
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
            all_events_summary.append(new_record)
            by_id[event_id] = new_record
            fully_enriched_ids.add(event_id)
            touched = True

        # 4. Save progress after completing the unskipped page
        if touched:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_events_summary, f, indent=4)

        current_page += 1

    print(
        f"\nDone! {len(all_events_summary)} total events tracked across {total_pages} pages "
        f"({len(fully_enriched_ids)} fully enriched)."
    )


if __name__ == "__main__":
    process_all_pages()
