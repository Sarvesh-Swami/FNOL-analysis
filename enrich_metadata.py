import json
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE_URL = "https://kinetic-api.dronaaim.ai"
API_URL = f"{API_BASE_URL}/ai-model/event-with-gs"
LIMIT = 10

# Same decision/file pairing as main.py and main_decision_yes.py
TARGETS = [
    {"decision": "no", "summary_file": "events_gforce_summary.json"},
    {"decision": "yes", "summary_file": "events_gforce_summary_yes.json"},
]


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


def enrich_file(session, decision, summary_file):
    with open(summary_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_id = {r["eventId"]: r for r in records if "eventId" in r}
    remaining = {
        eid for eid, r in by_id.items() if "imei" not in r or "deviceId" not in r
    }

    if not remaining:
        print(f"[{decision}] {summary_file} already fully enriched, skipping.")
        return

    print(
        f"[{decision}] {len(remaining)}/{len(by_id)} records need metadata in {summary_file}"
    )

    current_page = 1
    total_pages = 1

    while current_page <= total_pages and remaining:
        params = {"page": current_page, "limit": LIMIT, "decision": decision}

        try:
            response = session.get(API_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            print(
                f"[{decision}] Page {current_page} failed: {e}. Retrying in 10s..."
            )
            time.sleep(10)
            continue

        total_pages = payload.get("totalPages", total_pages)
        page_data = payload.get("data", [])

        touched = False
        for item in page_data:
            event_id = item.get("eventId")
            if event_id in remaining:
                record = by_id[event_id]
                record["imei"] = item.get("imei")
                record["deviceId"] = item.get("deviceId")
                remaining.discard(event_id)
                touched = True

        # Save progress after every page that actually updated something,
        # so an interrupted run can resume without losing work.
        if touched:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4)

        print(
            f"[{decision}] Page {current_page}/{total_pages} scanned, "
            f"{len(remaining)} record(s) still missing metadata"
        )
        current_page += 1

    if remaining:
        print(
            f"[{decision}] WARNING: {len(remaining)} eventId(s) not found in API: "
            f"{sorted(remaining)}"
        )

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print(f"[{decision}] Done enriching {summary_file}")


def main():
    session = get_resilient_session()
    for target in TARGETS:
        if os.path.exists(target["summary_file"]):
            enrich_file(session, target["decision"], target["summary_file"])
        else:
            print(f"Skipping missing file: {target['summary_file']}")


if __name__ == "__main__":
    main()
