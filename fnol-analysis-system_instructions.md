# FNOL Analysis System — project instructions

Working notes for whichever agent (or human) picks this up next.

## What this repo does

Pulls SmartWitness / Drona AIM crash-detection events off the kinetic-api feed,
downloads each event's `.gsdata` G-Sensor binary, and serves an interactive
telemetry + video dashboard over them.

Two datasets are kept strictly separate and never merged:

| Dataset | API `decision` | Sensor files | Summary JSON |
|---|---|---|---|
| No Crash | `no` | `downloads_gsensor/` | `events_gforce_summary.json` |
| Crash | `yes` | `downloads_gsensor_yes/` | `events_gforce_summary_yes.json` |

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env         # then fill in the Cognito values
python auth.py               # verifies the credentials, prints a token

python sync_server.py            # FastAPI poller on :8010 — keeps the JSONs fresh
python view_gsensor.py --serve   # dashboard on :8080 (separate terminal)
```

Both can run indefinitely. The dashboard works on its own against whatever is
already on disk; the poller is what makes it live.

## Authentication

`fleet/trip/livetrack` returns **401 without a Bearer token** — the GPS track on
the dashboard does not work unless `.env` is filled in. The event feed
(`ai-model/event-with-gs`) still answers unauthenticated; set
`APPLY_AUTH_TO_EVENT_API=true` when that changes.

`auth.py` mints and caches the token, renewing a minute before it expires and
preferring a cheap `REFRESH_TOKEN_AUTH` over a fresh login — so a 10s poller
costs one Cognito call an hour, not one per tick.

**`AUTH_FLOW=password` is the configured default**: a single `InitiateAuth` with
`USER_PASSWORD_AUTH`, needing only client id, email and password. It requires
`ALLOW_USER_PASSWORD_AUTH` on the app client; if that is off, Cognito answers
`InvalidParameterException: USER_PASSWORD_AUTH flow not enabled for this client`
and the fallback is `AUTH_FLOW=srp`, which additionally needs
`COGNITO_USER_POOL_ID` (readable from the `iss` claim of any IdToken).

Note on the `PASSWORD_VERIFIER` payload seen in devtools: those claim values are
signed against their `TIMESTAMP` and die within minutes, so they cannot be
pasted into `.env` and replayed. The `srp` flow regenerates them on every
refresh instead. The `password_verifier` flow exists for parity but is only good
for a one-off test.

A flow whose `.env` values are incomplete reports as disabled (`/auth` lists
exactly what is missing) rather than retrying a doomed login every tick.

If the API rejects the access token, flip `AUTH_TOKEN_KIND=id` — some Cognito
authorizers want the IdToken instead.

## How the pieces fit

- `auth.py` — Cognito token provider, `.env`-driven, thread-safe, shared by the
  poller and the dashboard. `authed_get()` attaches the Bearer token and retries
  once on a 401 with a freshly minted one.
- `event_sync.py` — the extraction core. One `sync_dataset(key)` call polls
  `event-with-gs` newest-first, downloads any new `.gsdata`, and folds new events
  into that dataset's summary JSON. Record shape is identical to what `main.py`
  wrote, so nothing downstream had to change.
- `livetrack_sync.py` — polls `fleet/trip/livetrack` per device with the token
  and keeps the newest fix in `livetrack_latest.json`. Devices come from
  `LIVETRACK_DEVICE_IDS`, or from the most recent events when that is empty.
- `sync_server.py` — FastAPI app. A background task runs the events sync and the
  livetrack sync every `POLL_INTERVAL_SECONDS` (10). Endpoints: `/status`,
  `/health`, `/auth`, `/events/{dataset}`, `/livetrack`,
  `/livetrack/{device_id}?refresh=true`, `POST /sync` (force a tick now).
- `view_gsensor.py --serve` — the dashboard. Its event index is cached against a
  fingerprint of (summary JSON mtime/size, `.gsdata` file count) and rebuilt on
  request only when that changes, so the poller's writes show up on their own.
  The page re-polls `/api/events` every 10s and refreshes the sidebar *without*
  disturbing the event currently being reviewed.
- `main.py` / `main_decision_yes.py` — the original one-shot full-archive
  backfill scripts. Still there for a cold start; day-to-day updating is the
  poller's job now.

## Things worth knowing

- Event files come from `sv.smartwitness.co`, **not** the kinetic-api. The API
  only ever returns relative paths.
- Both the summary JSONs and the `.gsdata` downloads are written via a temp file
  plus rename. The dashboard reads these files while the poller writes them, so
  a non-atomic write would hand it a truncated file.
- The API has no single "event instant", only a per-clip Start/EndDateTime; the
  dashboard uses that window's midpoint as the event time.
- An event only appears in the dashboard once its `.gsdata` file is on disk —
  the index globs the folder and enriches from the JSON, not the other way round.
