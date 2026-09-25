import gzip
import struct
import os
import sys
import json
import argparse
import math
import statistics
import requests
from datetime import datetime, timezone

from auth import auth, authed_get

LIVETRACK_API_BASE = "https://kinetic-api.dronaaim.ai/fleet/trip/livetrack"

RECORD_SIZE = 14  # 8 bytes timestamp (uint64) + 3 * 2 bytes int16 (x, y, z in milli-g)

SAMPLE_RATE_HZ = 100          # verified: every file in the corpus has uniform 10 ms deltas

# --- Gravity baseline estimation tunables -----------------------------------------
BASELINE_WIN = 50             # 0.5 s @ 100 Hz - window used to hunt for quiet stretches
BASELINE_STEP = 10            # 0.1 s hop between candidate windows
BASELINE_QUIET_DIVISOR = 4    # average over the quietest 1/4 of windows
BASELINE_MAX_TILT_DEG = 20.0  # drop quiet windows pointing >20 deg off the seed window
BASELINE_MAX_MAG_TOL = 0.20   # ...or differing >20% in magnitude (camera shifted mid-clip)
GRAVITY_MG_MIN = 500.0        # plausibility band for a 1 g baseline
GRAVITY_MG_MAX = 1500.0

# The device's nominal axis convention is X = lateral, Y = longitudinal, Z = vertical.
# When a camera is mounted sideways or upside-down a different axis carries gravity, so
# the remaining two axes have to be re-mapped.  Keyed by the detected vertical axis.
_HORIZONTAL_BASIS = {
    "z": ("x", "y"),   # upright or inverted   -> lateral X, longitudinal Y
    "x": ("z", "y"),   # rolled onto its side  -> lateral Z, longitudinal Y
    "y": ("x", "z"),   # pitched nose up/down  -> lateral X, longitudinal Z
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _prefix_sums(values):
    """Returns (sum, sum-of-squares) prefix arrays so window variance is O(1)."""
    n = len(values)
    s = [0.0] * (n + 1)
    q = [0.0] * (n + 1)
    for i, v in enumerate(values):
        s[i + 1] = s[i] + v
        q[i + 1] = q[i] + v * v
    return s, q


def estimate_gravity(xs, ys, zs):
    """Estimates the constant gravity vector (in mg) from the quiet parts of a clip.

    An accelerometer measures proper acceleration, so a stationary vehicle reads a full
    1 g along whichever axis happens to point down.  To recover real vehicle dynamics
    that baseline must be measured and subtracted - and it cannot be assumed to sit on
    +Z, because cameras get mounted sideways and upside-down.

    Method: slide a 0.5 s window across the clip and score each window by summed
    per-axis variance, then average the raw axes over the quietest quarter of windows.
    Windows whose mean direction disagrees with the quietest ("seed") window are
    dropped: without that filter, a clip where the camera is knocked loose mid-event
    blends two orientations and produces a physically impossible sub-1 g baseline.

    Returns (gravity_mg_tuple, method, quiet_noise_mg).
    """
    n = len(xs)
    median_fallback = (
        statistics.median(xs),
        statistics.median(ys),
        statistics.median(zs),
    )
    if n < BASELINE_WIN:
        return median_fallback, "median-short-clip", 0.0

    prefixes = [_prefix_sums(a) for a in (xs, ys, zs)]

    # Score every candidate window by how still the vehicle is during it.
    scored = []
    for start in range(0, n - BASELINE_WIN + 1, BASELINE_STEP):
        end = start + BASELINE_WIN
        activity = 0.0
        means = []
        for s, q in prefixes:
            mean = (s[end] - s[start]) / BASELINE_WIN
            means.append(mean)
            activity += (q[end] - q[start]) / BASELINE_WIN - mean * mean
        scored.append((activity, start, tuple(means)))

    scored.sort(key=lambda w: w[0])
    quiet_noise_mg = math.sqrt(max(0.0, scored[0][0]))
    candidates = scored[: max(1, len(scored) // BASELINE_QUIET_DIVISOR)]

    # Keep only candidates that agree with the seed window on where "down" is.
    seed = candidates[0][2]
    seed_mag = math.sqrt(sum(v * v for v in seed))
    cos_limit = math.cos(math.radians(BASELINE_MAX_TILT_DEG))
    selected = []
    if seed_mag > 0:
        for _, _, means in candidates:
            mag = math.sqrt(sum(v * v for v in means))
            if mag <= 0:
                continue
            cosine = sum(a * b for a, b in zip(means, seed)) / (mag * seed_mag)
            if cosine >= cos_limit and abs(mag - seed_mag) / seed_mag <= BASELINE_MAX_MAG_TOL:
                selected.append(means)
    if not selected:
        selected = [seed]

    gravity = tuple(
        sum(m[axis] for m in selected) / len(selected) for axis in range(3)
    )
    gravity_mag = math.sqrt(sum(v * v for v in gravity))

    if not (GRAVITY_MG_MIN <= gravity_mag <= GRAVITY_MG_MAX):
        return median_fallback, "median-implausible-baseline", quiet_noise_mg

    method = "quiet-windows" if len(selected) == len(candidates) else "quiet-windows-filtered"
    return gravity, method, quiet_noise_mg


def detect_orientation(gravity_mg):
    """Auto-detects how the camera is mounted from the direction of gravity."""
    gx, gy, gz = gravity_mg
    magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)
    components = {"x": gx, "y": gy, "z": gz}
    vertical_axis = max(components, key=lambda k: abs(components[k]))
    sign = 1 if components[vertical_axis] >= 0 else -1

    # How far the measured gravity vector sits from the axis it is closest to.
    tilt_deg = 0.0
    if magnitude > 0:
        tilt_deg = math.degrees(
            math.acos(min(1.0, abs(components[vertical_axis]) / magnitude))
        )

    if vertical_axis == "z":
        label = "Upright (normal mount)" if sign > 0 else "Inverted (upside-down mount)"
    elif vertical_axis == "x":
        label = f"Rolled onto side ({'+' if sign > 0 else '-'}X points down)"
    else:
        label = f"Pitched ({'+' if sign > 0 else '-'}Y points down)"

    lateral_axis, longitudinal_axis = _HORIZONTAL_BASIS[vertical_axis]

    # The horizontal/vertical split is a projection onto the measured gravity
    # direction, so it stays exact at any tilt.  Deciding which of the two REMAINING
    # axes is braking rather than turning does depend on the dominant-axis pick, and
    # that becomes a coin flip as the camera approaches 45 deg off-axis.
    if tilt_deg <= 15.0:
        axis_confidence = "high"
    elif tilt_deg <= 30.0:
        axis_confidence = "medium"
    else:
        axis_confidence = "low"

    return {
        "vertical_axis": vertical_axis,
        "vertical_sign": sign,
        "label": label,
        "tilt_deg": round(tilt_deg, 1),
        "lateral_axis": lateral_axis,
        "longitudinal_axis": longitudinal_axis,
        "axis_confidence": axis_confidence,
        "gravity_mg": [round(v, 1) for v in gravity_mg],
        "gravity_magnitude_mg": round(magnitude, 1),
        # Gravity pins down which axis is vertical, but not which way the vehicle
        # faces - a 180 deg yaw looks identical.  So forward/back and left/right
        # polarity comes from the device convention; it is not measured here.
        "polarity": "assumed-from-device-convention",
    }


def _horizontal_basis(gravity_unit, lateral_axis, longitudinal_axis):
    """Builds orthonormal lateral/longitudinal axes lying in the true horizontal plane.

    The nominal device axes are tilted by however the camera actually sits, so each is
    projected off the gravity direction (Gram-Schmidt) and renormalised.  This keeps
    the split correct for arbitrary mounting angles, not just axis-aligned ones.
    """
    def horizontalize(axis):
        basis = [0.0, 0.0, 0.0]
        basis[_AXIS_INDEX[axis]] = 1.0
        along = sum(a * b for a, b in zip(basis, gravity_unit))
        vec = [a - along * b for a, b in zip(basis, gravity_unit)]
        norm = math.sqrt(sum(c * c for c in vec))
        return [c / norm for c in vec] if norm > 1e-9 else None

    lateral = horizontalize(lateral_axis)
    longitudinal = horizontalize(longitudinal_axis)

    # Re-orthogonalise longitudinal against lateral so they cannot double-count
    # the same motion once both have been tilted into the horizontal plane.
    if lateral and longitudinal:
        along = sum(a * b for a, b in zip(longitudinal, lateral))
        vec = [a - along * b for a, b in zip(longitudinal, lateral)]
        norm = math.sqrt(sum(c * c for c in vec))
        longitudinal = [c / norm for c in vec] if norm > 1e-9 else None

    return lateral, longitudinal


def _empty_result(file_path):
    """A fully-formed result for a file with no usable records, so callers never KeyError."""
    return {
        "file": os.path.basename(file_path),
        "total_records": 0,
        "count": 0,
        "duration_s": 0,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "gravity": None,
        "orientation": None,
        "baseline_method": "no-records",
        "quiet_noise_g": 0.0,
        "peak_dyn_g": 0.0,
        "peak_horiz_g": 0.0,
        "peak_vert_g": 0.0,
        "peak_raw_mag_g": 0.0,
        "peak_mag_g": 0.0,
        "peak_record": None,
        "records": [],
    }


def parse_gsdata(file_path):
    """
    Parses a SmartWitness .gsdata file.
    Returns a dict with metadata and a list of records:
    - timestamp_ms: uint64 epoch milliseconds
    - rel_time_s: seconds from start of sample (0.0 to ~21.0s)
    - x_mg, y_mg, z_mg: raw milli-g values
    - x_g, y_g, z_g: raw acceleration in G, gravity included (1G = 1000 mg)
    - x_dyn_g, y_dyn_g, z_dyn_g: same axes with the gravity baseline subtracted
    - mag_g: raw total vector magnitude, gravity included (~1.0 g when parked)

    Gravity-compensated fields - these are the ones to use for shock severity:
    - dyn_g:   total dynamic acceleration, gravity removed (~0.0 g when parked)
    - horiz_g: horizontal magnitude - braking and cornering
    - vert_g:  signed vertical component, positive = upward (potholes, kerbs)
    - lat_g:   signed lateral (cornering)
    - lon_g:   signed longitudinal (braking / acceleration)

    The gravity baseline is measured from the quiet parts of each individual clip and
    the mounting orientation is auto-detected, so sideways and inverted cameras are
    handled without configuration.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with gzip.open(file_path, "rb") as f:
        raw_bytes = f.read()

    total_records = len(raw_bytes) // RECORD_SIZE
    if total_records == 0:
        return _empty_result(file_path)

    # Unpack once up front; the gravity baseline needs the whole clip before any
    # per-record value can be derived.
    unpacked = [
        struct.unpack("<Qhhh", raw_bytes[i * RECORD_SIZE : (i + 1) * RECORD_SIZE])
        for i in range(total_records)
    ]
    xs = [r[1] for r in unpacked]
    ys = [r[2] for r in unpacked]
    zs = [r[3] for r in unpacked]

    gravity, baseline_method, quiet_noise_mg = estimate_gravity(xs, ys, zs)
    orientation = detect_orientation(gravity)

    gravity_mag = math.sqrt(sum(v * v for v in gravity))
    if gravity_mag > 0:
        gravity_unit = [v / gravity_mag for v in gravity]
    else:  # pathological: no measurable baseline, treat +Z as down
        gravity_unit = [0.0, 0.0, 1.0]

    lateral_basis, longitudinal_basis = _horizontal_basis(
        gravity_unit, orientation["lateral_axis"], orientation["longitudinal_axis"]
    )

    initial_ts = unpacked[0][0]
    records = []
    peak_dyn = -1.0
    peak_record = None
    peak_horiz = 0.0
    peak_vert = 0.0
    peak_raw = 0.0

    for i, (ts, x, y, z) in enumerate(unpacked):
        rel_time_s = round((ts - initial_ts) / 1000.0, 3)

        # Raw magnitude, gravity included - kept for reference and back-compat.
        raw_mag_mg = math.sqrt(x * x + y * y + z * z)

        # Dynamic acceleration: what the vehicle actually did.
        dyn = (x - gravity[0], y - gravity[1], z - gravity[2])
        dyn_mg = math.sqrt(sum(v * v for v in dyn))

        # Split along / across the gravity direction.  Sign flipped so that
        # positive vertical means upward, which is what a pothole kick reads as.
        along_gravity = sum(a * b for a, b in zip(dyn, gravity_unit))
        vert_mg = -along_gravity
        horiz_vec = [a - along_gravity * b for a, b in zip(dyn, gravity_unit)]
        horiz_mg = math.sqrt(sum(v * v for v in horiz_vec))

        lat_mg = (
            sum(a * b for a, b in zip(horiz_vec, lateral_basis))
            if lateral_basis
            else 0.0
        )
        lon_mg = (
            sum(a * b for a, b in zip(horiz_vec, longitudinal_basis))
            if longitudinal_basis
            else 0.0
        )

        dt_utc = datetime.fromtimestamp(ts / 1000.0, timezone.utc)
        dt_iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        rec = {
            "idx": i,
            "timestamp_ms": ts,
            "datetime": dt_iso,
            "rel_time_s": rel_time_s,
            "x_mg": x,
            "y_mg": y,
            "z_mg": z,
            "x_g": round(x / 1000.0, 3),
            "y_g": round(y / 1000.0, 3),
            "z_g": round(z / 1000.0, 3),
            # Per-axis with the measured baseline removed: same X/Y/Z axes, but
            # reading ~0 when the vehicle is still instead of carrying gravity.
            "x_dyn_g": round(dyn[0] / 1000.0, 3),
            "y_dyn_g": round(dyn[1] / 1000.0, 3),
            "z_dyn_g": round(dyn[2] / 1000.0, 3),
            "mag_g": round(raw_mag_mg / 1000.0, 3),
            "dyn_g": round(dyn_mg / 1000.0, 3),
            "horiz_g": round(horiz_mg / 1000.0, 3),
            "vert_g": round(vert_mg / 1000.0, 3),
            "lat_g": round(lat_mg / 1000.0, 3),
            "lon_g": round(lon_mg / 1000.0, 3),
        }
        records.append(rec)

        # Peak is ranked on unrounded dynamic magnitude so the shock timestamp is
        # exact - ranking on the gravity-inclusive value picks the wrong moment.
        if dyn_mg > peak_dyn:
            peak_dyn = dyn_mg
            peak_record = rec
        horiz_g_abs = horiz_mg / 1000.0
        if horiz_g_abs > peak_horiz:
            peak_horiz = horiz_g_abs
        if abs(vert_mg) / 1000.0 > abs(peak_vert):
            peak_vert = vert_mg / 1000.0
        raw_g = raw_mag_mg / 1000.0
        if raw_g > peak_raw:
            peak_raw = raw_g

    peak_dyn_g = round(peak_dyn / 1000.0, 3)
    return {
        "file": os.path.basename(file_path),
        "total_records": total_records,
        "duration_s": records[-1]["rel_time_s"],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "gravity": {
            "x_mg": round(gravity[0], 1),
            "y_mg": round(gravity[1], 1),
            "z_mg": round(gravity[2], 1),
            "magnitude_mg": round(gravity_mag, 1),
        },
        "orientation": orientation,
        "baseline_method": baseline_method,
        "quiet_noise_g": round(quiet_noise_mg / 1000.0, 4),
        "peak_dyn_g": peak_dyn_g,
        "peak_horiz_g": round(peak_horiz, 3),
        "peak_vert_g": round(peak_vert, 3),
        "peak_raw_mag_g": round(peak_raw, 3),
        # Back-compat alias: existing callers reading peak_mag_g now get the
        # gravity-corrected figure, which is what they always meant.
        "peak_mag_g": peak_dyn_g,
        "peak_record": peak_record,
        "records": records,
    }


CSV_HEADER = (
    "Index,Timestamp_ms,DateTime_UTC,RelativeTime_s,"
    "X_mg,Y_mg,Z_mg,X_g,Y_g,Z_g,RawMagnitude_g,"
    "X_dyn_g,Y_dyn_g,Z_dyn_g,"
    "Dynamic_g,Horizontal_g,Vertical_g,Lateral_g,Longitudinal_g\n"
)


def csv_row(r):
    """One CSV line for a parsed record. Shared by file export and the dashboard."""
    return (
        f"{r['idx']},{r['timestamp_ms']},{r['datetime']},{r['rel_time_s']},"
        f"{r['x_mg']},{r['y_mg']},{r['z_mg']},"
        f"{r['x_g']},{r['y_g']},{r['z_g']},{r['mag_g']},"
        f"{r['x_dyn_g']},{r['y_dyn_g']},{r['z_dyn_g']},"
        f"{r['dyn_g']},{r['horiz_g']},{r['vert_g']},{r['lat_g']},{r['lon_g']}\n"
    )


def export_csv(parsed, output_csv_path):
    """Exports parsed gsensor records to a CSV file.

    Raw X/Y/Z and RawMagnitude_g still include gravity.  The trailing columns are
    gravity-compensated: Horizontal_g is braking + cornering, Vertical_g is potholes
    and kerbs, and Dynamic_g is their resultant.  A commented header records the
    measured gravity baseline and detected mounting so the numbers are reproducible.
    """
    with open(output_csv_path, "w", encoding="utf-8") as f:
        orientation = parsed.get("orientation")
        if orientation:
            gravity = parsed["gravity"]
            f.write(
                f"# file={parsed['file']} mounting=\"{orientation['label']}\" "
                f"vertical_axis={orientation['vertical_axis']}{'+' if orientation['vertical_sign'] > 0 else '-'} "
                f"tilt_deg={orientation['tilt_deg']} "
                f"axis_confidence={orientation['axis_confidence']} "
                f"gravity_baseline_mg=({gravity['x_mg']},{gravity['y_mg']},{gravity['z_mg']}) "
                f"|g|={gravity['magnitude_mg']}mg baseline_method={parsed['baseline_method']}\n"
            )
        f.write(CSV_HEADER)
        for r in parsed["records"]:
            f.write(csv_row(r))
    print(f"Exported CSV: {output_csv_path} ({len(parsed['records'])} rows)")


def generate_html_viewer(parsed, output_html_path):
    """Generates a standalone, beautiful interactive HTML chart using Chart.js."""
    labels = [r["rel_time_s"] for r in parsed["records"]]
    # X/Y/Z with the measured gravity baseline removed, so a still vehicle reads ~0.
    x_vals = [r["x_dyn_g"] for r in parsed["records"]]
    y_vals = [r["y_dyn_g"] for r in parsed["records"]]
    z_vals = [r["z_dyn_g"] for r in parsed["records"]]

    filename = parsed["file"]
    duration = parsed["duration_s"]
    total = parsed["total_records"]
    peak = parsed["peak_dyn_g"]
    peak_raw = parsed["peak_raw_mag_g"]
    peak_time = parsed["peak_record"]["rel_time_s"] if parsed["peak_record"] else 0
    start_dt = parsed["records"][0]["datetime"] if parsed["records"] else ""

    orientation = parsed.get("orientation") or {}
    mount_label = orientation.get("label", "Unknown")
    tilt = orientation.get("tilt_deg", 0)
    grav = parsed.get("gravity") or {}
    confidence = orientation.get("axis_confidence", "n/a")
    grav_note = (
        f"Gravity baseline {grav.get('magnitude_mg', 0)} mg removed "
        f"({parsed.get('baseline_method', 'n/a')}), tilt {tilt}&deg;, "
        f"turning/braking axis confidence: {confidence}"
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G-Sensor Telemetry - {filename}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-x: #ff6b6b;
            --accent-y: #4ecdc4;
            --accent-z: #ffd166;
            --accent-mag: #a78bfa;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 22px;
            color: #58a6ff;
        }}
        .meta-subtitle {{
            color: var(--text-secondary);
            font-size: 13px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: 600;
        }}
        .stat-x {{ color: var(--accent-x); }}
        .stat-y {{ color: var(--accent-y); }}
        .stat-z {{ color: var(--accent-z); }}
        .stat-mag {{ color: var(--accent-mag); }}
        .chart-container {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            position: relative;
            height: 520px;
        }}
        .controls {{
            margin-top: 16px;
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .btn {{
            background-color: #21262d;
            color: #c9d1d9;
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: 0.2s;
        }}
        .btn:hover {{
            background-color: #30363d;
            color: #fff;
        }}
        .legend-note {{
            color: var(--text-secondary);
            font-size: 13px;
            margin-left: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>G-Sensor Telemetry Viewer</h1>
                <div class="meta-subtitle">{filename} | Start: {start_dt} UTC | Sampling: 100 Hz (10 ms)</div>
                <div class="meta-subtitle">Mounting: <strong>{mount_label}</strong> &middot; {grav_note}</div>
            </div>
            <div>
                <button class="btn" onclick="resetZoom()">Reset Zoom</button>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{duration} s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak Shock (gravity removed)</div>
                <div class="stat-value stat-mag">{peak} g</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak Timestamp</div>
                <div class="stat-value">t = {peak_time} s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Raw Peak (incl. gravity)</div>
                <div class="stat-value" style="color: var(--text-secondary)">{peak_raw} g</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="telemetryChart"></canvas>
        </div>

        <div class="controls">
            <button class="btn" onclick="toggleDataset(0)">Toggle X</button>
            <button class="btn" onclick="toggleDataset(1)">Toggle Y</button>
            <button class="btn" onclick="toggleDataset(2)">Toggle Z</button>
            <span class="legend-note">Tip: Click and drag horizontally to zoom in on shock impact. Double-click or click Reset Zoom to restore.</span>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('telemetryChart').getContext('2d');
        const labels = {json.dumps(labels)};
        const xVals = {json.dumps(x_vals)};
        const yVals = {json.dumps(y_vals)};
        const zVals = {json.dumps(z_vals)};

        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'X',
                        data: xVals,
                        borderColor: '#ff6b6b',
                        backgroundColor: '#ff6b6b22',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1
                    }},
                    {{
                        label: 'Y',
                        data: yVals,
                        borderColor: '#4ecdc4',
                        backgroundColor: '#4ecdc422',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1
                    }},
                    {{
                        label: 'Z',
                        data: zVals,
                        borderColor: '#ffd166',
                        backgroundColor: '#ffd16622',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{
                            color: '#e6edf3',
                            font: {{ size: 12 }}
                        }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            title: (items) => `Time: ${{items[0].label}}s`,
                            label: (item) => `${{item.dataset.label}}: ${{item.formattedValue}} g`
                        }}
                    }},
                    zoom: {{
                        pan: {{
                            enabled: true,
                            mode: 'x',
                        }},
                        zoom: {{
                            wheel: {{ enabled: true }},
                            pinch: {{ enabled: true }},
                            drag: {{ enabled: true, backgroundColor: 'rgba(88, 166, 255, 0.2)' }},
                            mode: 'x',
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Relative Time (seconds)',
                            color: '#8b949e'
                        }},
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e', maxTicksLimit: 22 }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Acceleration (g)',
                            color: '#8b949e'
                        }},
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e' }}
                    }}
                }}
            }}
        }});

        function resetZoom() {{
            chart.resetZoom();
        }}

        function toggleDataset(index) {{
            chart.setDatasetVisibility(index, !chart.isDatasetVisible(index));
            chart.update();
        }}
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated Interactive HTML Viewer: {output_html_path}")


def batch_export_csv(output_dir="csv_exports"):
    """Converts all .gsdata files in downloads_gsensor to CSV in output_dir."""
    import glob
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join("downloads_gsensor", "*.gsdata"))
    if not files:
        print("No .gsdata files found in downloads_gsensor/")
        return

    print(f"Starting batch CSV export for {len(files)} files into '{output_dir}/'...")
    for idx, fn in enumerate(files, 1):
        try:
            parsed = parse_gsdata(fn)
            base_name = os.path.splitext(os.path.basename(fn))[0]
            out_csv = os.path.join(output_dir, f"{base_name}.csv")
            export_csv(parsed, out_csv)
            if idx % 100 == 0 or idx == len(files):
                print(f"Progress: {idx}/{len(files)} exported...")
        except Exception as e:
            print(f"Error converting {fn}: {e}")
    print(f"Batch conversion complete! CSVs saved in: {os.path.abspath(output_dir)}")


# decision=no ("No Crash") and decision=yes ("Crash") are extracted by separate
# scripts (main.py / main_decision_yes.py) into separate folders and summary files,
# so they are indexed and served as two independent datasets - never merged.
DATASETS = {
    "no_crash": {
        "download_dir": "downloads_gsensor",
        "summary_file": "events_gforce_summary.json",
    },
    "crash": {
        "download_dir": "downloads_gsensor_yes",
        "summary_file": "events_gforce_summary_yes.json",
    },
}


def _parse_iso_to_epoch_ms(iso_str):
    """Parses the API's .NET-style timestamps (e.g. "2026-08-23T02:11:08.0000000Z",
    arbitrary fractional-second digits) into epoch milliseconds. `datetime.fromisoformat`
    can't be trusted here across Python versions - it rejects the trailing "Z" and
    7-digit fractions on anything before 3.11 - so the fraction is normalized to
    microseconds by hand instead.
    """
    if not iso_str:
        return None
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        s = f"{main}.{frac}"
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%dT%H:%M:%S"
    try:
        dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def _build_events_index(download_dir, summary_file):
    """Indexes one dataset's .gsdata files, enriched with eventType/video_url
    from its summary JSON. Missing folder or summary file just yields an empty
    or metadata-less index rather than raising - a dataset not extracted yet
    (e.g. main_decision_yes.py hasn't been run) should show as empty, not crash
    the server.
    """
    import glob

    summary_map = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    summary_map[str(item.get("eventId"))] = item
        except Exception as e:
            print(f"Could not load {summary_file}: {e}")

    file_list = glob.glob(os.path.join(download_dir, "*.gsdata"))
    events_index = []

    for fn in file_list:
        base = os.path.basename(fn)
        event_id = None
        if base.startswith("event_"):
            parts = base.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                event_id = parts[1]

        item_info = summary_map.get(event_id, {})
        video_urls = item_info.get("video_urls", [])

        # The API only exposes a Start/EndDateTime per media clip, not a single
        # "event" instant - the shock trigger sits somewhere in that ~20s window,
        # so its midpoint is used as the best available estimate of event time.
        media_list = item_info.get("media") or []
        start_dt = media_list[0].get("StartDateTime") if media_list else None
        end_dt = media_list[0].get("EndDateTime") if media_list else None
        start_ms = _parse_iso_to_epoch_ms(start_dt)
        end_ms = _parse_iso_to_epoch_ms(end_dt)
        if start_ms is not None and end_ms is not None:
            event_time_ms = (start_ms + end_ms) // 2
        else:
            event_time_ms = start_ms if start_ms is not None else end_ms

        events_index.append({
            "event_id": event_id if event_id else base,
            "filename": base,
            "file_path": fn,
            "event_type": item_info.get("eventType", "Unknown"),
            "video_url": video_urls[0] if video_urls else None,
            "imei": item_info.get("imei"),
            "device_id": item_info.get("deviceId"),
            "event_time_ms": event_time_ms,
            "start_datetime": start_dt,
            "end_datetime": end_dt,
        })

    events_index.sort(key=lambda x: str(x["event_id"]), reverse=True)
    return events_index


# Built index per dataset, keyed by a fingerprint of the files it was built from.
_INDEX_CACHE = {}


def _dataset_fingerprint(download_dir, summary_file):
    """Cheap change-detector for one dataset.

    sync_server.py rewrites the summary JSON and drops new .gsdata files while
    this dashboard is running, so the index can no longer be built once at
    startup - but re-globbing and re-parsing a multi-megabyte JSON on every
    request would be wasteful. The summary file's (mtime, size) plus the .gsdata
    file count is enough to notice a poll that actually added something.
    """
    try:
        stat = os.stat(summary_file)
        summary_sig = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        summary_sig = None

    try:
        file_count = sum(
            1 for name in os.listdir(download_dir) if name.endswith(".gsdata")
        )
    except OSError:
        file_count = 0

    return (summary_sig, file_count)


def get_events_index(key):
    """Returns a dataset's index, rebuilding it only when its files have changed."""
    cfg = DATASETS[key]
    fingerprint = _dataset_fingerprint(cfg["download_dir"], cfg["summary_file"])
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    events_index = _build_events_index(cfg["download_dir"], cfg["summary_file"])
    _INDEX_CACHE[key] = (fingerprint, events_index)
    return events_index


def run_dashboard_server(port=8080):
    """Starts a local HTTP server providing an interactive telemetry dashboard."""
    import http.server
    import urllib.parse
    import webbrowser

    # Warm the cache once up front; every later request re-checks the files and
    # rebuilds only if sync_server.py changed them.
    for key in DATASETS:
        get_events_index(key)

    def dataset_key(query):
        key = query.get("dataset", ["no_crash"])[0]
        return key if key in DATASETS else "no_crash"

    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)

            if path == "/api/events":
                key = dataset_key(query)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(get_events_index(key)).encode("utf-8"))
                return

            if path == "/api/data":
                key = dataset_key(query)
                events_index = get_events_index(key)
                download_dir = DATASETS[key]["download_dir"]

                target_file = None
                if "file" in query:
                    target_file = os.path.join(download_dir, os.path.basename(query["file"][0]))
                elif "id" in query:
                    eid = query["id"][0]
                    for ev in events_index:
                        if str(ev["event_id"]) == str(eid):
                            target_file = ev["file_path"]
                            break

                if target_file and os.path.exists(target_file):
                    try:
                        parsed = parse_gsdata(target_file)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps(parsed).encode("utf-8"))
                        return
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(f"Error parsing file: {e}".encode("utf-8"))
                        return
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"File or Event ID not found")
                    return

            if path == "/api/livetrack":
                device_id = query.get("device", [None])[0]
                from_ms = query.get("from", [None])[0]
                to_ms = query.get("to", [None])[0]
                if not device_id or not from_ms or not to_ms:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing device/from/to query parameters")
                    return

                # Proxied server-side (rather than fetched directly from the
                # browser) so this works regardless of the external API's own
                # CORS policy, so it's called with the same requests/retry setup
                # as the rest of this project's API access - and so the Cognito
                # token stays on the server instead of being handed to the page.
                try:
                    resp = authed_get(
                        requests,
                        f"{LIVETRACK_API_BASE}/{device_id}",
                        params={"fromDate": from_ms, "toDate": to_ms},
                        timeout=20,
                    )
                    resp.raise_for_status()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp.content)
                except Exception as e:
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(f"Livetrack fetch failed: {e}".encode("utf-8"))
                return

            if path == "/" or path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
                return

            self.send_response(404)
            self.end_headers()

    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, DashboardHandler)
    url = f"http://localhost:{port}"
    print(f"\n========================================================")
    print(f"  G-Sensor Telemetry Dashboard is running!")
    print(f"  URL: {url}")
    print(f"  No Crash (decision=no):  {len(get_events_index('no_crash'))} sensor files in {DATASETS['no_crash']['download_dir']}/")
    print(f"  Crash (decision=yes):    {len(get_events_index('crash'))} sensor files in {DATASETS['crash']['download_dir']}/")
    print(f"  Live updates: run 'python sync_server.py' alongside this to poll the API every 10s.")
    auth_state = auth.status()
    print(
        f"  Auth: {'enabled (flow=' + auth_state['flow'] + ')' if auth_state['enabled'] else 'disabled - GPS calls go out unauthenticated'}"
    )
    print(f"  Press Ctrl+C in terminal to stop.")
    print(f"========================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartWitness Telemetry & Video Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {
            --bg-main: #0b0f19;
            --sidebar-bg: #111827;
            --card-bg: #1f2937;
            --border: #374151;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-blue: #3b82f6;
            --accent-red: #ef4444;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-pink: #ec4899;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        #sidebar {
            width: 320px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        .sidebar-header h2 {
            margin: 0;
            font-size: 16px;
            color: var(--accent-blue);
        }
        .header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 6px;
        }
        .dataset-select {
            background: #1e293b;
            color: #fff;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 5px 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }
        .dataset-select:focus {
            outline: 1px solid var(--accent-blue);
        }
        .search-box {
            width: 100%;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: #1e293b;
            color: #fff;
            font-size: 13px;
            margin-top: 8px;
        }
        #eventsList {
            flex: 1;
            overflow-y: auto;
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .event-item {
            padding: 12px 16px;
            border-bottom: 1px solid #1f2937;
            cursor: pointer;
            transition: background 0.15s;
        }
        .event-item:hover {
            background-color: #1e293b;
        }
        .event-item.active {
            background-color: #1e3a8a;
            border-left: 4px solid var(--accent-blue);
        }
        /* Unread, mail-client style: new/updated events from the background
           sync get a steady tint (not a blink) that persists until the event
           is opened - see markEventAsRead(), called from loadEvent(). They're
           also pinned to the top of the list (see sortEvents()) for as long as
           they stay unread, so nothing new needs to be hunted for by scrolling. */
        .event-item.is-new {
            background-color: rgba(16, 185, 129, 0.16);
            border-left: 3px solid var(--accent-green);
        }
        .event-item.is-new:hover {
            background-color: rgba(16, 185, 129, 0.24);
        }
        .event-item.is-new .event-title {
            font-weight: 700;
        }
        .event-title {
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
        }
        /* Position in the current sort/filter (1..N) - lets "N events found"
           in the header be checked directly against what's actually rendered. */
        .event-index {
            font-weight: 400;
            font-size: 11px;
            color: var(--text-muted);
            font-family: monospace;
        }
        .event-sub {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .event-timestamp {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
            opacity: 0.75;
        }
        .group-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px;
            background: #0d1424;
            border-bottom: 1px solid #1f2937;
            cursor: pointer;
            user-select: none;
        }
        .group-header:hover {
            background-color: #1e293b;
        }
        .group-toggle {
            color: var(--text-muted);
            font-size: 10px;
            width: 10px;
        }
        .group-imei {
            font-weight: 600;
            font-size: 13px;
            font-family: monospace;
        }
        .group-device {
            font-size: 11px;
            color: var(--text-muted);
        }
        .group-count {
            margin-left: auto;
            background: #1e3a8a;
            color: #bfdbfe;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
        }
        .grouped-item {
            padding-left: 28px;
        }
        #mainContent {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 20px 24px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
        }
        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .metric-val {
            font-size: 20px;
            font-weight: 700;
        }
        
        /* Video & Live HUD Layout */
        .workspace-split {
            display: grid;
            grid-template-columns: 480px 1fr;
            gap: 16px;
            margin-bottom: 16px;
            height: 700px;
        }
        @media (max-width: 1200px) {
            .workspace-split {
                grid-template-columns: 1fr;
                height: auto;
            }
        }
        .video-panel {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            display: flex;
            flex-direction: column;
        }
        .video-wrapper {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
        }
        video {
            width: 100%;
            height: 270px;
            display: block;
            object-fit: contain;
        }
        .hud-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 12px;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            gap: 4px;
            z-index: 10;
        }
        .hud-row {
            display: flex;
            gap: 10px;
            justify-content: space-between;
        }
        .hud-val-x { color: var(--accent-red); font-weight: bold; }
        .hud-val-y { color: var(--accent-green); font-weight: bold; }
        .hud-val-z { color: var(--accent-yellow); font-weight: bold; }
        .hud-val-horiz { color: var(--accent-pink); font-weight: bold; }
        .hud-val-mag { color: var(--accent-purple); font-weight: bold; }

        .video-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }
        .hud-panel {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .live-gauges {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            gap: 12px;
        }
        .gauge-card {
            background: #111827;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        .gauge-title {
            font-size: 11px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .gauge-number {
            font-size: 26px;
            font-weight: 700;
            font-family: monospace;
        }

        .chart-box {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: 650px;
            position: relative;
        }
        .chart-box-large {
            height: 100%;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }
        .chart-canvas-row {
            flex: 1;
            min-height: 0;
            display: flex;
            gap: 8px;
        }
        .chart-canvas-wrap {
            position: relative;
            flex: 1;
            min-width: 0;
        }

        /* Custom scrollbars for the zoomed chart - Chart.js's own pan/zoom (drag,
           wheel) has no visible track, so these give a draggable handle showing how
           far the current view is scrolled through the full X (time) / Y (g) range. */
        .chart-vscroll, .chart-hscroll {
            background: #0b0f19;
            border: 1px solid var(--border);
            border-radius: 7px;
            position: relative;
            flex-shrink: 0;
        }
        .chart-vscroll { width: 14px; }
        .chart-hscroll { height: 14px; margin-top: 8px; }
        .chart-vscroll-thumb, .chart-hscroll-thumb {
            position: absolute;
            background: #4b5563;
            border-radius: 6px;
            cursor: grab;
            transition: background 0.15s;
        }
        .chart-vscroll-thumb { left: 2px; right: 2px; min-height: 16px; }
        .chart-hscroll-thumb { top: 2px; bottom: 2px; min-width: 16px; }
        .chart-vscroll-thumb:hover, .chart-hscroll-thumb:hover,
        .chart-vscroll-thumb.dragging, .chart-hscroll-thumb.dragging {
            background: var(--accent-blue);
            cursor: grabbing;
        }
        .toolbar {
            display: flex;
            gap: 10px;
            margin-top: 12px;
            align-items: center;
        }
        .btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.9; }
        .btn-secondary {
            background: #374151;
            color: #d1d5db;
        }
        .btn-secondary:hover { background: #4b5563; }
        .sync-tip {
            font-size: 12px;
            color: var(--text-muted);
            margin-left: auto;
        }

        /* --- playback speed controller --- */
        .speed-panel {
            margin-top: 12px;
            padding: 12px;
            background: #111827;
            border: 1px solid var(--border);
            border-radius: 6px;
        }
        .speed-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
        }
        .speed-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }
        .speed-value {
            font-family: monospace;
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-cyan);
        }
        .speed-slider {
            width: 100%;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }
        .speed-presets {
            display: flex;
            gap: 6px;
            margin-top: 10px;
        }
        .speed-preset {
            flex: 1;
            background: #1f2937;
            color: var(--text-muted);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 5px 0;
            font-size: 11px;
            font-family: monospace;
            cursor: pointer;
            transition: 0.15s;
        }
        .speed-preset:hover { background: #374151; color: #fff; }
        .speed-preset.active {
            background: var(--accent-cyan);
            border-color: var(--accent-cyan);
            color: #06202b;
            font-weight: 700;
        }
        .speed-hint {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
        }

        /* --- current-sample readout --- */
        .readout-time {
            font-family: monospace;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 10px;
            text-align: center;
        }
        .swatch {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 2px;
            margin-right: 5px;
            vertical-align: middle;
        }

        /* --- inline toggle switches in the chart toolbar --- */
        .switch {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
            cursor: pointer;
            user-select: none;
        }
        .switch input {
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }
        .switch.disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }
        .switch.disabled input {
            cursor: not-allowed;
        }

        /* --- Event GPS Track modal --- */
        .gps-modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.6);
            align-items: center;
            justify-content: center;
            z-index: 100;
        }
        .gps-modal-overlay.open {
            display: flex;
        }
        .gps-modal {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            width: min(900px, 92vw);
            padding: 18px 20px;
        }
        .gps-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .gps-modal-header h3 {
            margin: 0;
            font-size: 15px;
            color: var(--accent-blue);
        }
        .gps-modal-status {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 10px;
            min-height: 16px;
        }
        .gps-modal-chart-wrap {
            position: relative;
            height: 460px;
        }
        #gpsMap {
            position: absolute;
            inset: 0;
            border-radius: 6px;
            background: #0b0f19;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="sidebar-header">
            <div class="header-row">
                <h2>Telemetry Events</h2>
                <select id="datasetSelect" class="dataset-select" onchange="switchDataset(this.value)">
                    <option value="no_crash">No Crash</option>
                    <option value="crash">Crash</option>
                </select>
            </div>
            <div id="fileCount" style="font-size: 12px; color: var(--text-muted)">Loading events...</div>
            <input type="text" id="searchBox" class="search-box" placeholder="Search Event ID, Type, or IMEI..." oninput="filterEvents()">
            <div class="header-row" style="margin-top: 8px; margin-bottom: 0;">
                <label class="switch"><input type="checkbox" id="groupByImeiCheckbox" onchange="toggleGroupByImei(this)">
                    Group by IMEI</label>
                <label style="font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 6px;">
                    Sort:
                    <select id="sortSelect" class="dataset-select" onchange="setSortBy(this.value)">
                        <option value="time_desc">Time (Newest)</option>
                    </select>
                </label>
            </div>
        </div>
        <ul id="eventsList"></ul>
    </div>

    <div id="mainContent">
        <header>
            <div>
                <h1 id="selectedTitle" style="margin: 0; font-size: 20px;">Select an Event</h1>
                <div id="selectedMeta" style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Choose an event from sidebar to load video and 100Hz G-Force telemetry</div>
            </div>
            <div style="display: flex; gap: 8px;">
                <button class="btn btn-secondary" onclick="showGpsTrack()">📍 Event GPS Track</button>
                <button class="btn btn-secondary" onclick="exportCurrentCSV()">📥 Download CSV</button>
                <button class="btn btn-secondary" onclick="chart.resetZoom(); syncScrollbars();">🔍 Reset Zoom</button>
            </div>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Peak Shock (gravity removed)</div>
                <div class="metric-val" id="metricPeak" style="color: var(--accent-purple)">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Shock Peak At</div>
                <div class="metric-val" id="metricPeakTime">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Camera Mounting</div>
                <div class="metric-val" id="metricMount" style="font-size: 14px; color: var(--accent-blue)">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Duration / Samples</div>
                <div class="metric-val" id="metricDuration" style="font-size: 15px;">--</div>
            </div>
        </div>

        <div class="workspace-split">
            <!-- Video Player Column -->
            <div class="video-panel">
                <div class="video-wrapper">
                    <video id="eventVideo" controls preload="auto"></video>
                    <div id="hudOverlay" class="hud-overlay" style="display: none;">
                        <div class="hud-row"><span>TIME:</span><span id="hudTime">0.00s</span></div>
                        <div class="hud-row"><span>X:</span><span id="hudX" class="hud-val-x">0.000g</span></div>
                        <div class="hud-row"><span>Y:</span><span id="hudY" class="hud-val-y">0.000g</span></div>
                        <div class="hud-row"><span>Z:</span><span id="hudZ" class="hud-val-z">0.000g</span></div>
                    </div>
                </div>
                <div class="video-controls">
                    <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play / Pause</button>
                    <button class="btn btn-secondary" onclick="jumpToShock()">⚡ Jump to Shock Impact</button>
                </div>

                <div class="speed-panel">
                    <div class="speed-row">
                        <span class="speed-label">Playback speed</span>
                        <span class="speed-value" id="speedValue">1.00x</span>
                    </div>
                    <input type="range" id="speedSlider" class="speed-slider"
                           min="0.05" max="4" step="0.05" value="1"
                           oninput="setSpeed(Number(this.value))">
                    <div class="speed-presets">
                        <button class="speed-preset" data-rate="0.1"  onclick="setSpeed(0.1)">0.1x</button>
                        <button class="speed-preset" data-rate="0.25" onclick="setSpeed(0.25)">0.25x</button>
                        <button class="speed-preset" data-rate="0.5"  onclick="setSpeed(0.5)">0.5x</button>
                        <button class="speed-preset active" data-rate="1" onclick="setSpeed(1)">1x</button>
                        <button class="speed-preset" data-rate="2"    onclick="setSpeed(2)">2x</button>
                        <button class="speed-preset" data-rate="4"    onclick="setSpeed(4)">4x</button>
                    </div>
                    <div class="speed-hint">Slow the video down to read the 100 Hz trace sample by sample.</div>
                </div>
            </div>

            <!-- LARGE CHART on the right -->
            <div class="chart-box chart-box-large">
                <div class="chart-canvas-row">
                    <div class="chart-canvas-wrap">
                        <canvas id="telemetryChart"></canvas>
                    </div>
                    <div class="chart-vscroll" id="chartVScrollTrack">
                        <div class="chart-vscroll-thumb" id="chartVScrollThumb"></div>
                    </div>
                </div>
                <div class="chart-hscroll" id="chartHScrollTrack">
                    <div class="chart-hscroll-thumb" id="chartHScrollThumb"></div>
                </div>
            </div>
        </div>

        <!-- Real-time HUD Gauges below the video/chart split -->
        <div class="hud-panel">
            <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--accent-blue);">
                SYNCHRONIZED TELEMETRY READOUT (100 Hz)
            </div>
            <div class="live-gauges">
                <div class="gauge-card">
                    <div class="gauge-title"><span class="swatch" style="background:#ef4444"></span>X</div>
                    <div class="gauge-number" id="gaugeX" style="color: var(--accent-red)">--</div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-title"><span class="swatch" style="background:#10b981"></span>Y</div>
                    <div class="gauge-number" id="gaugeY" style="color: var(--accent-green)">--</div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-title"><span class="swatch" style="background:#f59e0b"></span>Z</div>
                    <div class="gauge-number" id="gaugeZ" style="color: var(--accent-yellow)">--</div>
                </div>
            </div>
            <div class="readout-time" id="readoutTime">--</div>
            <div id="mountInfo" style="font-size: 11px; color: var(--text-muted); margin-top: 10px; line-height: 1.5;"></div>
        </div>

        <div class="toolbar">
            <button class="btn btn-secondary" onclick="toggleDataset(0)">Toggle X</button>
            <button class="btn btn-secondary" onclick="toggleDataset(1)">Toggle Y</button>
            <button class="btn btn-secondary" onclick="toggleDataset(2)">Toggle Z</button>
            <label class="switch"><input type="checkbox" checked onchange="toggleLiveRender(this)">
                Draw as video plays</label>
            <label class="switch" id="gravitySwitchLabel"><input type="checkbox" checked id="gravityCheckbox" onchange="toggleGravity(this)">
                Subtract gravity</label>
            <label class="switch" title="Plots the sensor's native milli-g output directly, with no gravity subtraction or unit conversion applied.">
                <input type="checkbox" onchange="toggleRawMode(this)">
                Raw sensor data (mg)</label>
            <span class="sync-tip">
                Click chart to jump video. Press Space to play/pause, [ ] to change speed, arrow keys to step frame-by-frame.
            </span>
        </div>
    </div>

    <div id="gpsModalOverlay" class="gps-modal-overlay" onclick="if (event.target === this) closeGpsModal()">
        <div class="gps-modal">
            <div class="gps-modal-header">
                <h3>Event GPS Track &middot; &plusmn;100s</h3>
                <button class="btn btn-secondary" onclick="closeGpsModal()">✕ Close</button>
            </div>
            <div id="gpsModalStatus" class="gps-modal-status"></div>
            <div class="gps-modal-chart-wrap">
                <div id="gpsMap"></div>
            </div>
        </div>
    </div>

    <script>
        let allEvents = [];
        // Mail-client style "unread" tracking: every event_id that has arrived
        // or been updated via the background sync since it was last opened.
        // Accumulates across ticks (so several arrivals all stay marked, not
        // just the latest one) and is cleared one event at a time - by
        // markEventAsRead(), from loadEvent() - never on a timer.
        let unreadEventIds = new Set();

        // Only one option today (time, newest first); more (severity, speed, ...)
        // can be added to the #sortSelect dropdown and this switch later without
        // touching how sorting is applied.
        let currentSort = 'time_desc';

        function setSortBy(value) {
            currentSort = value;
            allEvents = sortEvents(allEvents);
            filterEvents();
        }

        // Applies the chosen sort, then always pins unread events to the very
        // top (in the order they arrived) regardless of that sort - so nothing
        // unread ever needs to be hunted for by scrolling. An event drops out
        // of the pin the moment it's opened (see markEventAsRead()).
        function sortEvents(events) {
            const sorted = [...events].sort((a, b) => {
                if (currentSort === 'time_desc') {
                    const at = a.event_time_ms ?? -Infinity;
                    const bt = b.event_time_ms ?? -Infinity;
                    if (at !== bt) return bt - at;
                }
                // Tie-break (and the only rule if event_time_ms is missing on both):
                // numeric event ID, newest/highest first.
                return (Number(b.event_id) || 0) - (Number(a.event_id) || 0);
            });
            if (unreadEventIds.size === 0) return sorted;
            const isUnread = (ev) => unreadEventIds.has(ev.event_id);
            return [...sorted.filter(isUnread), ...sorted.filter(ev => !isUnread(ev))];
        }

        // Called when an event is opened (loadEvent) - the "read" action. Drops
        // the highlight in place immediately, without a full list re-render, so
        // the list doesn't reorder/jump out from under the click; the next
        // natural re-render (search, sort, or the next background refresh)
        // reflects the event's normal (unpinned) sort position for good.
        function markEventAsRead(ev) {
            if (!unreadEventIds.has(ev.event_id)) return;
            unreadEventIds.delete(ev.event_id);
            document
                .querySelectorAll(`.event-item[data-event-id="${CSS.escape(String(ev.event_id))}"]`)
                .forEach(el => el.classList.remove('is-new'));
        }
        let currentEvent = null;
        let currentDataset = 'no_crash';   // 'no_crash' = decision=no, 'crash' = decision=yes
        let currentData = null;
        let chart = null;
        let playheadTime = null;
        let activeRecord = null;
        let liveRender = true;      // draw the trace progressively as the video plays
        let removeGravity = true;   // plot X/Y/Z with the measured baseline subtracted
        let rawMode = false;        // plot the sensor's native milli-g output, unprocessed
        let gpsMap = null;
        let gpsPathLayer = null;
        const videoEl = document.getElementById('eventVideo');

        // Full extent of the currently loaded event - what 100%-zoomed-out looks
        // like. The custom scrollbar thumbs are sized/positioned as a fraction of
        // this range, so they need it kept separately from the chart's live
        // (possibly zoomed) scales.x/y.min/max.
        let fullXRange = { min: 0, max: 1 };
        let fullYRange = { min: -1, max: 1 };

        // Reveals the traces only up to the playhead, so the graph draws itself as
        // the video plays.  Implemented as a clip region rather than by trimming the
        // data arrays: the line geometry is computed once, so each frame costs one
        // ctx.clip() instead of a full Chart.js data update.
        const progressiveRevealPlugin = {
            id: 'progressiveReveal',
            beforeDatasetsDraw(c) {
                if (!liveRender || playheadTime === null || playheadTime === undefined) return;
                const { ctx, chartArea: { top, bottom, left, right }, scales: { x } } = c;
                const cut = Math.min(right, Math.max(left, x.getPixelForValue(playheadTime)));
                ctx.save();
                ctx.beginPath();
                ctx.rect(left, top, cut - left, bottom - top);
                ctx.clip();
            },
            afterDatasetsDraw(c) {
                if (!liveRender || playheadTime === null || playheadTime === undefined) return;
                c.ctx.restore();
            }
        };

        // Vertical playhead plus a dot marking the exact X/Y/Z value at the frame
        // the video is showing.
        const playheadPlugin = {
            id: 'playheadLine',
            afterDatasetsDraw(c) {
                if (playheadTime === null || playheadTime === undefined) return;
                const { ctx, chartArea: { top, bottom, left, right }, scales: { x, y } } = c;
                const xPos = x.getPixelForValue(playheadTime);
                if (xPos >= left && xPos <= right) {
                    ctx.save();

                    // Glowing vertical playhead bar
                    ctx.shadowColor = '#06b6d4';
                    ctx.shadowBlur = 8;
                    ctx.beginPath();
                    ctx.strokeStyle = '#22d3ee';
                    ctx.lineWidth = 2.5;
                    ctx.moveTo(xPos, top);
                    ctx.lineTo(xPos, bottom);
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Playhead marker on top
                    ctx.fillStyle = '#06b6d4';
                    ctx.beginPath();
                    ctx.moveTo(xPos - 6, top);
                    ctx.lineTo(xPos + 6, top);
                    ctx.lineTo(xPos, top + 8);
                    ctx.closePath();
                    ctx.fill();

                    // Highlight the exact X/Y/Z sample at the current video moment
                    if (activeRecord) {
                        const drawDot = (val, color) => {
                            if (val === null || val === undefined) return;
                            const yPos = y.getPixelForValue(val);
                            ctx.beginPath();
                            ctx.arc(xPos, yPos, 5, 0, Math.PI * 2);
                            ctx.fillStyle = color;
                            ctx.strokeStyle = '#ffffff';
                            ctx.lineWidth = 2;
                            ctx.fill();
                            ctx.stroke();
                        };
                        const v = axisValues(activeRecord);
                        if (c.isDatasetVisible(0)) drawDot(v.x, '#ef4444');
                        if (c.isDatasetVisible(1)) drawDot(v.y, '#10b981');
                        if (c.isDatasetVisible(2)) drawDot(v.z, '#f59e0b');
                    }
                    ctx.restore();
                }
            }
        };

        // X/Y/Z for a record, honouring the "raw sensor data" and "subtract gravity"
        // switches. Raw mode plots the sensor's native milli-g integers exactly as
        // they came off the device - no unit conversion, no gravity subtraction -
        // and takes priority over the gravity switch, which does not apply to it.
        function axisValues(rec) {
            if (rawMode) return { x: rec.x_mg, y: rec.y_mg, z: rec.z_mg };
            return removeGravity
                ? { x: rec.x_dyn_g, y: rec.y_dyn_g, z: rec.z_dyn_g }
                : { x: rec.x_g, y: rec.y_g, z: rec.z_g };
        }

        // Formats one axis value for tooltips/HUD/gauges, unit-aware.
        function formatAxisValue(v) {
            return rawMode ? `${Math.round(v)} mg` : `${v.toFixed(3)} g`;
        }

        const ctx = document.getElementById('telemetryChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'X', data: [], borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0 },
                    { label: 'Y', data: [], borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 },
                    { label: 'Z', data: [], borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            plugins: [progressiveRevealPlugin, playheadPlugin],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                normalized: true,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#e5e7eb' } },
                    tooltip: {
                        callbacks: {
                            title: (items) => `t = ${items[0].parsed.x.toFixed(2)} s`,
                            label: (item) => `${item.dataset.label}: ${formatAxisValue(item.parsed.y)}`
                        }
                    },
                    zoom: {
                        pan: { enabled: true, mode: 'xy', onPanComplete: () => syncScrollbars() },
                        zoom: {
                            wheel: { enabled: true },
                            drag: { enabled: true, backgroundColor: 'rgba(6, 182, 212, 0.2)' },
                            mode: 'xy',
                            onZoomComplete: () => syncScrollbars()
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Relative Time (seconds)', color: '#9ca3af' },
                        grid: { color: '#1f2937' },
                        ticks: { color: '#9ca3af', maxTicksLimit: 22 }
                    },
                    y: {
                        title: { display: true, text: 'Acceleration (g)', color: '#9ca3af' },
                        grid: { color: '#1f2937' },
                        ticks: { color: '#9ca3af' }
                    }
                }
            }
        });

        setupScrollbarDrag(
            document.getElementById('chartHScrollThumb'),
            document.getElementById('chartHScrollTrack'), 'x');
        setupScrollbarDrag(
            document.getElementById('chartVScrollThumb'),
            document.getElementById('chartVScrollTrack'), 'y');

        // Click on chart to jump video to that moment
        ctx.canvas.addEventListener('click', (e) => {
            if (!chart || !currentData) return;
            const rect = ctx.canvas.getBoundingClientRect();
            const xPixel = e.clientX - rect.left;
            const clickedTime = chart.scales.x.getValueForPixel(xPixel);
            if (clickedTime !== undefined && clickedTime >= 0 && clickedTime <= currentData.duration_s) {
                videoEl.currentTime = clickedTime;
                syncAtTime(clickedTime);
            }
        });

        function syncAtTime(t) {
            if (!currentData || !currentData.records.length) return;
            playheadTime = t;

            // Uniform 10 ms sampling, so the sample for a video time is a direct index.
            const rate = currentData.sample_rate_hz || 100;
            const idx = Math.min(Math.max(0, Math.round(t * rate)), currentData.records.length - 1);
            const rec = currentData.records[idx];
            activeRecord = rec;

            if (rec) {
                const signed = (v) => `${v >= 0 ? '+' : ''}${rawMode ? Math.round(v) + 'mg' : v.toFixed(3) + 'g'}`;
                const v = axisValues(rec);

                // HUD over the video: the exact X/Y/Z sample at this frame
                document.getElementById('hudTime').textContent = `${t.toFixed(2)}s`;
                document.getElementById('hudX').textContent = signed(v.x);
                document.getElementById('hudY').textContent = signed(v.y);
                document.getElementById('hudZ').textContent = signed(v.z);

                // Large readout panel
                document.getElementById('gaugeX').textContent = signed(v.x);
                document.getElementById('gaugeY').textContent = signed(v.y);
                document.getElementById('gaugeZ').textContent = signed(v.z);
                document.getElementById('readoutTime').textContent =
                    `t = ${t.toFixed(2)} s  ·  sample #${rec.idx}  ·  ${rec.datetime} UTC`;
            }

            // Cheap: geometry is already computed, this just repaints + re-clips.
            chart.draw();
        }

        // --- playback speed -----------------------------------------------------------
        function setSpeed(rate) {
            videoEl.playbackRate = rate;
            document.getElementById('speedValue').textContent = `${rate.toFixed(2)}x`;
            const slider = document.getElementById('speedSlider');
            if (Number(slider.value) !== rate) slider.value = rate;
            document.querySelectorAll('.speed-preset').forEach(b => {
                b.classList.toggle('active', Number(b.dataset.rate) === rate);
            });
        }

        function toggleLiveRender(el) {
            liveRender = el.checked;
            chart.draw();
        }

        function toggleGravity(el) {
            removeGravity = el.checked;
            if (currentData) applyChartData(currentData);
            if (playheadTime !== null) syncAtTime(playheadTime);
        }

        // Raw mode plots the sensor's native milli-g output unprocessed, so the
        // gravity switch has nothing to apply to - grey it out rather than leaving
        // it checked-but-inert, which would look like it's doing something.
        function toggleRawMode(el) {
            rawMode = el.checked;
            document.getElementById('gravityCheckbox').disabled = rawMode;
            document.getElementById('gravitySwitchLabel').classList.toggle('disabled', rawMode);
            if (currentData) applyChartData(currentData);
            if (playheadTime !== null) syncAtTime(playheadTime);
        }

        // timeupdate only fires ~4x a second, which is far too coarse to advance a
        // 100 Hz trace smoothly, so playback is driven by requestAnimationFrame and
        // timeupdate is only a fallback for when the tab is throttled.
        videoEl.addEventListener('timeupdate', () => {
            if (videoEl.paused) syncAtTime(videoEl.currentTime);
        });

        let rafHandle = null;
        function startRenderLoop() {
            if (rafHandle !== null) return;
            const step = () => {
                if (videoEl.paused || videoEl.ended) {
                    rafHandle = null;
                    return;
                }
                syncAtTime(videoEl.currentTime);
                rafHandle = requestAnimationFrame(step);
            };
            rafHandle = requestAnimationFrame(step);
        }

        videoEl.addEventListener('play', startRenderLoop);
        videoEl.addEventListener('pause', () => {
            if (rafHandle !== null) { cancelAnimationFrame(rafHandle); rafHandle = null; }
            syncAtTime(videoEl.currentTime);
        });
        videoEl.addEventListener('seeked', () => syncAtTime(videoEl.currentTime));

        // Loading a new src resets playbackRate to 1, so the chosen speed has to be
        // re-applied once the new clip's metadata is in.
        videoEl.addEventListener('loadedmetadata', () => {
            videoEl.playbackRate = Number(document.getElementById('speedSlider').value);
            syncAtTime(0);
        });

        function togglePlay() {
            if (videoEl.paused) {
                videoEl.play();
            } else {
                videoEl.pause();
            }
        }

        function jumpToShock() {
            if (currentData && currentData.peak_record) {
                const shockTime = currentData.peak_record.rel_time_s;
                videoEl.currentTime = Math.max(0, shockTime - 1.0); // 1 sec before impact
                videoEl.play();
            }
        }

        // `silent` is the 10s background refresh: it must never yank the event
        // the user is currently reviewing, so it only touches the list, and only
        // when sync_server.py actually added something.
        async function loadEventsList({ silent = false } = {}) {
            try {
                const res = await fetch(`/api/events?dataset=${currentDataset}`);
                const events = await res.json();

                if (silent) {
                    const previousIds = new Set(allEvents.map(ev => ev.event_id));
                    const added = events.filter(ev => !previousIds.has(ev.event_id));
                    if (added.length === 0) return;

                    // Accumulate rather than replace: an event arriving this tick
                    // shouldn't un-mark one still unread from a previous tick.
                    // Set before sorting - sortEvents() pins whatever is in
                    // unreadEventIds to the top regardless of the chosen sort.
                    added.forEach(ev => unreadEventIds.add(ev.event_id));
                    allEvents = sortEvents(events);
                    updateFileCount(added.length);
                    filterEvents();   // re-renders honouring the search box, highlights the new ones

                    // Nothing selected yet (empty dataset on first load): the
                    // first event to arrive can safely open itself.
                    if (!currentEvent && allEvents.length > 0) loadEvent(allEvents[0]);
                    return;
                }

                allEvents = sortEvents(events);
                updateFileCount(0);
                renderEventsList(allEvents);
                if (allEvents.length > 0) {
                    loadEvent(allEvents[0]);
                } else {
                    clearEventView();
                }
            } catch (e) {
                console.error('Failed to load events:', e);
            }
        }

        function updateFileCount(added) {
            const el = document.getElementById('fileCount');
            const base = allEvents.length > 0 ? `${allEvents.length} events found` : 'No events found';
            el.innerHTML = added > 0
                ? `${base} <span style="color: var(--accent-green)">&middot; +${added} new</span>`
                : base;
        }

        // Resets the main panel when switching datasets to an empty list, or before
        // the first event of the new dataset has loaded.
        function clearEventView() {
            currentEvent = null;
            currentData = null;
            playheadTime = null;
            activeRecord = null;
            document.getElementById('selectedTitle').textContent = 'Select an Event';
            document.getElementById('selectedMeta').textContent =
                'Choose an event from sidebar to load video and 100Hz G-Force telemetry';
            document.getElementById('metricDuration').textContent = '--';
            document.getElementById('metricPeak').textContent = '--';
            document.getElementById('metricPeakTime').textContent = '--';
            document.getElementById('metricMount').textContent = '--';
            document.getElementById('mountInfo').textContent = '';
            document.getElementById('hudOverlay').style.display = 'none';
            videoEl.pause();
            videoEl.removeAttribute('src');
            videoEl.load();
            chart.data.datasets.forEach(ds => { ds.data = []; });
            fullXRange = { min: 0, max: 1 };
            fullYRange = { min: -1, max: 1 };
            chart.options.scales.x.min = fullXRange.min;
            chart.options.scales.x.max = fullXRange.max;
            chart.options.scales.y.min = fullYRange.min;
            chart.options.scales.y.max = fullYRange.max;
            chart.update();
            chart.resetZoom();
            syncScrollbars();
        }

        // Dropdown next to "Telemetry Events": No Crash (decision=no) vs
        // Crash (decision=yes) - two independent datasets, never mixed.
        function switchDataset(value) {
            currentDataset = value;
            document.getElementById('searchBox').value = '';
            document.getElementById('fileCount').textContent = 'Loading events...';
            unreadEventIds = new Set();   // event IDs don't carry across datasets
            allEvents = [];
            clearEventView();
            loadEventsList();
        }

        // Matches sync_server.py's poll interval: the server refreshes the JSONs
        // every 10s, the dashboard picks the new events up on the same cadence.
        const REFRESH_INTERVAL_MS = 10000;

        async function init() {
            await loadEventsList();
            setInterval(() => loadEventsList({ silent: true }), REFRESH_INTERVAL_MS);
        }

        // Keyboard: space toggles play, [ and ] step the speed, arrows nudge one frame.
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            const rates = [0.05, 0.1, 0.25, 0.5, 1, 2, 4];
            if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
            else if (e.key === '[') {
                const i = rates.findIndex(r => r >= videoEl.playbackRate);
                setSpeed(rates[Math.max(0, i - 1)]);
            } else if (e.key === ']') {
                const i = rates.findIndex(r => r >= videoEl.playbackRate);
                setSpeed(rates[Math.min(rates.length - 1, i + 1)]);
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                e.preventDefault();
                const rate = (currentData && currentData.sample_rate_hz) || 100;
                const delta = (e.key === 'ArrowRight' ? 1 : -1) / rate;
                videoEl.pause();
                videoEl.currentTime = Math.max(0, videoEl.currentTime + delta);
            }
        });

        // Sidebar can list events flat (default) or bucketed by IMEI, so events
        // from the same vehicle/device can be reviewed together. Which IMEI groups
        // are expanded is kept separately from the event list itself, so toggling
        // a group or re-filtering doesn't collapse groups the user already opened.
        let groupByImei = false;
        let lastRenderedEvents = [];
        const expandedImeiGroups = new Set();

        function toggleGroupByImei(el) {
            groupByImei = el.checked;
            renderEventsList(lastRenderedEvents);
        }

        // Position of each event in the current sort order (1-based), computed
        // once per render from the full list before it gets grouped/sliced - so
        // the number stays a stable "1..N of the full list" reference whether
        // you're looking at the flat view or a single IMEI group, letting you
        // cross-check the rendered count against the "N events found" header.
        let eventIndexById = new Map();

        // event_time_ms is the midpoint of the first media clip's Start/EndDateTime
        // (see _build_events_index in view_gsensor.py) - null only for an event
        // whose media hasn't been enriched yet (see is_fully_enriched / the
        // sync's enrich pass, which exists specifically to close that gap).
        function formatEventTimestamp(ev) {
            if (!ev.event_time_ms) return null;
            const d = new Date(ev.event_time_ms);
            return d.toLocaleString(undefined, {
                year: 'numeric', month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
            });
        }

        function buildEventItem(ev) {
            const li = document.createElement('li');
            li.dataset.eventId = ev.event_id;   // lets markEventAsRead() find this row directly
            li.className = 'event-item'
                + (currentEvent && currentEvent.filename === ev.filename ? ' active' : '')
                + (unreadEventIds.has(ev.event_id) ? ' is-new' : '');
            const subtitle = ev.imei
                ? `IMEI: ${ev.imei}${ev.device_id ? ' &middot; ' + ev.device_id : ''}`
                : `${ev.filename.substring(0, 32)}...`;
            const index = eventIndexById.get(ev.event_id);
            const timestamp = formatEventTimestamp(ev);
            li.innerHTML = `
                <div class="event-title">
                    <span><span class="event-index">${index != null ? '#' + index : ''}</span> Event #${ev.event_id}</span>
                    <span style="color: var(--accent-yellow)">${ev.event_type}</span>
                </div>
                <div class="event-sub">${subtitle}</div>
                <div class="event-timestamp">${timestamp || 'Timestamp pending (media still syncing)'}</div>
            `;
            li.onclick = () => loadEvent(ev);
            return li;
        }

        function renderEventsList(events) {
            lastRenderedEvents = events;
            const listEl = document.getElementById('eventsList');
            listEl.innerHTML = '';

            // 1-based position in this exact list, fixed before grouping/slicing
            // touches it - the number shown in the UI, so "N events found" above
            // and the highest #N row in the list are directly comparable.
            eventIndexById = new Map(events.map((ev, i) => [ev.event_id, i + 1]));

            if (!groupByImei) {
                // Previously capped at the first 200 - silently hid the rest of
                // the list while the header above still (correctly) reported the
                // full count. Every event in the current sort/filter is rendered
                // now; the browser handles a few thousand simple <li> rows fine.
                events.forEach((ev) => listEl.appendChild(buildEventItem(ev)));
                return;
            }

            const groups = new Map();
            events.forEach((ev) => {
                const key = ev.imei || 'Unknown IMEI';
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(ev);
            });

            // Devices with the most events surface first; ties break alphabetically.
            const sortedKeys = [...groups.keys()].sort((a, b) => {
                const diff = groups.get(b).length - groups.get(a).length;
                return diff !== 0 ? diff : a.localeCompare(b);
            });

            sortedKeys.forEach((imei) => {
                const groupEvents = groups.get(imei);
                const deviceId = groupEvents.find(ev => ev.device_id)?.device_id || '';
                const expanded = expandedImeiGroups.has(imei);

                const header = document.createElement('li');
                header.className = 'group-header';
                header.innerHTML = `
                    <span class="group-toggle">${expanded ? '▾' : '▸'}</span>
                    <span class="group-imei">${imei}</span>
                    <span class="group-device">${deviceId}</span>
                    <span class="group-count">${groupEvents.length}</span>
                `;
                header.onclick = () => {
                    if (expandedImeiGroups.has(imei)) expandedImeiGroups.delete(imei);
                    else expandedImeiGroups.add(imei);
                    renderEventsList(events);
                };
                listEl.appendChild(header);

                if (expanded) {
                    groupEvents.forEach((ev) => {
                        const item = buildEventItem(ev);
                        item.classList.add('grouped-item');
                        listEl.appendChild(item);
                    });
                }
            });
        }

        function filterEvents() {
            const q = document.getElementById('searchBox').value.toLowerCase();
            const filtered = allEvents.filter(ev =>
                String(ev.event_id).toLowerCase().includes(q) ||
                ev.event_type.toLowerCase().includes(q) ||
                ev.filename.toLowerCase().includes(q) ||
                (ev.imei || '').toLowerCase().includes(q) ||
                (ev.device_id || '').toLowerCase().includes(q)
            );
            renderEventsList(filtered);
        }

        // Builds the three X/Y/Z series as {x, y} points.  The x scale is linear and
        // `parsing: false`, so Chart.js consumes these directly without re-parsing -
        // that is what keeps the per-frame redraw cheap at 2100 samples x 3 series.
        function applyChartData(parsed) {
            const series = [[], [], []];
            let yMin = Infinity, yMax = -Infinity;
            for (const r of parsed.records) {
                const v = axisValues(r);
                series[0].push({ x: r.rel_time_s, y: v.x });
                series[1].push({ x: r.rel_time_s, y: v.y });
                series[2].push({ x: r.rel_time_s, y: v.z });
                if (v.x < yMin) yMin = v.x; if (v.x > yMax) yMax = v.x;
                if (v.y < yMin) yMin = v.y; if (v.y > yMax) yMax = v.y;
                if (v.z < yMin) yMin = v.z; if (v.z > yMax) yMax = v.z;
            }
            for (let i = 0; i < 3; i++) chart.data.datasets[i].data = series[i];

            // Pad the vertical range so the trace doesn't touch the top/bottom edge,
            // with a floor so a near-flat clip still gets a sane window instead of a
            // razor-thin one that makes the vertical scrollbar thumb meaningless.
            // The floor is unit-aware: 0.5 g == 500 mg, so raw mode isn't left with
            // a window a thousand times too tight for its own units.
            const flatFloor = rawMode ? 500 : 0.5;
            const span = Math.max(yMax - yMin, flatFloor);
            const pad = span * 0.1;
            fullXRange = { min: 0, max: parsed.duration_s };
            fullYRange = { min: yMin - pad, max: yMax + pad };

            chart.options.scales.x.min = fullXRange.min;
            chart.options.scales.x.max = fullXRange.max;
            chart.options.scales.y.min = fullYRange.min;
            chart.options.scales.y.max = fullYRange.max;
            chart.options.scales.y.title.text = rawMode ? 'Raw Sensor Output (mg)' : 'Acceleration (g)';
            chart.update();
            chart.resetZoom();
            syncScrollbars();
        }

        // --- custom scrollbars for the chart's pan/zoom state -------------------------
        // Chart.js's own pan (drag) and zoom (wheel/drag-box) have no visible track, so
        // these give a draggable handle showing how far the current view has scrolled
        // through the full X (time) / Y (g) range, and let you drag to pan directly.
        function syncScrollbars() {
            if (!chart || !chart.scales.x || !chart.scales.y) return;
            const xMin = chart.scales.x.min, xMax = chart.scales.x.max;
            const yMin = chart.scales.y.min, yMax = chart.scales.y.max;
            const xSpan = (fullXRange.max - fullXRange.min) || 1;
            const ySpan = (fullYRange.max - fullYRange.min) || 1;

            const hThumb = document.getElementById('chartHScrollThumb');
            let hWidthPct = ((xMax - xMin) / xSpan) * 100;
            hWidthPct = Math.max(4, Math.min(100, hWidthPct));
            let hLeftPct = ((xMin - fullXRange.min) / xSpan) * 100;
            hLeftPct = Math.max(0, Math.min(100 - hWidthPct, hLeftPct));
            hThumb.style.width = hWidthPct + '%';
            hThumb.style.left = hLeftPct + '%';

            // Track runs top (= fullYRange.max) to bottom (= fullYRange.min), the
            // opposite direction of the data axis, so the thumb position is inverted.
            const vThumb = document.getElementById('chartVScrollThumb');
            let vHeightPct = ((yMax - yMin) / ySpan) * 100;
            vHeightPct = Math.max(4, Math.min(100, vHeightPct));
            let vTopPct = ((fullYRange.max - yMax) / ySpan) * 100;
            vTopPct = Math.max(0, Math.min(100 - vHeightPct, vTopPct));
            vThumb.style.height = vHeightPct + '%';
            vThumb.style.top = vTopPct + '%';
        }

        // Wires one scrollbar: dragging the thumb pans the chart along `axis`;
        // clicking the bare track re-centers the current view on the click point.
        function setupScrollbarDrag(thumbEl, trackEl, axis) {
            const horizontal = axis === 'x';
            let dragging = false;
            let startPos = 0, startMin = 0, startMax = 0;

            const clampToFull = (min, max, full) => {
                const span = max - min;
                if (min < full.min) { min = full.min; max = min + span; }
                if (max > full.max) { max = full.max; min = max - span; }
                return [min, max];
            };

            thumbEl.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragging = true;
                thumbEl.classList.add('dragging');
                startPos = horizontal ? e.clientX : e.clientY;
                startMin = chart.scales[axis].min;
                startMax = chart.scales[axis].max;
            });

            document.addEventListener('mousemove', (e) => {
                if (!dragging) return;
                const rect = trackEl.getBoundingClientRect();
                const trackSize = horizontal ? rect.width : rect.height;
                if (trackSize <= 0) return;
                const pos = horizontal ? e.clientX : e.clientY;
                const full = horizontal ? fullXRange : fullYRange;
                const dataPerPx = (full.max - full.min) / trackSize;
                // Vertical track direction is inverted vs. the data axis (top = max),
                // so dragging the V-thumb down must subtract, not add.
                const deltaData = (pos - startPos) * dataPerPx * (horizontal ? 1 : -1);
                const [newMin, newMax] = clampToFull(startMin + deltaData, startMax + deltaData, full);
                chart.options.scales[axis].min = newMin;
                chart.options.scales[axis].max = newMax;
                chart.update('none');
                syncScrollbars();
            });

            document.addEventListener('mouseup', () => {
                if (!dragging) return;
                dragging = false;
                thumbEl.classList.remove('dragging');
            });

            trackEl.addEventListener('mousedown', (e) => {
                if (e.target === thumbEl) return;
                const rect = trackEl.getBoundingClientRect();
                const trackSize = horizontal ? rect.width : rect.height;
                if (trackSize <= 0) return;
                const clickFrac = (horizontal ? (e.clientX - rect.left) : (e.clientY - rect.top)) / trackSize;
                const full = horizontal ? fullXRange : fullYRange;
                const clickValue = horizontal
                    ? full.min + clickFrac * (full.max - full.min)
                    : full.max - clickFrac * (full.max - full.min);
                const span = chart.scales[axis].max - chart.scales[axis].min;
                const [newMin, newMax] = clampToFull(clickValue - span / 2, clickValue + span / 2, full);
                chart.options.scales[axis].min = newMin;
                chart.options.scales[axis].max = newMax;
                chart.update('none');
                syncScrollbars();
            });
        }

        async function loadEvent(ev) {
            currentEvent = ev;
            markEventAsRead(ev);   // opening an event is the "read" action - clears its unread highlight
            document.querySelectorAll('.event-item').forEach(el => el.classList.remove('active'));
            document
                .querySelectorAll(`.event-item[data-event-id="${CSS.escape(String(ev.event_id))}"]`)
                .forEach(el => el.classList.add('active'));
            document.getElementById('selectedTitle').textContent = `Event #${ev.event_id} (${ev.event_type})`;
            document.getElementById('selectedMeta').textContent = ev.imei
                ? `File: ${ev.filename}  ·  IMEI: ${ev.imei}${ev.device_id ? '  ·  Device: ' + ev.device_id : ''}`
                : `File: ${ev.filename}`;

            try {
                const res = await fetch(`/api/data?file=${encodeURIComponent(ev.filename)}&dataset=${currentDataset}`);
                const parsed = await res.json();
                currentData = parsed;

                document.getElementById('metricDuration').textContent =
                    `${parsed.duration_s} s / ${parsed.total_records}`;
                document.getElementById('metricPeak').textContent = `${parsed.peak_dyn_g} g`;
                document.getElementById('metricPeakTime').textContent = parsed.peak_record ? `t = ${parsed.peak_record.rel_time_s} s` : '--';

                // Mounting is still reported: it is what makes the gravity baseline
                // correct for sideways and inverted cameras.
                const o = parsed.orientation;
                document.getElementById('metricMount').textContent = o ? o.label : '--';
                const mountInfo = document.getElementById('mountInfo');
                if (o && parsed.gravity) {
                    mountInfo.innerHTML =
                        `Mounting auto-detected: <strong>${o.label}</strong>, tilt ${o.tilt_deg}&deg;. ` +
                        `Gravity baseline ${parsed.gravity.magnitude_mg} mg ` +
                        `(${parsed.gravity.x_mg}, ${parsed.gravity.y_mg}, ${parsed.gravity.z_mg}) mg ` +
                        `measured from the quiet parts of this clip [${parsed.baseline_method}].`;
                } else {
                    mountInfo.textContent = '';
                }

                applyChartData(parsed);

                const hudOverlay = document.getElementById('hudOverlay');
                if (ev.video_url) {
                    hudOverlay.style.display = 'flex';
                    videoEl.src = ev.video_url;
                    videoEl.load();
                } else {
                    hudOverlay.style.display = 'none';
                    videoEl.src = '';
                }

                syncAtTime(0);
            } catch (err) {
                console.error("Failed to load telemetry data:", err);
            }
        }

        function toggleDataset(index) {
            chart.setDatasetVisibility(index, !chart.isDatasetVisible(index));
            chart.update();
        }

        function exportCurrentCSV() {
            if (!currentData) return;
            const og = currentData.orientation, gv = currentData.gravity;
            let csv = "";
            if (og && gv) {
                csv += `# file=${currentData.file} mounting="${og.label}" `
                     + `vertical_axis=${og.vertical_axis}${og.vertical_sign > 0 ? '+' : '-'} `
                     + `tilt_deg=${og.tilt_deg} `
                     + `gravity_baseline_mg=(${gv.x_mg},${gv.y_mg},${gv.z_mg}) `
                     + `|g|=${gv.magnitude_mg}mg baseline_method=${currentData.baseline_method}\\n`;
            }
            csv += "Index,Timestamp_ms,DateTime_UTC,RelativeTime_s,X_mg,Y_mg,Z_mg,X_g,Y_g,Z_g,RawMagnitude_g,X_dyn_g,Y_dyn_g,Z_dyn_g,Dynamic_g,Horizontal_g,Vertical_g,Lateral_g,Longitudinal_g\\n";
            currentData.records.forEach(r => {
                csv += `${r.idx},${r.timestamp_ms},${r.datetime},${r.rel_time_s},${r.x_mg},${r.y_mg},${r.z_mg},${r.x_g},${r.y_g},${r.z_g},${r.mag_g},${r.x_dyn_g},${r.y_dyn_g},${r.z_dyn_g},${r.dyn_g},${r.horiz_g},${r.vert_g},${r.lat_g},${r.lon_g}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentEvent.filename.replace('.gsdata', '')}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // --- Event GPS Track modal ------------------------------------------------
        // Fetches the vehicle's live GPS breadcrumbs for a +/-100s window around the
        // event and draws the actual path driven on a real map (Leaflet + OpenStreetMap
        // tiles - free, no API key), rather than plotting lat/lon as separate time series.
        function openGpsModal() {
            document.getElementById('gpsModalOverlay').classList.add('open');
            if (gpsMap) setTimeout(() => gpsMap.invalidateSize(), 0);
        }
        function closeGpsModal() {
            document.getElementById('gpsModalOverlay').classList.remove('open');
        }
        function setGpsModalStatus(msg) {
            document.getElementById('gpsModalStatus').textContent = msg;
        }

        function initGpsMapIfNeeded() {
            if (gpsMap) return;
            gpsMap = L.map('gpsMap');
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(gpsMap);
        }

        function renderGpsMap(points, centerMs) {
            initGpsMapIfNeeded();
            if (gpsPathLayer) gpsMap.removeLayer(gpsPathLayer);

            const sorted = [...points].sort((a, b) => a.tsInMilliSeconds - b.tsInMilliSeconds);
            const latLngs = sorted.map(p => [p.latitude, p.longitude]);

            // Point closest to the event time is highlighted distinctly on the path.
            let eventIdx = 0, bestDiff = Infinity;
            sorted.forEach((p, i) => {
                const diff = Math.abs(p.tsInMilliSeconds - centerMs);
                if (diff < bestDiff) { bestDiff = diff; eventIdx = i; }
            });

            const group = L.layerGroup();
            L.polyline(latLngs, { color: '#3b82f6', weight: 4, opacity: 0.85 }).addTo(group);

            sorted.forEach((p, i) => {
                const isEvent = i === eventIdx;
                const isEndpoint = i === 0 || i === sorted.length - 1;
                const color = isEvent ? '#ef4444' : (isEndpoint ? '#10b981' : '#3b82f6');
                const marker = L.circleMarker([p.latitude, p.longitude], {
                    radius: isEvent ? 9 : (isEndpoint ? 6 : 4),
                    color: '#fff',
                    weight: isEvent ? 2 : 1,
                    fillColor: color,
                    fillOpacity: 0.9
                });
                const relS = ((p.tsInMilliSeconds - centerMs) / 1000).toFixed(1);
                marker.bindPopup(
                    `t = ${relS}s${isEvent ? ' (closest to event)' : ''}<br>` +
                    `${p.latitude.toFixed(5)}, ${p.longitude.toFixed(5)}<br>` +
                    `Speed: ${p.speed ?? '--'} mph`
                );
                marker.addTo(group);
            });

            group.addTo(gpsMap);
            gpsPathLayer = group;

            // The map container was hidden (display:none) until the modal just
            // opened, so Leaflet can't have measured it correctly at creation -
            // force a re-measure once the browser has actually painted it.
            setTimeout(() => {
                gpsMap.invalidateSize();
                gpsMap.fitBounds(L.latLngBounds(latLngs), { padding: [24, 24] });
            }, 50);
        }

        async function showGpsTrack() {
            openGpsModal();
            if (gpsPathLayer && gpsMap) { gpsMap.removeLayer(gpsPathLayer); gpsPathLayer = null; }

            if (!currentEvent || !currentEvent.device_id || !currentEvent.event_time_ms) {
                setGpsModalStatus('No event time / device ID available for this event yet - rerun the extraction script to backfill it.');
                return;
            }

            const centerMs = currentEvent.event_time_ms;
            const fromMs = centerMs - 100000;
            const toMs = centerMs + 100000;
            setGpsModalStatus('Loading live GPS track...');

            try {
                const res = await fetch(`/api/livetrack?device=${encodeURIComponent(currentEvent.device_id)}&from=${fromMs}&to=${toMs}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                const points = data.vehicleLiveTracks || [];
                if (!points.length) {
                    setGpsModalStatus('No GPS points returned for this +/-100s window.');
                    return;
                }
                renderGpsMap(points, centerMs);
                setGpsModalStatus(`${points.length} GPS points · event at ${new Date(centerMs).toISOString()}`);
            } catch (err) {
                setGpsModalStatus(`Failed to load GPS track: ${err.message}`);
            }
        }

        init();
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="View, convert, and explore SmartWitness / Drona AIM .gsdata sensor files.")
    parser.add_argument("file", nargs="?", help="Path to a single .gsdata file to inspect")
    parser.add_argument("--csv", help="Output path for converted CSV file")
    parser.add_argument("--html", help="Output path for standalone interactive HTML viewer")
    parser.add_argument("--batch-csv", nargs="?", const="csv_exports", help="Convert all .gsdata files in downloads_gsensor/ to CSV (default folder: csv_exports)")
    parser.add_argument("--serve", action="store_true", help="Launch local interactive Web Dashboard on http://localhost:8080")
    parser.add_argument("--port", type=int, default=8080, help="Port for web dashboard (default: 8080)")

    args = parser.parse_args()

    if args.batch_csv:
        batch_export_csv(args.batch_csv)
        return

    if args.serve:
        run_dashboard_server(args.port)
        return

    if not args.file:
        print("Usage:")
        print("  1. Interactive Web Dashboard: python view_gsensor.py --serve")
        print("  2. Inspect a single file:     python view_gsensor.py <file.gsdata>")
        print("  3. Convert to CSV:            python view_gsensor.py <file.gsdata> --csv output.csv")
        print("  4. Generate Standalone HTML:  python view_gsensor.py <file.gsdata> --html output.html")
        print("  5. Batch convert all to CSV:  python view_gsensor.py --batch-csv [output_dir]")
        return

    parsed = parse_gsdata(args.file)

    print(f"\n==========================================")
    print(f"File: {parsed['file']}")
    print(f"Records: {parsed['total_records']} samples ({parsed['sample_rate_hz']} Hz)")
    print(f"Duration: {parsed['duration_s']:.2f} seconds")
    if not parsed["records"]:
        print("No usable records in this file.")
        print(f"==========================================\n")
        return

    print(f"Start Time: {parsed['records'][0]['datetime']} UTC")
    print(f"End Time:   {parsed['records'][-1]['datetime']} UTC")

    o = parsed["orientation"]
    g = parsed["gravity"]
    print(f"\n-- Mounting (auto-detected) --")
    print(f"  {o['label']}, tilt {o['tilt_deg']} deg off axis")
    print(f"  Vertical axis: {o['vertical_axis'].upper()}{'+' if o['vertical_sign'] > 0 else '-'}"
          f"   Turning: {o['lateral_axis'].upper()}   Braking: {o['longitudinal_axis'].upper()}"
          f"   (axis confidence: {o['axis_confidence']})")
    if o["axis_confidence"] != "high":
        print(f"  NOTE: {o['tilt_deg']} deg tilt makes the turning/braking split uncertain. "
              f"Horizontal vs vertical stays exact.")
    print(f"  Gravity baseline removed: ({g['x_mg']}, {g['y_mg']}, {g['z_mg']}) mg, "
          f"|g| = {g['magnitude_mg']} mg  [{parsed['baseline_method']}]")

    p = parsed["peak_record"]
    print(f"\n-- Peak shock (gravity removed) --")
    print(f"  Combined dynamic |G|: {parsed['peak_dyn_g']} g at t={p['rel_time_s']} s")
    print(f"  Horizontal (brake/turn): {parsed['peak_horiz_g']} g")
    print(f"  Vertical (potholes):     {parsed['peak_vert_g']} g")
    print(f"  At the peak: turning={p['lat_g']}g, braking={p['lon_g']}g, vertical={p['vert_g']}g")
    print(f"\n  Raw peak incl. gravity (for reference only): {parsed['peak_raw_mag_g']} g")
    print(f"==========================================\n")

    if args.csv:
        export_csv(parsed, args.csv)
    if args.html:
        generate_html_viewer(parsed, args.html)

if __name__ == "__main__":
    main()
