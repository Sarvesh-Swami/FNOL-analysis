#!/usr/bin/env python3
"""
Process SmartWitness sensor data files (.gsdata, .hdgyro, .gpsdata)
and export coordinate points (dateTime, x, y, z) into text files.

Based on the reference parser logic:
  - .gsdata:  int64 LE epoch timestamp (ms) + 3 x int16 LE (x, y, z in milli-g)
  - .hdgyro:  int64 LE epoch timestamp (ms) + 3 x int32 LE (x, y, z)
  - .gpsdata: JSON text compressed with gzip
"""

import os
import sys
import gzip
import struct
import argparse
from datetime import datetime, timezone


def parse_gsdata_buffer(buffer: bytes):
    """
    Parses uncompressed gsdata buffer.
    Format per record:
      - 8 bytes: epochTime (int64 LE)
      - 2 bytes: x (int16 LE)
      - 2 bytes: y (int16 LE)
      - 2 bytes: z (int16 LE)
    Record size: 14 bytes
    """
    record_size = 14
    points = []
    offset = 0
    total_len = len(buffer)

    while offset + record_size <= total_len:
        epoch_time, x, y, z = struct.unpack_from("<qhhh", buffer, offset)
        offset += record_size
        points.append({
            "dateTime": epoch_time,
            "x": x,
            "y": y,
            "z": z,
        })
    return {"type": "GS Data", "data": points}


def parse_hdgyro_buffer(buffer: bytes):
    """
    Parses uncompressed hdgyro buffer.
    Format per record:
      - 8 bytes: epochTime (int64 LE)
      - 4 bytes: x (int32 LE)
      - 4 bytes: y (int32 LE)
      - 4 bytes: z (int32 LE)
    Record size: 20 bytes
    """
    record_size = 20
    points = []
    offset = 0
    total_len = len(buffer)

    while offset + record_size <= total_len:
        epoch_time, x, y, z = struct.unpack_from("<qiii", buffer, offset)
        offset += record_size
        points.append({
            "dateTime": epoch_time,
            "x": x,
            "y": y,
            "z": z,
        })
    return {"type": "HD Gyro", "data": points}


def parse_sensor_file(file_path: str):
    """
    Reads a gzip-compressed sensor file and extracts records.
    """
    lower_file = file_path.lower()

    with open(file_path, "rb") as f:
        compressed_bytes = f.read()

    # Decompress gzip
    try:
        decompressed_bytes = gzip.decompress(compressed_bytes)
    except Exception as e:
        raise ValueError(f"Failed to gunzip {file_path}: {e}")

    if lower_file.endswith("gsdata"):
        return parse_gsdata_buffer(decompressed_bytes)
    elif lower_file.endswith("hdgyro"):
        return parse_hdgyro_buffer(decompressed_bytes)
    elif lower_file.endswith("gpsdata"):
        import json
        return {
            "type": "GPS JSON",
            "data": json.loads(decompressed_bytes.decode("utf-8")),
        }
    else:
        return {"message": f"Unknown file type: {file_path}"}


def format_iso_time(epoch_ms: int) -> str:
    """Converts epoch milliseconds to ISO 8601 UTC string."""
    try:
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except Exception:
        return ""


def save_points_to_txt(points: list, output_file_path: str, source_filename: str, data_type: str):
    """
    Writes extracted coordinate points to a clean text file.
    """
    with open(output_file_path, "w", encoding="utf-8") as f:
        # Header comments
        f.write(f"# Source File: {source_filename}\n")
        f.write(f"# Data Type:   {data_type}\n")
        f.write(f"# Record Count: {len(points)}\n")
        f.write("# Format: dateTime (epoch ms), dateTime (UTC ISO), x, y, z\n")
        f.write("dateTime,dateTimeISO,x,y,z\n")

        for pt in points:
            epoch = pt["dateTime"]
            iso_str = format_iso_time(epoch)
            x = pt["x"]
            y = pt["y"]
            z = pt["z"]
            f.write(f"{epoch},{iso_str},{x},{y},{z}\n")


def process_folder(input_folder: str, output_folder: str):
    """
    Processes all sensor files in input_folder and writes .txt files to output_folder.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Source folder: {input_folder}")
    print(f"Destination folder: {output_folder}")
    print("-" * 60)

    files = [
        f for f in sorted(os.listdir(input_folder))
        if f.lower().endswith((".gsdata", ".hdgyro", ".gpsdata"))
    ]

    if not files:
        print(f"No matching sensor files found in '{input_folder}'.")
        return

    processed_count = 0
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        base_name, _ = os.path.splitext(filename)
        output_txt_path = os.path.join(output_folder, f"{base_name}.txt")

        try:
            result = parse_sensor_file(input_path)
            data_type = result.get("type", "Unknown")
            data = result.get("data", [])

            if isinstance(data, list) and data_type in ("GS Data", "HD Gyro"):
                save_points_to_txt(data, output_txt_path, filename, data_type)
                print(f"[OK] Processed [{data_type}]: {filename}")
                print(f"     -> Records: {len(data):,}")
                print(f"     -> Saved:   {output_txt_path}")
                if data:
                    print(f"     -> First record: {data[0]}")
                    print(f"     -> Last record:  {data[-1]}")
                print()
                processed_count += 1
            else:
                # E.g. GPS JSON or other types
                import json
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, indent=2))
                print(f"[OK] Processed [{data_type}]: {filename} -> {output_txt_path}\n")
                processed_count += 1

        except Exception as err:
            print(f"[ERROR] Failed to process {filename}: {err}\n")

    print("-" * 60)
    print(f"Done! Successfully converted {processed_count} files into '{output_folder}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SmartWitness sensor data files (.gsdata, etc.) to text files."
    )
    parser.add_argument(
        "--input-dir",
        default="downloads_gsensor_yes",
        help="Path to folder containing sensor files (default: downloads_gsensor_yes)",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads_gsensor_yes_txt",
        help="Path to folder where output text files will be saved (default: downloads_gsensor_yes_txt)",
    )

    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)
