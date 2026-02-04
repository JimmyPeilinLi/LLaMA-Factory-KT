#!/usr/bin/env python3
"""Merge pyinstrument Chrome trace and C++ sft_trace into one aligned trace file.

Both traces are converted to absolute epoch-based timestamps (microseconds)
and assigned different pid namespaces so they appear as separate "processes"
in chrome://tracing or Perfetto.

Usage (with patched worker_pool.cpp that writes metadata.start_epoch_us):
    python scripts/merge_traces.py \
        --pyinstrument in_trace_rank0_chrome.json \
        --sft sft_trace.json \
        -o merged_trace.json

Usage (manual offset, before worker_pool.cpp is rebuilt):
    python scripts/merge_traces.py \
        --pyinstrument in_trace_rank0_chrome.json \
        --sft sft_trace.json \
        --sft-epoch-us 1769875500000000 \
        -o merged_trace.json
"""

import argparse
import json
import os
import re
import sys
from typing import Iterator, Tuple

CHUNK_SIZE = 1024 * 1024
SEARCH_KEEP = 4096
TRIM_THRESHOLD = 1024 * 1024


def _iter_trace_events(path: str) -> Iterator[str]:
    """Stream-parse traceEvents from a potentially huge Chrome trace JSON."""
    if os.path.getsize(path) == 0:
        return

    with open(path, "r", encoding="utf-8") as handle:
        buf = ""
        found = False
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                if not found:
                    raise ValueError(f"traceEvents not found in {path}")
                return
            buf += chunk
            idx = buf.find('"traceEvents"')
            if idx != -1:
                bracket_idx = buf.find("[", idx)
                if bracket_idx != -1:
                    buf = buf[bracket_idx + 1:]
                    found = True
                    break
            if len(buf) > SEARCH_KEEP:
                buf = buf[-SEARCH_KEEP:]

        decoder = json.JSONDecoder()
        idx = 0
        while True:
            while True:
                if idx >= len(buf):
                    chunk = handle.read(CHUNK_SIZE)
                    if not chunk:
                        return
                    buf = buf[idx:] + chunk
                    idx = 0
                while idx < len(buf) and buf[idx] in " \r\n\t,":
                    idx += 1
                if idx < len(buf):
                    break

            if buf[idx] == "]":
                return

            try:
                _, end = decoder.raw_decode(buf, idx)
            except json.JSONDecodeError:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"truncated traceEvents array in {path}")
                buf += chunk
                continue

            yield buf[idx:end]
            idx = end
            if idx > TRIM_THRESHOLD:
                buf = buf[idx:]
                idx = 0


def _get_metadata_field(path: str, field: str):
    """Try to extract a top-level metadata field from a trace JSON without loading everything."""
    with open(path, "r", encoding="utf-8") as f:
        # Read first 4KB to find metadata
        head = f.read(4096)
    # Try to find "metadata" : { ... "field": value ... }
    m = re.search(r'"metadata"\s*:\s*\{[^}]*"' + re.escape(field) + r'"\s*:\s*([\d.eE+\-]+)', head)
    if m:
        return float(m.group(1))
    # Also check the tail of the file
    with open(path, "r", encoding="utf-8") as f:
        f.seek(max(0, os.path.getsize(path) - 4096))
        tail = f.read()
    m = re.search(r'"metadata"\s*:\s*\{[^}]*"' + re.escape(field) + r'"\s*:\s*([\d.eE+\-]+)', tail)
    if m:
        return float(m.group(1))
    return None


def merge_aligned(py_path: str, sft_path: str, output_path: str, sft_epoch_us: float | None):
    """Merge pyinstrument + sft traces with timeline alignment."""

    # --- Determine sft absolute time offset ---
    if sft_epoch_us is None:
        sft_epoch_us = _get_metadata_field(sft_path, "start_epoch_us")
    if sft_epoch_us is None:
        print("ERROR: sft_trace has no metadata.start_epoch_us and --sft-epoch-us not given.\n"
              "Rebuild with patched worker_pool.cpp or supply --sft-epoch-us manually.",
              file=sys.stderr)
        sys.exit(1)

    print(f"sft epoch offset: {sft_epoch_us:.0f} us", file=sys.stderr)

    # --- Stream-write merged output ---
    py_pid_base = 1000
    sft_pid_base = 2000

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        out.write('{"traceEvents":[\n')

        # Process name metadata events
        meta_events = [
            {"name": "process_name", "ph": "M", "pid": py_pid_base, "tid": 0,
             "args": {"name": "Python (pyinstrument)"}},
            {"name": "process_name", "ph": "M", "pid": sft_pid_base, "tid": 0,
             "args": {"name": "C++ worker_pool (NUMA 0)"}},
            {"name": "process_name", "ph": "M", "pid": sft_pid_base + 1, "tid": 0,
             "args": {"name": "C++ worker_pool (NUMA 1)"}},
        ]
        for i, me in enumerate(meta_events):
            if i > 0:
                out.write(",\n")
            out.write(json.dumps(me, separators=(",", ":")))
            count += 1

        # Write pyinstrument events (already absolute epoch us, just remap pid)
        print(f"Processing pyinstrument trace: {py_path}", file=sys.stderr)
        for raw in _iter_trace_events(py_path):
            ev = json.loads(raw)
            raw_pid = ev.get("pid", 0)
            ev["pid"] = py_pid_base + (int(raw_pid) if isinstance(raw_pid, (int, float)) else 0)
            out.write(",\n")
            out.write(json.dumps(ev, separators=(",", ":")))
            count += 1
            if count % 500000 == 0:
                print(f"  {count} events...", file=sys.stderr)

        # Write sft events (shift ts by epoch offset, remap pid)
        print(f"Processing sft trace: {sft_path}", file=sys.stderr)
        for raw in _iter_trace_events(sft_path):
            ev = json.loads(raw)
            ev["ts"] = ev["ts"] + sft_epoch_us
            ev["pid"] = sft_pid_base + ev.get("pid", 0)
            out.write(",\n")
            out.write(json.dumps(ev, separators=(",", ":")))
            count += 1
            if count % 500000 == 0:
                print(f"  {count} events...", file=sys.stderr)

        out.write('\n],\n"displayTimeUnit":"ns"\n}\n')

    print(f"Merged {count} events into {output_path}", file=sys.stderr)


# --- Legacy simple merge (no alignment) ---

def _split_trace_file(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        prefix_parts = []
        buf = ""
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"traceEvents not found in {path}")
            buf += chunk
            idx = buf.find('"traceEvents"')
            if idx != -1:
                bracket_idx = buf.find("[", idx)
                if bracket_idx != -1:
                    prefix_parts.append(buf[:bracket_idx + 1])
                    buf = buf[bracket_idx + 1:]
                    break
            if len(buf) > SEARCH_KEEP:
                prefix_parts.append(buf[:-SEARCH_KEEP])
                buf = buf[-SEARCH_KEEP:]

        prefix = "".join(prefix_parts)
        depth = 1
        in_string = False
        escape = False
        pos = 0
        while True:
            if pos >= len(buf):
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"traceEvents array not closed in {path}")
                buf = buf[pos:] + chunk
                pos = 0

            ch = buf[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        suffix = buf[pos + 1:] + handle.read()
                        return prefix, suffix
            pos += 1


def merge_traces_simple(output_path: str, inputs: list[str]) -> int:
    non_empty = [p for p in inputs if os.path.exists(p) and os.path.getsize(p) > 0]
    if not non_empty:
        raise ValueError("no non-empty trace files to merge")

    prefix, suffix = _split_trace_file(non_empty[0])
    event_count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(prefix)
        first_event = True
        for path in non_empty:
            for event_json in _iter_trace_events(path):
                if first_event:
                    handle.write("\n")
                    first_event = False
                else:
                    handle.write(",\n")
                handle.write(event_json)
                event_count += 1
        if not first_event:
            handle.write("\n")
        handle.write("]")
        handle.write(suffix)
    return event_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Chrome trace JSON files. "
        "Use --pyinstrument/--sft for aligned merge, or positional args for simple concat.")
    # Aligned merge mode
    parser.add_argument("--pyinstrument", help="Pyinstrument chrome trace JSON")
    parser.add_argument("--sft", help="sft_trace.json from C++ worker_pool")
    parser.add_argument("--sft-epoch-us", type=float, default=None,
                        help="Override: wall-clock epoch (us) of sft_trace t=0")
    # Simple merge mode
    parser.add_argument("inputs", nargs="*", help="Input trace files (simple concat mode)")
    # Common
    parser.add_argument("-o", "--output", default="merged_trace.json", help="Output merged trace file")
    args = parser.parse_args()

    if args.pyinstrument and args.sft:
        merge_aligned(args.pyinstrument, args.sft, args.output, args.sft_epoch_us)
        return 0

    if args.inputs:
        try:
            count = merge_traces_simple(args.output, args.inputs)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"merged {count} events into {args.output}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
