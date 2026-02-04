#!/usr/bin/env python3
"""Convert pyinstrument JSON profile to Chrome Trace Event Format (chrome://tracing)."""

import json
import sys


def convert(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)

    start_time_s = data["start_time"]
    events = []

    def walk(frame, offset_us: float):
        name = frame["function"]
        cat = frame.get("file_path_short", "")
        dur_us = frame["time"] * 1e6

        events.append({
            "name": name,
            "cat": cat,
            "ph": "X",  # complete event
            "ts": offset_us,
            "dur": dur_us,
            "pid": 0,
            "tid": 0,
            "args": {
                "file": frame.get("file_path", ""),
                "line": frame.get("line_no", 0),
            },
        })

        child_offset = offset_us
        children = frame.get("children", [])
        # self time before children
        children_time = sum(c["time"] for c in children)
        self_time_us = max(0, dur_us - children_time * 1e6)
        # distribute children sequentially after self-time gap
        child_offset += self_time_us
        for child in children:
            walk(child, child_offset)
            child_offset += child["time"] * 1e6

    walk(data["root_frame"], start_time_s * 1e6)

    trace = {"traceEvents": events}
    with open(output_path, "w") as f:
        json.dump(trace, f)
    print(f"Written {len(events)} events to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.json> [output.json]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".json", "_chrome.json")
    convert(inp, out)
