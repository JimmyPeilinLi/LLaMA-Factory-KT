#!/usr/bin/env python3
"""
Slice a Chrome trace JSON by percent of events.

Examples:
  python scripts/slice_trace.py 80% -i sft_trace.json -o sft_trace_slice_p80.json
  python scripts/slice_trace.py -80 -i sft_trace.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def split_line(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def strip_trailing_comma(line: str) -> str:
    core, newline = split_line(line)
    rstripped = core.rstrip()
    if rstripped.endswith(","):
        rstripped = rstripped[:-1]
        return rstripped + newline
    return line


def ensure_trailing_comma(line: str) -> str:
    core, newline = split_line(line)
    rstripped = core.rstrip()
    if rstripped.endswith(","):
        return line
    return core + "," + newline


def parse_percent(text: str) -> float:
    value_text = text.strip()
    if value_text.endswith("%"):
        value_text = value_text[:-1]
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"invalid percent: {text!r}") from exc
    if value < -100 or value > 100:
        raise ValueError("percent must be between -100 and 100")
    return value


def count_events(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        in_events = False
        count = 0
        for line in handle:
            if not in_events:
                if '"traceEvents"' in line and "[" in line:
                    in_events = True
                continue
            stripped = line.lstrip()
            if stripped.startswith("]"):
                break
            if stripped.startswith("{"):
                count += 1
        if not in_events:
            raise ValueError("traceEvents array not found")
        return count


def compute_slice(total: int, percent: float) -> tuple[int, int, int]:
    fraction = abs(percent) / 100.0
    keep = int(total * fraction)
    if fraction > 0 and keep == 0:
        keep = 1
    if keep > total:
        keep = total
    if keep == 0:
        return keep, 0, -1
    if percent >= 0:
        return keep, 0, keep - 1
    return keep, total - keep, total - 1


def write_slice(input_path: Path, output_path: Path, start: int, end: int) -> None:
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        in_events = False
        idx = 0
        for line in src:
            if not in_events:
                dst.write(line)
                if '"traceEvents"' in line and "[" in line:
                    in_events = True
                continue
            stripped = line.lstrip()
            if stripped.startswith("]"):
                dst.write(line)
                for rest in src:
                    dst.write(rest)
                return
            if not stripped.startswith("{"):
                continue
            if idx < start or idx > end:
                idx += 1
                continue
            if idx == end:
                dst.write(strip_trailing_comma(line))
            else:
                dst.write(ensure_trailing_comma(line))
            idx += 1
        if not in_events:
            raise ValueError("traceEvents array not found")
        raise ValueError("traceEvents array not terminated with ]")


def default_output(input_path: Path, percent: float) -> Path:
    pct = abs(percent)
    if pct.is_integer():
        pct_text = str(int(pct))
    else:
        pct_text = str(pct).rstrip("0").rstrip(".")
        pct_text = pct_text.replace(".", "_")
    tag = f"m{pct_text}" if percent < 0 else f"p{pct_text}"
    return input_path.with_name(f"{input_path.stem}_slice_{tag}.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice a Chrome trace JSON by percent of traceEvents."
    )
    parser.add_argument(
        "percent",
        help="Percent of events to keep (e.g. 80, 80%%, -80, -80%%).",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="sft_trace.json",
        help="Input trace JSON path (default: sft_trace.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output trace JSON path (default: input name with suffix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts and the selected range.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"input not found: {input_path}")

    percent = parse_percent(args.percent)
    total = count_events(input_path)
    keep, start, end = compute_slice(total, percent)

    output_path = Path(args.output) if args.output else default_output(input_path, percent)
    if args.dry_run:
        print(
            f"total={total} keep={keep} range=[{start}, {end}] output={output_path}"
        )
        return

    write_slice(input_path, output_path, start, end)
    print(f"wrote {keep}/{total} events to {output_path}")


if __name__ == "__main__":
    main()
