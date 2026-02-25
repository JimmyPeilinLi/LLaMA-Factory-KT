#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


TS_FMT = "%Y-%m-%d %H:%M:%S"


def parse_hms(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def find_latest_checkpoint_json(train_out: Path, name: str):
    cands = sorted(train_out.glob(f"checkpoint-*/{name}"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for p in cands:
        if p.exists():
            return read_json(p)
    return None


def parse_mem_monitor(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\d+)", line.strip())
        if not m:
            continue
        rows.append((datetime.strptime(m.group(1), TS_FMT), int(m.group(2))))
    return rows


def parse_mpstat(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        # Example:
        # 05:47:53     all    1.02 ... 96.59
        if not re.match(r"^\d{2}:\d{2}:\d{2}\s+all\s+", line):
            continue
        parts = line.split()
        if len(parts) < 12:
            continue
        rows.append(
            {
                "time_hms": parts[0],
                "cpu": parts[1],
                "usr": maybe_float(parts[2]),
                "sys": maybe_float(parts[4]),
                "iowait": maybe_float(parts[5]),
                "idle": maybe_float(parts[-1]),
                "busy": (100.0 - float(parts[-1])) if maybe_float(parts[-1]) is not None else None,
            }
        )
    return rows


def parse_dmon(path: Path):
    rows = []
    if not path.exists():
        return rows
    header = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# gpu"):
            # "# gpu  rxpci  txpci  sm ..."
            header = re.split(r"\s+", line.lstrip("# ").strip())
            continue
        if line.startswith("# Idx"):
            # Units row, ignore.
            continue
        if line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        if header is None:
            continue
        # dmon rows are prefixed by gpu idx and then values matching header[1:].
        row = {"Idx": int(parts[0])}
        cols = header[1:]
        vals = parts[1 : 1 + len(cols)]
        for c, v in zip(cols, vals):
            if v == "-":
                row[c] = None
            else:
                row[c] = maybe_float(v)
        rows.append(row)
    return rows


def parse_train_log(path: Path):
    out = {
        "timestamps": {},
        "param_counts": {},
        "diagnostics": {},
        "errors": [],
        "diag_mem_samples": [],
    }
    if not path.exists():
        return out

    ts_pat = re.compile(r"\[(?:INFO|WARNING|ERROR)\|(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
    for line in path.read_text(errors="ignore").splitlines():
        if "Loading dataset " in line and "timestamps" in out:
            m = ts_pat.search(line)
            if m and "dataset_load" not in out["timestamps"]:
                out["timestamps"]["dataset_load"] = m.group(1)
        if "===== Shard-by-shard loading START =====" in line:
            m = ts_pat.search(line)
            if m:
                out["timestamps"]["shard_loading_start"] = m.group(1)
        if "===== Shard-by-shard loading DONE =====" in line:
            m = ts_pat.search(line)
            if m:
                out["timestamps"]["shard_loading_done"] = m.group(1)
        if "trainable params:" in line.lower():
            # HF/PEFT style pretty print
            out["diagnostics"].setdefault("trainable_lines", []).append(line.strip())
        if "num_trainable_params" in line.lower():
            out["diagnostics"].setdefault("trainable_lines", []).append(line.strip())
        if "num_params =" in line and "num_elems =" in line:
            m = re.search(r"num_params\s*=\s*(\d+),\s*num_elems\s*=\s*([0-9.A-Za-z]+)", line)
            if m:
                out["param_counts"]["num_param_objects"] = int(m.group(1))
                out["param_counts"]["num_elems_text"] = m.group(2)
        if "'train_steps_per_second':" in line or '"train_steps_per_second"' in line:
            out["diagnostics"].setdefault("speed_lines", []).append(line.strip())
        if "Training completed" in line:
            m = ts_pat.search(line)
            if m:
                out["timestamps"]["training_completed"] = m.group(1)
        if "Traceback (most recent call last)" in line or "RuntimeError:" in line or "ERROR" in line:
            if "pkg_resources is deprecated" not in line:
                out["errors"].append(line.strip())
        if "RSS=" in line and "Avail=" in line and "Total=" in line:
            m_ts = ts_pat.search(line)
            m_mem = re.search(r"RSS=([0-9.]+)GB,\s*Avail=([0-9.]+)GB,\s*Total=([0-9.]+)GB", line)
            if m_ts and m_mem:
                rss = float(m_mem.group(1))
                avail = float(m_mem.group(2))
                total = float(m_mem.group(3))
                out["diag_mem_samples"].append(
                    {
                        "ts": m_ts.group(1),
                        "rss_gb": rss,
                        "avail_gb": avail,
                        "total_gb": total,
                        "used_effective_gb": total - avail,
                        "line": line.strip(),
                    }
                )
    return out


def parse_steady_start(path: Path):
    if not path.exists():
        return None
    txt = path.read_text().strip().splitlines()
    for line in txt:
        line = line.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", line):
            return line
    return None


def summarize_numeric(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    def pct(p):
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return vals_sorted[idx]
    return {
        "count": n,
        "min": min(vals_sorted),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "max": max(vals_sorted),
        "mean": sum(vals_sorted) / n,
    }


def parse_trainer_log_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def extract_step_times_from_trainer_log(rows):
    # Uses elapsed_time deltas between consecutive current_steps entries.
    step_rows = [r for r in rows if "current_steps" in r and "elapsed_time" in r]
    step_rows = sorted(step_rows, key=lambda r: r["current_steps"])
    out = []
    prev_step = None
    prev_elapsed = None
    for r in step_rows:
        step = int(r["current_steps"])
        elapsed = parse_hms(str(r["elapsed_time"]))
        if prev_step is not None and step > prev_step and prev_elapsed is not None:
            dt = elapsed - prev_elapsed
            ds = step - prev_step
            if ds > 0:
                out.extend([dt / ds] * ds)
        prev_step = step
        prev_elapsed = elapsed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--train-output", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    raw = args.raw_dir
    train_out = args.train_output

    trainer_log_rows = parse_trainer_log_jsonl(train_out / "trainer_log.jsonl")
    trainer_state = read_json(train_out / "trainer_state.json") or find_latest_checkpoint_json(train_out, "trainer_state.json")
    all_results = read_json(train_out / "all_results.json") or find_latest_checkpoint_json(train_out, "all_results.json")
    train_log = parse_train_log(raw / "train.log")
    mem_rows = parse_mem_monitor(raw / "mem_monitor.log")
    mp_rows = parse_mpstat(raw / "mpstat.log")
    dmon_rows = parse_dmon(raw / "nvidia_dmon.log")
    dmon_steady_rows = parse_dmon(raw / "nvidia_dmon_steady.log")
    steady_start_ts = parse_steady_start(raw / "steady_sampling_start.txt")
    mp_steady_rows = parse_mpstat(raw / "mpstat_steady.log")

    step_times = extract_step_times_from_trainer_log(trainer_log_rows)
    warmup_cut = 5
    steady_step_times = step_times[warmup_cut:] if len(step_times) > warmup_cut else step_times

    tokens_seen = None
    global_step = None
    if trainer_state:
        tokens_seen = trainer_state.get("num_input_tokens_seen")
        global_step = trainer_state.get("global_step")
    avg_tokens_per_step = (tokens_seen / global_step) if (tokens_seen and global_step) else None

    gpu_groups = defaultdict(list)
    for r in dmon_rows:
        gpu_groups[int(r["Idx"])].append(r)

    dmon_summary = {}
    for gid, rows in sorted(gpu_groups.items()):
        dmon_summary[str(gid)] = {
            "rxpci_MBps": summarize_numeric([r.get("rxpci") for r in rows]),
            "txpci_MBps": summarize_numeric([r.get("txpci") for r in rows]),
            "sm_pct": summarize_numeric([r.get("sm") for r in rows]),
            "mem_pct": summarize_numeric([r.get("mem") for r in rows]),
            "fb_MB": summarize_numeric([r.get("fb") for r in rows]),
            "pwr_W": summarize_numeric([r.get("pwr") for r in rows]),
            "gtemp_C": summarize_numeric([r.get("gtemp") for r in rows]),
        }

    dmon_steady_summary = {}
    gpu_groups_steady = defaultdict(list)
    for r in dmon_steady_rows:
        gpu_groups_steady[int(r["Idx"])].append(r)
    for gid, rows in sorted(gpu_groups_steady.items()):
        dmon_steady_summary[str(gid)] = {
            "rxpci_MBps": summarize_numeric([r.get("rxpci") for r in rows]),
            "txpci_MBps": summarize_numeric([r.get("txpci") for r in rows]),
            "sm_pct": summarize_numeric([r.get("sm") for r in rows]),
            "mem_pct": summarize_numeric([r.get("mem") for r in rows]),
            "fb_MB": summarize_numeric([r.get("fb") for r in rows]),
            "pwr_W": summarize_numeric([r.get("pwr") for r in rows]),
            "gtemp_C": summarize_numeric([r.get("gtemp") for r in rows]),
        }

    mem_vals = [m for _, m in mem_rows]
    mem_summary = summarize_numeric(mem_vals)
    if mem_rows:
        mem_summary["first_GB"] = mem_rows[0][1]
        mem_summary["last_GB"] = mem_rows[-1][1]

    cpu_busy_summary = summarize_numeric([r["busy"] for r in mp_rows])
    cpu_busy_steady_summary = summarize_numeric([r["busy"] for r in mp_steady_rows])
    diag_mem_samples = train_log.get("diag_mem_samples", [])
    diag_used_summary = summarize_numeric([s["used_effective_gb"] for s in diag_mem_samples])
    diag_rss_summary = summarize_numeric([s["rss_gb"] for s in diag_mem_samples])

    result = {
        "paths": {
            "raw_dir": str(raw),
            "train_output": str(train_out),
        },
        "training": {
            "all_results": all_results,
            "trainer_state": {
                "global_step": global_step,
                "max_steps": trainer_state.get("max_steps") if trainer_state else None,
                "num_input_tokens_seen": tokens_seen,
                "num_train_epochs": trainer_state.get("num_train_epochs") if trainer_state else None,
            },
            "avg_tokens_per_step": avg_tokens_per_step,
            "step_time_from_trainer_log_jsonl": {
                "count": len(step_times),
                "warmup_cut": warmup_cut,
                "all": summarize_numeric(step_times),
                "steady": summarize_numeric(steady_step_times),
            },
            "trainer_log_jsonl_rows": len(trainer_log_rows),
            "train_log": train_log,
        },
        "system": {
            "mem_monitor_GB": mem_summary,
            "mpstat_busy_pct": cpu_busy_summary,
            "mpstat_busy_pct_steady": cpu_busy_steady_summary,
            "diag_mem_used_effective_gb": diag_used_summary,
            "diag_process_rss_gb": diag_rss_summary,
            "nvidia_dmon": dmon_summary,
            "nvidia_dmon_steady": dmon_steady_summary,
            "steady_sampling_start": steady_start_ts,
            "dmon_row_count": len(dmon_rows),
            "dmon_steady_row_count": len(dmon_steady_rows),
            "mpstat_row_count": len(mp_rows),
            "mpstat_steady_row_count": len(mp_steady_rows),
            "mem_row_count": len(mem_rows),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
