import json
import numpy as np
from scipy.interpolate import interp1d

# ---------- data sources ----------
sources = {
    "ZeRO-LoRA": "/mnt/data/lpl/kernel_new_test_adapter/zero_lora_qwen3_235b_livebench_math/trainer_log.jsonl",
    "KT-FT": "/mnt/data/hxx/saves/qwen3-235-lora-math/trainer_log.jsonl",
}


def parse_elapsed_time(s: str) -> float:
    parts = s.split(":")
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 60 + m + sec / 60.0


def smooth(y, window: int = 15):
    alpha = 2 / (window + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def load_data(path: str):
    steps, losses, times = [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "loss" not in d:
                continue
            steps.append(d["current_steps"])
            losses.append(d["loss"])
            times.append(parse_elapsed_time(d["elapsed_time"]))
    return np.array(steps), np.array(losses), np.array(times)


# ---------- load ----------
data = {}
for name, path in sources.items():
    s, l, t = load_data(path)
    data[name] = {"steps": s, "loss": l, "time": t, "loss_smooth": smooth(l)}

zero = data["ZeRO-LoRA"]
ktft = data["KT-FT"]

# ============================================================
# Part 1: Every 25 steps — time & loss difference
# ============================================================
print("=" * 72)
print(f"{'Step':>6} | {'ZeRO Time':>10} | {'KT-FT Time':>10} | {'Time Diff':>10} | "
      f"{'Speedup':>7} | {'ZeRO Loss':>10} | {'KT-FT Loss':>10} | {'Loss Diff':>10}")
print("-" * 72)

step_points = list(range(25, int(min(zero["steps"][-1], ktft["steps"][-1])) + 1, 25))

# Interpolate smooth loss & time at exact step points
zero_time_fn = interp1d(zero["steps"], zero["time"], kind="linear")
ktft_time_fn = interp1d(ktft["steps"], ktft["time"], kind="linear")
zero_loss_fn = interp1d(zero["steps"], zero["loss_smooth"], kind="linear")
ktft_loss_fn = interp1d(ktft["steps"], ktft["loss_smooth"], kind="linear")

max_time_diff = 0
max_time_diff_step = 0

for step in step_points:
    zt = float(zero_time_fn(step))
    kt = float(ktft_time_fn(step))
    zl = float(zero_loss_fn(step))
    kl = float(ktft_loss_fn(step))
    td = zt - kt
    speedup = zt / kt if kt > 0 else float("inf")
    ld = zl - kl

    if td > max_time_diff:
        max_time_diff = td
        max_time_diff_step = step

    print(f"{step:>6} | {zt:>8.1f}m | {kt:>8.1f}m | {td:>+8.1f}m | "
          f"{speedup:>6.2f}x | {zl:>10.4f} | {kl:>10.4f} | {ld:>+10.4f}")

print("-" * 72)
print(f"Max time diff at same step: Step {max_time_diff_step}, "
      f"ZeRO is {max_time_diff:.1f} min slower")

# ============================================================
# Part 2: For the same (smoothed) loss, find max time difference
# ============================================================
print()
print("=" * 72)
print("Same loss -> time comparison (using smoothed curves)")
print("=" * 72)

# Build loss->time mapping (use smoothed loss, monotonically decreasing part)
# Find the point after which smooth loss is mostly decreasing
# Use all data and interpolate: for a given loss, what time does each method reach it?

# Make smooth loss monotonically decreasing for interpolation
def make_monotone_decreasing(loss, time):
    """Keep only points where loss is strictly decreasing (from the start)."""
    mono_loss = [loss[0]]
    mono_time = [time[0]]
    min_so_far = loss[0]
    for i in range(1, len(loss)):
        if loss[i] < min_so_far:
            min_so_far = loss[i]
            mono_loss.append(loss[i])
            mono_time.append(time[i])
    return np.array(mono_loss), np.array(mono_time)


zero_ml, zero_mt = make_monotone_decreasing(zero["loss_smooth"], zero["time"])
ktft_ml, ktft_mt = make_monotone_decreasing(ktft["loss_smooth"], ktft["time"])

# Interpolate: loss -> time (loss is decreasing, so flip for interp1d)
zero_loss2time = interp1d(zero_ml[::-1], zero_mt[::-1], kind="linear",
                          bounds_error=False, fill_value=np.nan)
ktft_loss2time = interp1d(ktft_ml[::-1], ktft_mt[::-1], kind="linear",
                          bounds_error=False, fill_value=np.nan)

# Common loss range
loss_min = max(zero_ml[-1], ktft_ml[-1])
loss_max = min(zero_ml[0], ktft_ml[0])
loss_probe = np.linspace(loss_max, loss_min, 200)

print(f"\n{'Loss':>10} | {'ZeRO Time':>10} | {'KT-FT Time':>10} | {'Time Diff':>10} | {'Speedup':>7}")
print("-" * 65)

best_diff = 0
best_loss = 0
best_zt = 0
best_kt = 0

results = []
for loss_val in loss_probe:
    zt = float(zero_loss2time(loss_val))
    kt = float(ktft_loss2time(loss_val))
    if np.isnan(zt) or np.isnan(kt):
        continue
    td = zt - kt
    results.append((loss_val, zt, kt, td))
    if td > best_diff:
        best_diff = td
        best_loss = loss_val
        best_zt = zt
        best_kt = kt

# Print sampled points (every ~10th)
step_size = max(1, len(results) // 15)
for i in range(0, len(results), step_size):
    loss_val, zt, kt, td = results[i]
    speedup = zt / kt if kt > 0 else float("inf")
    print(f"{loss_val:>10.4f} | {zt:>8.1f}m | {kt:>8.1f}m | {td:>+8.1f}m | {speedup:>6.2f}x")

print("-" * 65)
print(f"\nMax time gap at same loss:")
print(f"  Loss = {best_loss:.4f}")
print(f"  ZeRO-LoRA reaches it at {best_zt:.1f} min")
print(f"  KT-FT     reaches it at {best_kt:.1f} min")
print(f"  Difference: {best_diff:.1f} min  (ZeRO is {best_zt/best_kt:.2f}x slower)")
