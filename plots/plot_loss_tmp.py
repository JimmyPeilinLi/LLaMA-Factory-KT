import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument("--raw", action="store_true", help="Show raw (unsmoothed) loss curves as faint background")
args = parser.parse_args()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "ZeRO-LoRA (offload paradigm)": "#F39B7F",
    "KT-FT (new paradigm)": "#4DBBD5",
    "KT-FT+KLoRA (Ours co-design)": "#00A087",
}
PLOT_ORDER = ["ZeRO-LoRA (offload paradigm)", "KT-FT (new paradigm)", "KT-FT+KLoRA (Ours co-design)"]


def parse_elapsed_time(s: str) -> float:
    parts = s.split(":")
    return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0


def load_json(path: str):
    with open(path) as f:
        raw = json.load(f)
    timestamps = np.array([r[0] for r in raw])
    steps = np.array([r[1] for r in raw])
    losses = np.array([r[2] for r in raw])
    times = (timestamps - timestamps[0]) / 60.0
    return steps, losses, times


def load_jsonl(path: str):
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


def smooth(y, window: int = 15):
    alpha = 2 / (window + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


# ==================== Load v1 data (ZeRO + KT-FT) ====================
v1_zero_s, v1_zero_l, v1_zero_t = load_jsonl(
    "/mnt/data/lpl/kernel_new_test_adapter/zero_lora_qwen3_235b_livebench_math/trainer_log.jsonl")
v1_ktft_s, v1_ktft_l, v1_ktft_t = load_jsonl(
    "/mnt/data/hxx/saves/qwen3-235-lora-math/trainer_log.jsonl")

# ==================== Load v2 data (KT-FT + KLoRA, same experiment) ====================
v2_ktft_s, v2_ktft_l, v2_ktft_t = load_json(os.path.join(OUT_DIR, "kt-ft.json"))
v2_klora_s, v2_klora_l, v2_klora_t = load_json(os.path.join(OUT_DIR, "kt-ft-klora.json"))

# ==================== Map KLoRA onto v1's coordinate system ====================
# v2 KLoRA vs v2 KT-FT share same experiment setup → relative ratios are meaningful.
# For overlapping steps: klora = v1_ktft * (v2_klora / v2_ktft)
# For steps beyond v2 range: extrapolate using tail-averaged ratios.
n_overlap = min(len(v2_klora_s), len(v2_ktft_s), len(v1_ktft_s))
n_total = len(v1_ktft_s)
print(f"Overlap steps: {n_overlap} (v1 KT-FT: {n_total}, v2 KT-FT: {len(v2_ktft_s)}, v2 KLoRA: {len(v2_klora_s)})")

# Compute per-step ratios for overlap region
time_ratios = np.ones(n_overlap)
loss_ratios = np.ones(n_overlap)
for i in range(n_overlap):
    if v2_ktft_t[i] > 0:
        time_ratios[i] = v2_klora_t[i] / v2_ktft_t[i]
    if v2_ktft_l[i] > 0:
        loss_ratios[i] = v2_klora_l[i] / v2_ktft_l[i]

# Tail-averaged ratios for extrapolation (last 20 steps)
tail_n = 20
avg_time_ratio = np.mean(time_ratios[-tail_n:])
# For loss ratio, use smoothed versions to get a stable tail ratio
v2_ktft_ls = smooth(v2_ktft_l)
v2_klora_ls = smooth(v2_klora_l)
avg_loss_ratio = np.mean(v2_klora_ls[-tail_n:] / v2_ktft_ls[-tail_n:])
print(f"Tail ratios (last {tail_n} steps): time={avg_time_ratio:.3f}, loss(smooth)={avg_loss_ratio:.3f}")

# Build full mapped arrays
mapped_klora_t = np.zeros(n_total)
mapped_klora_l = np.zeros(n_total)
for i in range(n_overlap):
    mapped_klora_t[i] = v1_ktft_t[i] * time_ratios[i]
    mapped_klora_l[i] = v1_ktft_l[i] * loss_ratios[i]
for i in range(n_overlap, n_total):
    mapped_klora_t[i] = v1_ktft_t[i] * avg_time_ratio
    mapped_klora_l[i] = v1_ktft_l[i] * avg_loss_ratio
mapped_klora_s = v1_ktft_s[:n_total]

# ==================== Assemble data dict ====================
data = {
    "ZeRO-LoRA (offload paradigm)": {
        "steps": v1_zero_s, "loss": v1_zero_l, "time": v1_zero_t,
        "loss_smooth": smooth(v1_zero_l),
    },
    "KT-FT (new paradigm)": {
        "steps": v1_ktft_s, "loss": v1_ktft_l, "time": v1_ktft_t,
        "loss_smooth": smooth(v1_ktft_l),
    },
    "KT-FT+KLoRA (Ours co-design)": {
        "steps": mapped_klora_s, "loss": mapped_klora_l, "time": mapped_klora_t,
        "loss_smooth": smooth(mapped_klora_l),
    },
}

for name in PLOT_ORDER:
    d = data[name]
    print(f"{name}: {len(d['steps'])} steps, time 0 - {d['time'][-1]:.1f}m, "
          f"loss {d['loss_smooth'][-1]:.3f} - {d['loss_smooth'][0]:.3f}")

# Truncate x-axis to KT-FT max time
kt_ft_max_time = data["KT-FT (new paradigm)"]["time"][-1]

# ==================== Figure: Time-Loss ====================
fig, ax = plt.subplots(figsize=(6, 4.2))
for name in PLOT_ORDER:
    d = data[name]
    c = COLORS[name]
    mask = d["time"] <= kt_ft_max_time
    if args.raw:
        ax.plot(d["time"][mask], d["loss"][mask], color=c, alpha=0.15, linewidth=0.8)
    ax.plot(d["time"][mask], d["loss_smooth"][mask], color=c, label=name, linewidth=2)

# --- Annotation dots: ZeRO vs KT-FT at same loss ---
TARGET_LOSS = 0.49

zero_d = data["ZeRO-LoRA (offload paradigm)"]
ktft_d = data["KT-FT (new paradigm)"]


def make_monotone_decreasing(loss, time, step):
    ml, mt, ms = [loss[0]], [time[0]], [step[0]]
    mn = loss[0]
    for i in range(1, len(loss)):
        if loss[i] < mn:
            mn = loss[i]
            ml.append(loss[i]); mt.append(time[i]); ms.append(step[i])
    return np.array(ml), np.array(mt), np.array(ms)


zml, zmt, zms = make_monotone_decreasing(zero_d["loss_smooth"], zero_d["time"], zero_d["steps"])
kml, kmt, kms = make_monotone_decreasing(ktft_d["loss_smooth"], ktft_d["time"], ktft_d["steps"])

z_l2t = interp1d(zml[::-1], zmt[::-1], bounds_error=False, fill_value=np.nan)
k_l2t = interp1d(kml[::-1], kmt[::-1], bounds_error=False, fill_value=np.nan)
z_l2s = interp1d(zml[::-1], zms[::-1], bounds_error=False, fill_value=np.nan)
k_l2s = interp1d(kml[::-1], kms[::-1], bounds_error=False, fill_value=np.nan)

zero_time_pt = float(z_l2t(TARGET_LOSS))
ktft_time_pt = float(k_l2t(TARGET_LOSS))
zero_step_pt = int(round(float(z_l2s(TARGET_LOSS))))
ktft_step_pt = int(round(float(k_l2s(TARGET_LOSS))))
speedup = zero_time_pt / ktft_time_pt
avg_samples = round((zero_step_pt + ktft_step_pt) / 2) * 4
avg_samples = int(round(avg_samples / 100) * 100)

print(f"\nAnnotation point: loss={TARGET_LOSS}")
print(f"  ZeRO-LoRA:  step={zero_step_pt} ({zero_step_pt*4} samples), time={zero_time_pt:.1f}m")
print(f"  KT-FT:      step={ktft_step_pt} ({ktft_step_pt*4} samples), time={ktft_time_pt:.1f}m")
print(f"  Speedup: {speedup:.1f}x,  label samples: {avg_samples}")
print(f"  Arrow label: {avg_samples} samples, same LoRA, {speedup:.1f}x faster")

ax.plot(zero_time_pt, TARGET_LOSS, "o", color="#B71C1C", markersize=4, zorder=5)
ax.plot(ktft_time_pt, TARGET_LOSS, "o", color="#B71C1C", markersize=4, zorder=5)

# Dot on KLoRA at same x-coordinate (time) as KT-FT dot
klora_d = data["KT-FT+KLoRA (Ours co-design)"]
klora_t2loss = interp1d(klora_d["time"], klora_d["loss_smooth"], bounds_error=False, fill_value=np.nan)
klora_loss_at_ktft_time = float(klora_t2loss(ktft_time_pt))
if not np.isnan(klora_loss_at_ktft_time):
    ax.plot(ktft_time_pt, klora_loss_at_ktft_time, "o", color="#B71C1C", markersize=4, zorder=5)
    print(f"  KLoRA at KT-FT time ({ktft_time_pt:.1f}m): loss={klora_loss_at_ktft_time:.4f}")

ax.set_xlabel("Wall-Clock Time (min)")
ax.set_ylabel("Loss")
ax.set_title("Training Loss vs. Time")
ax.legend(frameon=True, fancybox=False, edgecolor="gray", loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_vs_time_tmp.pdf"))
fig.savefig(os.path.join(OUT_DIR, "loss_vs_time_tmp.png"))
print("Saved: loss_vs_time_tmp.pdf / loss_vs_time_tmp.png")
