import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

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

# ---------- data sources ----------
sources = {
    "ZeRO-LoRA": "/mnt/data/lpl/kernel_new_test_adapter/zero_lora_qwen3_235b_livebench_math/trainer_log.jsonl",
    "KT-FT (new paradigm)": "/mnt/data/hxx/saves/qwen3-235-lora-math/trainer_log.jsonl",
    "KT-FT+KLoRA (Ours co-design)": "/mnt/data/hxx/saves/qwen3-235-lora-math-le/trainer_log.jsonl",
}

# Academic color palette (color-blind friendly, Nature-style)
COLORS = ["#F39B7F", "#4DBBD5", "#00A087"]


def parse_elapsed_time(s: str) -> float:
    """Convert 'H:MM:SS' to minutes."""
    parts = s.split(":")
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 60 + m + sec / 60.0


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


def smooth(y, window: int = 15):
    """EMA-style smoothing (tensorboard-like)."""
    alpha = 2 / (window + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


# ---------- load ----------
data = {}
for name, path in sources.items():
    s, l, t = load_data(path)
    data[name] = {"steps": s, "loss": l, "time": t, "loss_smooth": smooth(l)}

# Get KT-FT max time for truncation
kt_ft_max_time = data["KT-FT (new paradigm)"]["time"][-1]


def plot_single(ax, x_key, xlabel, title):
    for idx, (name, d) in enumerate(data.items()):
        c = COLORS[idx]
        ax.plot(d[x_key], d["loss"], color=c, alpha=0.15, linewidth=0.8)
        ax.plot(d[x_key], d["loss_smooth"], color=c, label=name, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=False, edgecolor="gray", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- Figure 1: Step-Loss ----------
fig1, ax1 = plt.subplots(figsize=(6, 4.2))
plot_single(ax1, "steps", "Training Step", "Training Loss vs. Step")
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "loss_vs_step.pdf"))
fig1.savefig(os.path.join(OUT_DIR, "loss_vs_step.png"))
print("Saved: loss_vs_step.pdf / loss_vs_step.png")

# ---------- Figure 2: Time-Loss (truncated to KT-FT max time) ----------
fig2, ax2 = plt.subplots(figsize=(6, 4.2))
for idx, (name, d) in enumerate(data.items()):
    c = COLORS[idx]
    mask = d["time"] <= kt_ft_max_time
    ax2.plot(d["time"][mask], d["loss"][mask], color=c, alpha=0.15, linewidth=0.8)
    ax2.plot(d["time"][mask], d["loss_smooth"][mask], color=c, label=name, linewidth=2)

# --- Annotation: arrow between ZeRO and KT-FT at same loss ---
TARGET_LOSS = 0.49

zero_d = data["ZeRO-LoRA"]
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
# Round to nearest 100
avg_samples = int(round(avg_samples / 100) * 100)

print(f"Annotation point: loss={TARGET_LOSS}")
print(f"  ZeRO-LoRA:  step={zero_step_pt} ({zero_step_pt*4} samples), time={zero_time_pt:.1f}m")
print(f"  KT-FT:      step={ktft_step_pt} ({ktft_step_pt*4} samples), time={ktft_time_pt:.1f}m")
print(f"  Speedup: {speedup:.1f}x,  label samples: {avg_samples}")

# Draw two marker dots
ax2.plot(zero_time_pt, TARGET_LOSS, "o", color="#B71C1C", markersize=4, zorder=5)
ax2.plot(ktft_time_pt, TARGET_LOSS, "o", color="#B71C1C", markersize=4, zorder=5)

# Draw arrow from ZeRO point to KT-FT point
ax2.annotate(
    "",
    xy=(ktft_time_pt, TARGET_LOSS),
    xytext=(zero_time_pt, TARGET_LOSS),
    arrowprops=dict(
        arrowstyle="->,head_width=0.3,head_length=0.15",
        color="#B71C1C",
        lw=2.0,
        connectionstyle="arc3,rad=0",
    ),
    zorder=6,
)

# Label above the arrow
mid_time = (zero_time_pt + ktft_time_pt) / 2
ax2.text(
    mid_time, TARGET_LOSS + 0.12,
    f"{avg_samples} samples, same LoRA, {speedup:.1f}x faster",
    ha="center", va="bottom",
    fontsize=8, fontstyle="italic", color="#B71C1C",
    fontweight="bold",
)

ax2.set_xlabel("Wall-Clock Time (min)")
ax2.set_ylabel("Loss")
ax2.set_title("Training Loss vs. Time")
ax2.legend(frameon=True, fancybox=False, edgecolor="gray", loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "loss_vs_time.pdf"))
fig2.savefig(os.path.join(OUT_DIR, "loss_vs_time.png"))
print("Saved: loss_vs_time.pdf / loss_vs_time.png")
