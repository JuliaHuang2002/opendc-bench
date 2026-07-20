"""Two plots: (1) time-domain forecast vs truth (2) FFT power spectrum."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["t5-small", "t5-large", "moirai-large"]
COLORS = {"t5-small": "#2ca02c", "t5-large": "#ff7f0e", "moirai-large": "#d62728"}
LABELS = {"t5-small": "Chronos-T5-small (46M)",
          "t5-large": "Chronos-T5-large (710M)",
          "moirai-large": "Moirai-1.1-R-large (311M)"}

data = {m: np.load(f"preds/{m}.npz") for m in MODELS}
trues = data["t5-small"]["trues"]  # (300, 144)

# pick a representative window (median MAE across models to be fair)
maes = np.mean([np.abs(data[m]["preds"] - trues).mean(axis=1) for m in MODELS], axis=0)
idx = int(np.argsort(maes)[len(maes) // 2])
print(f"[plot] using window idx={idx}, median-MAE across models")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# --- (1) time domain ---
ax = axes[0]
x = np.arange(144) / 6.0  # hours (10-min steps, 6 per hour)
ax.plot(x, trues[idx], "k-", lw=2, label="Ground Truth", alpha=0.85)
for m in MODELS:
    ax.plot(x, data[m]["preds"][idx], color=COLORS[m], lw=1.4, label=LABELS[m], alpha=0.9)
ax.set_xlabel("Forecast horizon (hours)")
ax.set_ylabel("GPU utilization (%)")
ax.set_title("(a) 24h-ahead forecast — representative window")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# --- (2) FFT spectrum ---
ax = axes[1]
freqs = np.fft.rfftfreq(144, d=1/144)  # cycles per day
def spec(x):
    return (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0)

ax.plot(freqs, spec(trues), "k-", lw=2, label="Ground Truth", alpha=0.85)
for m in MODELS:
    ax.plot(freqs, spec(data[m]["preds"]), color=COLORS[m], lw=1.4, label=LABELS[m], alpha=0.9)

# mark key harmonics
for c, name in [(1, "24h"), (2, "12h"), (4, "6h"), (8, "3h")]:
    ax.axvline(c, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(c, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1, name,
            rotation=90, fontsize=7, va="top", color="gray")

ax.set_xlabel("Frequency (cycles/day)")
ax.set_ylabel("Power")
ax.set_yscale("log")
ax.set_xlim(0, 20)
ax.set_title("(b) Prediction power spectrum")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = "preds/mechanism_fig.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[plot] saved {out}")
