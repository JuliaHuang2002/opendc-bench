"""FFT power at 24h harmonic, broken down by prediction sub-window."""
import numpy as np

MODELS = ["t5-small", "t5-large", "moirai-large"]

def power_24h(x):
    # x: (N, L). FFT bin for 24h = round(L/144)... but need L=144 for direct read.
    # Use fixed strategy: pad/window fixed 144 chunks.
    fft = np.fft.rfft(x, axis=1)
    return (np.abs(fft) ** 2)[:, 1].mean() if x.shape[1] == 144 else None

# Split each 144-step prediction into halves: first 12h vs last 12h
# to see if periodicity injection increases over horizon
print(f"{'model':<15} {'first_12h':>12} {'last_12h':>12} {'ratio(last/first)':>20}")
print("-" * 70)

for m in MODELS:
    d = np.load(f"preds/{m}.npz")
    p = d["preds"]  # (N, 144)
    # detrend each half so 24h power isn't dominated by DC
    first = p[:, :72] - p[:, :72].mean(axis=1, keepdims=True)
    last = p[:, 72:] - p[:, 72:].mean(axis=1, keepdims=True)
    # 12h half → 24h harmonic isn't representable; use dominant harmonic in half = 12h cycle within half
    # bin 1 in 72-length rfft = 1 cycle per 12h = 12h period
    def pk(x):
        return (np.abs(np.fft.rfft(x, axis=1)) ** 2)[:, 1].mean()
    a, b = pk(first), pk(last)
    print(f"{m:<15} {a:>12.2f} {b:>12.2f} {b/a:>20.2f}")

# reference: same for truth
d = np.load("preds/t5-small.npz")
t = d["trues"]
first = t[:, :72] - t[:, :72].mean(axis=1, keepdims=True)
last = t[:, 72:] - t[:, 72:].mean(axis=1, keepdims=True)
def pk(x):
    return (np.abs(np.fft.rfft(x, axis=1)) ** 2)[:, 1].mean()
a, b = pk(first), pk(last)
print(f"{'truth':<15} {a:>12.2f} {b:>12.2f} {b/a:>20.2f}")
