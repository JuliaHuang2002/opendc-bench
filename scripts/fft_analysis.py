"""FFT power spectrum: verify large models force 24h periodicity prior."""
import numpy as np

MODELS = ["t5-small", "t5-large", "moirai-large"]
HARMONICS = {
    "24h": 1, "12h": 2, "8h": 3, "6h": 4, "4h": 6, "3h": 8,
}

def power_at_freq(x, cycles_per_day):
    fft = np.fft.rfft(x, axis=1)
    power = (np.abs(fft) ** 2)
    return power[:, cycles_per_day].mean()

# absolute power
print("=" * 90)
print("ABSOLUTE POWER (higher = model puts more energy at this frequency)")
print("=" * 90)
print(f"{'model':<15} " + " ".join(f"{k:>10}" for k in HARMONICS.keys()))
print("-" * 90)

trues = None
model_powers = {}
for m in MODELS:
    d = np.load(f"preds/{m}.npz")
    preds = d["preds"]
    if trues is None:
        trues = d["trues"]
    powers = np.array([power_at_freq(preds, c) for c in HARMONICS.values()])
    model_powers[m] = powers
    print(f"{m:<15} " + " ".join(f"{p:>10.2f}" for p in powers))

truth_p = np.array([power_at_freq(trues, c) for c in HARMONICS.values()])
print(f"{'truth':<15} " + " ".join(f"{p:>10.2f}" for p in truth_p))

# normalized
print()
print("=" * 90)
print("RATIO vs TRUTH (>1.0 = over-emphasized; <1.0 = under-emphasized)")
print("=" * 90)
print(f"{'model':<15} " + " ".join(f"{k:>10}" for k in HARMONICS.keys()))
print("-" * 90)
for m in MODELS:
    ratio = model_powers[m] / truth_p
    print(f"{m:<15} " + " ".join(f"{r:>10.2f}" for r in ratio))
