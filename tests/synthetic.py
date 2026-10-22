"""Small synthetic dataset in the CirCor DigiScope format, used by the end-to-end test (tests/test_pipeline.py).

    python tests/synthetic.py --out /tmp/synthetic --n-patients 150     # to inspect it by hand

The signals are crude imitations of phonocardiograms: S1/S2 bursts at a random heart
rate; recordings where a murmur is "heard" get band-limited noise during systole, and
"Unknown" patients get strong noise. Results on these data say nothing about the real
task: the only purpose is to check that the whole pipeline runs end to end.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

FS = 4000
LOCATIONS = ["AV", "PV", "TV", "MV"]
COLUMNS = ["Patient ID", "Recording locations:", "Age", "Sex", "Height", "Weight", "Pregnancy status", "Murmur",
           "Murmur locations", "Most audible location", "Systolic murmur timing", "Systolic murmur shape",
           "Systolic murmur grading", "Systolic murmur pitch", "Systolic murmur quality", "Diastolic murmur timing",
           "Diastolic murmur shape", "Diastolic murmur grading", "Diastolic murmur pitch", "Diastolic murmur quality",
           "Outcome", "Campaign", "Additional ID"]


def burst(duration, freq, rng):
    t = np.arange(int(duration * FS)) / FS
    return np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / (duration / 4)) * (1 - np.exp(-t / 0.005))


def add(x, start, y):
    end = min(len(x), start + len(y))
    x[start:end] += y[:end - start]


def synth_recording(duration, murmur, noisy, rng):
    n = int(duration * FS)
    x = np.zeros(n)
    period = 60 / rng.uniform(70, 150)
    systole = 0.35 * period
    murmur_filter = butter(4, [150, 450], "bandpass", fs=FS, output="sos")
    murmur_amp = rng.uniform(0.15, 0.5)
    segments, t = [], rng.uniform(0, period)
    while t + period < duration:
        add(x, int(t * FS), burst(0.10, rng.uniform(40, 70), rng))
        add(x, int((t + systole) * FS), 0.7 * burst(0.08, rng.uniform(60, 100), rng))
        if murmur:
            a, b = int((t + 0.1) * FS), int((t + systole) * FS)
            add(x, a, murmur_amp * sosfilt(murmur_filter, rng.normal(size=b - a)) * np.hanning(b - a) * 3)
        segments += [(t, t + 0.1, 1), (t + 0.1, t + systole, 2), (t + systole, t + systole + 0.08, 3),
                     (t + systole + 0.08, t + period, 4)]
        t += period * rng.uniform(0.95, 1.05)
    x += rng.normal(scale=0.02, size=n)
    if noisy:
        x += rng.normal(scale=rng.uniform(0.3, 0.8), size=n)
        x += rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * rng.uniform(1, 5) * np.arange(n) / FS)
    x = 0.8 * x / np.abs(x).max()
    return (x * 32767).astype(np.int16), segments


def make_dataset(out, n_patients=150, seed=0):
    """Write `n_patients` synthetic patients (training_data.csv + training_data/) to `out`; returns `out`."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    rec_dir = out / "training_data"
    rec_dir.mkdir(parents=True, exist_ok=True)

    ids = rng.choice(np.arange(10000, 90000), n_patients, replace=False)
    labels = rng.choice(["Absent", "Present", "Unknown"], n_patients, p=[0.74, 0.19, 0.07])
    additional = {}
    for a, b in rng.choice(n_patients, (n_patients // 20, 2), replace=False):  # ~10% seen twice
        additional[ids[a]], additional[ids[b]] = ids[b], ids[a]
        labels[b] = labels[a]

    rows = []
    for pid, label in zip(ids, labels):
        locations = LOCATIONS if rng.random() < 0.7 else sorted(rng.choice(LOCATIONS, rng.integers(1, 4), replace=False))
        locations = list(locations) + ([locations[0]] if rng.random() < 0.05 else [])  # sometimes 2 recordings at a location
        murmur_locs = sorted(set(rng.choice(locations, rng.integers(1, len(locations) + 1), replace=False))) \
            if label == "Present" else []
        names, counts = [], {loc: locations.count(loc) for loc in locations}
        seen = {}
        for loc in locations:
            seen[loc] = seen.get(loc, 0) + 1
            name = f"{pid}_{loc}" + (f"_{seen[loc]}" if counts[loc] > 1 else "")
            signal, segments = synth_recording(rng.uniform(5, 30), loc in murmur_locs, label == "Unknown", rng)
            wavfile.write(rec_dir / f"{name}.wav", FS, signal)
            (rec_dir / f"{name}.hea").write_text(f"{name} 1 {FS} {len(signal)}\n{name}.wav 16+44 1 16 0 0 0 0 {loc}\n")
            (rec_dir / f"{name}.tsv").write_text("".join(f"{a:.6f}\t{b:.6f}\t{c}\n" for a, b, c in segments))
            names.append((loc, name))
        row = dict.fromkeys(COLUMNS, np.nan)
        row.update({"Patient ID": pid, "Recording locations:": "+".join(locations), "Age": rng.choice(["Child", "Infant"]),
                    "Sex": rng.choice(["Female", "Male"]), "Pregnancy status": False, "Murmur": label,
                    "Murmur locations": "+".join(murmur_locs) or np.nan,
                    "Outcome": rng.choice(["Normal", "Abnormal"]), "Campaign": rng.choice(["CC2014", "CC2015"]),
                    "Additional ID": additional.get(pid, np.nan)})
        rows.append(row)
        header = [f"{pid} {len(names)} {FS}"] + [f"{loc} {n}.hea {n}.wav {n}.tsv" for loc, n in names]
        fields = [f"#{c}: {row[c]}" for c in COLUMNS[2:]]
        (rec_dir / f"{pid}.txt").write_text("\n".join(header + fields) + "\n")

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["Additional ID"] = df["Additional ID"].astype("Int64")
    df.to_csv(out / "training_data.csv", index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-patients", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"synthetic dataset written to {make_dataset(args.out, args.n_patients, args.seed)}")
