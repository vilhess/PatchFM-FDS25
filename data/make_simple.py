"""Generate data/simple.npy: easy, kid-friendly signals.

Families are taken from SyntheticTimeSeriesDataset, keeping only the clearly
predictable ones (no random walks, AR noise, chirps or random events). Periods
are chosen so that the window shown in draw.py (last ~96 points) contains one
to three cycles, which keeps the shape readable for children.

Usage: python data/make_simple.py
"""
import os
import random

import numpy as np

SEQ_LEN = 544          # 512 context + 32 forecast, same as before
N_TOTAL = 1000
SEED = 42
KID_PERIODS = [32, 40, 48, 64, 80, 96]

random.seed(SEED)
rng = np.random.default_rng(SEED)
t = np.arange(SEQ_LEN, dtype=float)
u = np.linspace(0, 1, SEQ_LEN)


def period():
    return random.choice(KID_PERIODS)


def phase():
    return random.uniform(0, 2 * np.pi)


def linear():
    return random.choice([-1, 1]) * random.uniform(0.1, 5) * t


def sine():
    return random.uniform(1, 100) * np.sin(2 * np.pi * t / period() + phase())


def linear_sin():
    p = period()
    amp = random.uniform(5, 50)
    # trend kept gentle so the wave stays visible
    slope = random.uniform(-1, 1) * amp / p
    return amp * np.sin(2 * np.pi * t / p + phase()) + slope * t


def staircase():
    p = random.choice([16, 24, 32])
    h = random.uniform(1, 30) * random.choice([-1, 1])
    return h * np.floor((t + random.uniform(0, p)) / p)


def square():
    p = period()
    duty = random.uniform(0.3, 0.7)
    return np.where(np.remainder(t + random.uniform(0, p), p) < duty * p, 1.0, -1.0)


def triangle():
    p = period()
    return np.abs(np.remainder(t + random.uniform(0, p), p) - p / 2)


def sawtooth():
    p = period()
    return random.choice([-1, 1]) * np.remainder(t + random.uniform(0, p), p)


def rectified_sine():
    wave = np.sin(2 * np.pi * t / period() + phase())
    return np.abs(wave) if random.random() < 0.5 else np.clip(wave, 0, None)


def clipped_sine():
    clip = random.uniform(0.3, 0.8)
    return np.clip(np.sin(2 * np.pi * t / period() + phase()), -clip, clip)


def gaussian_pulses():
    p = random.choice([32, 48, 64])
    width = random.uniform(p / 12, p / 5)
    d = np.remainder(t - random.uniform(0, p), p)
    d = np.minimum(d, p - d)
    return random.choice([-1, 1]) * np.exp(-(d ** 2) / (2 * width ** 2))


def trend_seasonal():
    p = period()
    amp = random.uniform(5, 30)
    slope = random.uniform(-1, 1) * amp / p
    return slope * t + amp * np.sin(2 * np.pi * t / p + phase())


def harmonic_sum():
    p = random.choice([48, 64, 96])
    out = np.zeros(SEQ_LEN)
    for h in range(1, random.randint(2, 3) + 1):
        out += random.uniform(1, 20) / h * np.sin(2 * np.pi * h * t / p + phase())
    return out


def power_trend():
    return random.choice([-1, 1]) * u ** random.choice([0.5, 1.5, 2.0, 2.5])


def exp_relaxation():
    start, end = random.uniform(-100, 100), random.uniform(-100, 100)
    return end + (start - end) * np.exp(-random.uniform(2, 4) * u)


FAMILIES = [
    linear, sine, linear_sin, staircase, square, triangle, sawtooth,
    rectified_sine, clipped_sine, gaussian_pulses, trend_seasonal,
    harmonic_sum, power_trend, exp_relaxation,
]


def make_one(i):
    y = FAMILIES[i % len(FAMILIES)]()
    # light noise only (0-3% of the spread of the visible window) so the shape stays obvious
    y = y + rng.standard_normal(SEQ_LEN) * y[-96:].std() * random.uniform(0, 0.03)
    return (y - y.mean()) / (y.std() + 1e-6)


if __name__ == "__main__":
    data = np.stack([make_one(i) for i in range(N_TOTAL)]).astype(np.float32)
    rng.shuffle(data)
    out = os.path.join(os.path.dirname(__file__), "simple.npy")
    np.save(out, data)
    print(f"Saved {data.shape} to {out}")
