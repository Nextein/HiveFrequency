import torch
import torchaudio
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ----------------------
# Utility functions
# ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def minmax_scale(arr):
    a = np.array(arr, dtype=float)
    if len(a) == 0:
        return a
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return ((a - lo) / (hi - lo)).astype(float)

def load_presets(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def jitter_params(params, jitter_fraction=0.05):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            jitter = np.random.uniform(-1, 1) * jitter_fraction * v
            new_params[k] = float(v + jitter)
        else:
            new_params[k] = v
    return new_params

# ----------------------
# Synthesis functions
# ----------------------
def adsr_envelope(y, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    N = y.shape[0]
    env = torch.ones(N)
    A = int(sr*attack)
    D = int(sr*decay)
    R = int(sr*release)
    S = N - (A+D+R)
    if A > 0:
        env[:A] = torch.linspace(0, 1, A)
    if D > 0:
        env[A:A+D] = torch.linspace(1, sustain, D)
    if S > 0:
        env[A+D:A+D+S] = sustain
    if R > 0:
        env[-R:] = torch.linspace(env[A+D+S], 0, R)
    return y * env

def white_noise(duration, sr=44100):
    samples = int(duration * sr)
    return torch.rand(samples) * 2 - 1

def sine_wave(freq, duration, sr=44100):
    t = torch.arange(0, duration, 1/sr)
    return torch.sin(2 * np.pi * freq * t)

# Simple filters using torchaudio functional
def lowpass(y, cutoff, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)  # [1,1,N]
    y = torchaudio.functional.lowpass_biquad(y, sr, cutoff)
    return y.squeeze()

def highpass(y, cutoff, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.highpass_biquad(y, sr, cutoff)
    return y.squeeze()

def bandpass(y, center_freq, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.bandpass_biquad(y, sr, center_freq, Q=1.0)
    return y.squeeze()

# ----------------------
# Instrument generators
# ----------------------
def build_kick_fx(params, sr=44100):
    freq = params.get('freq', 60)
    duration = params.get('duration', 0.5)
    y = sine_wave(freq, duration, sr)
    y = adsr_envelope(y, sr, attack=0.01, decay=0.2, sustain=0.0, release=0.05)
    y = lowpass(y, cutoff=150, sr=sr)
    return y

def build_bass_fx(params, sr=44100):
    freq = params.get('freq', 100)
    duration = params.get('duration', 1.0)
    y = sine_wave(freq, duration, sr)
    y = adsr_envelope(y, sr, attack=0.01, decay=0.2, sustain=0.8, release=0.1)
    cutoff = params.get('filter_cut', 400)
    y = lowpass(y, cutoff=cutoff, sr=sr)
    return y

def build_hat_fx(params, sr=44100):
    duration = params.get('duration', 0.2)
    y = white_noise(duration, sr)
    y = adsr_envelope(y, sr, attack=0.01, decay=0.05, sustain=0.3, release=0.05)
    cutoff = params.get('filter_cut', 8000)
    y = bandpass(y, center_freq=cutoff, sr=sr)
    return y

# ----------------------
# Feature extraction
# ----------------------
def compute_features(y, sr=44100):
    feats = {}
    Y = y.numpy().astype(float)
    feats['rms'] = float(np.sqrt(np.mean(Y**2)))
    return feats

# ----------------------
# Sample generation
# ----------------------
def generate_samples(presets_json, out_dir, n_total=1000, sr=44100, seed=42):
    np.random.seed(seed)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    presets = load_presets(presets_json)
    rows = []
    pbar = tqdm(total=n_total, desc='Generating samples')
    for i in range(n_total):
        inst = np.random.choice(list(presets.keys()))
        inst_dir = out_dir / inst
        ensure_dir(inst_dir)
        preset_list = presets[inst]
        preset = np.random.choice(preset_list)
        params = jitter_params(preset, jitter_fraction=0.05)
        params['duration'] = params.get('duration', 1.0)
        if inst == 'kick':
            audio = build_kick_fx(params, sr=sr)
        elif inst == 'bass':
            audio = build_bass_fx(params, sr=sr)
        elif inst == 'hat':
            audio = build_hat_fx(params, sr=sr)
        else:
            continue
        wav_path = inst_dir / f"{inst}_{i:05d}.wav"
        torchaudio.save(str(wav_path), audio.unsqueeze(0), sr)
        feats = compute_features(audio, sr=sr)
        row = {'file': str(wav_path.relative_to(out_dir)), 'instrument': inst}
        row.update(params)
        row.update(feats)
        rows.append(row)
        pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'metadata.csv', index=False)
    print(f"Saved {len(df)} samples to {out_dir} and metadata.csv")
    return df

# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    presets_json = 'dnb_presets.json'
    out_dir = './dnb_samples'
    generate_samples(presets_json, out_dir)
