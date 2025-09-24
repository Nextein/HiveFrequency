import os
import json
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# -----------------------------
# Utility
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------
# Flattened Loader
# -----------------------------
def group_presets_by_instrument(flat_presets):
    grouped = {}
    for p in flat_presets:
        inst = p.get("instrument")
        if inst not in grouped:
            grouped[inst] = []
        grouped[inst].append(p)
    return grouped

# -----------------------------
# Sound Generators (unchanged)
# -----------------------------
def adsr(y, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    n = len(y)
    A = int(attack * sr)
    D = int(decay * sr)
    R = int(release * sr)
    S = max(0, n - (A + D + R))

    env = torch.zeros(n)
    if A > 0:
        env[:A] = torch.linspace(0, 1, A)
    if D > 0:
        env[A:A+D] = torch.linspace(1, sustain, D)
    if S > 0:
        env[A+D:A+D+S] = sustain
    if R > 0:
        env[A+D+S:] = torch.linspace(sustain, 0, R)
    return y * env

def build_kick(params, sr=44100, length=0.5):
    t = torch.linspace(0, length, int(sr*length))
    f0 = params.get("sub_freq", 50)
    decay = params.get("sub_decay", 0.2)
    pitch_env = torch.exp(-t/decay)
    y = torch.sin(2*np.pi * f0 * pitch_env.cumsum(0)/sr)
    return adsr(y, sr, attack=0.001, decay=decay, sustain=0.0, release=0.05)

def build_snare(params, sr=44100, length=0.5):
    t = torch.linspace(0, length, int(sr*length))
    noise = torch.randn_like(t)
    tone = torch.sin(2*np.pi*params.get("tone_freq",200)*t)
    mix = 0.5*noise + 0.5*tone
    return adsr(mix, sr, attack=0.005, decay=0.2, sustain=0.0, release=0.1)

def build_hat(params, sr=44100, length=0.3):
    t = torch.linspace(0, length, int(sr*length))
    noise = torch.randn_like(t)
    return adsr(noise, sr, attack=0.001, decay=0.05, sustain=0.0, release=0.02)

def build_bass(params, sr=44100, length=1.0):
    t = torch.linspace(0, length, int(sr*length))
    freq = params.get("freq", 55)
    osc_type = params.get("osc_type", "sine")
    if osc_type == "saw":
        y = 2*(t*freq - torch.floor(0.5 + t*freq))
    elif osc_type == "square":
        y = torch.sign(torch.sin(2*np.pi*freq*t))
    else:
        y = torch.sin(2*np.pi*freq*t)
    return adsr(y, sr, attack=0.01, decay=0.2, sustain=0.6, release=0.2)

# -----------------------------
# Pattern Player
# -----------------------------
def render_pattern(instrument, sound_params, pattern, sr, bpm, bars=1):
    steps = 16
    sec_per_bar = 60/bpm*4
    step_len = sec_per_bar/steps
    total_len = int(sec_per_bar * bars * sr)
    y = torch.zeros(total_len)

    for bar in range(bars):
        for i, hit in enumerate(pattern):
            if hit == 1:
                start = int((bar*steps + i) * step_len * sr)
                if instrument == "kick":
                    sample = build_kick(sound_params, sr)
                elif instrument == "snare":
                    sample = build_snare(sound_params, sr)
                elif instrument == "hat":
                    sample = build_hat(sound_params, sr)
                elif instrument == "bass":
                    sample = build_bass(sound_params, sr)
                else:
                    continue
                end = min(start+len(sample), total_len)
                y[start:end] += sample[:end-start]
    return y

# -----------------------------
# Mixer for 4-bar loops
# -----------------------------
def generate_four_bar_mix(sound_presets, pattern_presets, sr=44100, bpm=174):
    bars = 4
    sec_per_bar = 60/bpm*4
    total_len = int(sr*sec_per_bar*bars)
    mix = torch.zeros(total_len)

    for inst, presets in sound_presets.items():
        if not presets:
            continue
        sound_params = random.choice(presets)
        inst_patterns = [p for p in pattern_presets if p["instrument"] == inst]
        if not inst_patterns:
            continue

        base_pattern = random.choice(inst_patterns)["pattern"]
        for bar_num in range(bars):
            if bar_num == bars-1:
                pattern = random.choice(inst_patterns)["pattern"]
            else:
                pattern = base_pattern
            track = render_pattern(inst, sound_params, pattern, sr, bpm, bars=1)
            start_idx = int(bar_num * sec_per_bar * sr)
            end_idx = start_idx + len(track)
            mix[start_idx:end_idx] += track

    # Normalize
    mix = mix / (mix.abs().max() + 1e-6)
    return mix

# -----------------------------
# Main Loop Generator
# -----------------------------
def generate_loops(sound_file, pattern_file, out_dir="mixed_loops_4bar", n_loops=50, sr=44100, bpm=174):
    ensure_dir(out_dir)

    raw_sound_presets = load_json(sound_file)
    sound_presets = group_presets_by_instrument(raw_sound_presets)

    raw_pattern_presets = load_json(pattern_file)

    for i in tqdm(range(n_loops), desc="Generating 4-bar mixed loops"):
        loop = generate_four_bar_mix(sound_presets, raw_pattern_presets, sr, bpm)
        wav_path = os.path.join(out_dir, f"loop_{i:04d}.wav")
        torchaudio.save(wav_path, loop.unsqueeze(0), sr)

    print(f"Saved {n_loops} 4-bar mixed loops to {out_dir}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    generate_loops("dnb_presets.json", "pattern_presets.json", out_dir="mixed_loops_4bar", n_loops=20)
