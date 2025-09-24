import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from one_shot_generator import *
# ----------------------
# Utility
# ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
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
# One-bar generator
# ----------------------
def generate_one_bar(pattern_presets, sound_presets, style, complexity, sr=44100, bpm=174):
    step_dur = 60 / bpm / 4
    bar_samples = int(16 * step_dur * sr)
    final_bar = torch.zeros(bar_samples)

    instruments = ["kick", "snare", "hat", "bass"]

    for inst in instruments:
        # choose sound preset
        inst_sounds = [p for p in sound_presets if p["instrument"] == inst and p["style"] == style]
        if not inst_sounds:
            continue
        sound_params = jitter_params(np.random.choice(inst_sounds))

        # choose pattern preset
        inst_patterns = [
            p for p in pattern_presets
            if p["instrument"] == inst and p["style"] == style and p["complexity"] == complexity
        ]
        if not inst_patterns:
            continue
        pattern = np.random.choice(inst_patterns)["pattern"]

        # synthesize hits
        for i, hit in enumerate(pattern):
            if hit == 0:
                continue
            start_sample = int(i * step_dur * sr)

            if inst == "kick":
                y = build_kick(sound_params, sr)
            elif inst == "snare":
                y = build_snare(sound_params, sr)
            elif inst == "hat":
                y = build_hat(sound_params, sr)
            elif inst == "bass":
                y = build_bass(sound_params, sr)
            else:
                continue

            end_sample = start_sample + len(y)
            if end_sample > bar_samples:
                y = y[:bar_samples - start_sample]
                end_sample = bar_samples
            final_bar[start_sample:end_sample] += y

    # normalize
    final_bar = final_bar / (final_bar.abs().max() + 1e-9)
    return final_bar


# ----------------------
# 4-bar generator
# ----------------------
def generate_four_bar_loop(pattern_presets, sound_presets, style="jungle", complexity=3, sr=44100, bpm=174):
    base_bar = generate_one_bar(pattern_presets, sound_presets, style, complexity, sr, bpm)

    # Bars 1–3 repeat
    bars = [base_bar, base_bar.clone(), base_bar.clone()]

    # Bar 4 variation
    if np.random.rand() < 0.5:
        # generate fresh pattern
        bar4 = generate_one_bar(pattern_presets, sound_presets, style, complexity, sr, bpm)
    else:
        # mutate base bar
        bar4 = base_bar.clone()
        variation = torch.zeros_like(bar4)
        step = bar4.shape[0] // 16
        for _ in range(np.random.randint(1, 3)):
            inst = np.random.choice(["snare", "hat", "kick"])
            sound_params = jitter_params(np.random.choice(
                [p for p in sound_presets if p["instrument"] == inst and p["style"] == style]
            ))
            if inst == "kick":
                y = build_kick(sound_params, sr)
            elif inst == "snare":
                y = build_snare(sound_params, sr)
            else:
                y = build_hat(sound_params, sr)
            pos = np.random.choice([12, 13, 14, 15]) * step  # variation at end
            end = min(pos + len(y), variation.shape[0])
            variation[pos:end] += y[:end - pos]
        bar4 += variation

    bars.append(bar4)

    # Concatenate 4 bars
    loop = torch.cat(bars)
    loop = loop / (loop.abs().max() + 1e-9)
    return loop


# ----------------------
# Batch generator
# ----------------------
def generate_many_loops(sound_json, pattern_json, out_dir, n_loops=100, style="jungle", complexity=3, bpm=174, sr=44100):
    sound_presets = load_json(sound_json)
    pattern_presets = load_json(pattern_json)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    rows = []
    pbar = tqdm(total=n_loops, desc="Generating 4-bar loops")
    for i in range(n_loops):
        loop = generate_four_bar_loop(pattern_presets, sound_presets, style=style, complexity=complexity, sr=sr, bpm=bpm)
        file_path = out_dir / f"{style}_c{complexity}_{i:05d}.wav"
        torchaudio.save(str(file_path), loop.unsqueeze(0), sr)
        rows.append({"file": str(file_path), "style": style, "complexity": complexity, "bpm": bpm})
        pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Saved {n_loops} 4-bar loops to {out_dir} and metadata.csv")


# ----------------------
# Example main
# ----------------------
if __name__ == "__main__":
    generate_many_loops("dnb_presets.json", "pattern_presets.json", out_dir="dnb_4bar_loops", n_loops=50, style="jungle", complexity=3)
