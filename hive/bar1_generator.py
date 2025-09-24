import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from one_shot_generator import *

# ----------------------
# Utility functions
# ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def jitter_params(params, jitter_fraction=0.05):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            jitter = np.random.uniform(-1, 1) * jitter_fraction * v
            new_params[k] = float(v + jitter)
        else:
            new_params[k] = v
    return new_params

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# ----------------------
# Generate one bar for a single instrument
# ----------------------
def generate_one_bar(sound_presets, pattern_presets, instrument, bpm=174, sr=44100):
    step_dur = 60/bpm/4
    bar_samples = int(16*step_dur*sr)
    bar = torch.zeros(bar_samples)

    # Filter presets for this instrument
    inst_sounds = [p for p in sound_presets if p['instrument']==instrument]
    if not inst_sounds:
        return bar, {}
    sound_params = jitter_params(np.random.choice(inst_sounds))

    inst_patterns = [p for p in pattern_presets if p['instrument']==instrument]
    if not inst_patterns:
        return bar, {}
    pattern = np.random.choice(inst_patterns)['pattern']

    for i, hit in enumerate(pattern):
        if hit==0: continue
        start_sample = int(i*step_dur*sr)
        if instrument=='kick': y = build_kick(sound_params,sr)
        elif instrument=='snare': y = build_snare(sound_params,sr)
        elif instrument=='hat': y = build_hat(sound_params,sr)
        elif instrument=='bass': y = build_bass(sound_params,sr)
        else: continue
        end_sample = start_sample+len(y)
        if end_sample>bar_samples:
            y = y[:bar_samples-start_sample]
        bar[start_sample:start_sample+len(y)] += y

    bar = bar / (bar.abs().max()+1e-9)  # normalize
    return bar, sound_params

# ----------------------
# Batch generation per instrument
# ----------------------
def generate_many_loops_per_instrument(sound_json, pattern_json, out_dir, n_loops=1000, bpm=174, sr=44100):
    sound_presets = load_json(sound_json)
    pattern_presets = load_json(pattern_json)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    instruments = ['kick','snare','hat','bass']
    metadata_rows = {inst: [] for inst in instruments}

    for inst in instruments:
        inst_dir = out_dir / inst
        ensure_dir(inst_dir)
        pbar = tqdm(total=n_loops, desc=f'Generating {inst} loops')
        for i in range(n_loops):
            bar, params = generate_one_bar(sound_presets, pattern_presets, inst, bpm, sr)
            if bar.abs().max() < 1e-6:
                continue
            file_path = inst_dir / f"{inst}_loop_{i:05d}_1bar_{bpm}bpm.wav"
            torchaudio.save(str(file_path), bar.unsqueeze(0), sr)
            feats = compute_features(bar)
            row = {'file': str(file_path.relative_to(out_dir)), 'instrument':inst, 'bpm':bpm}
            row.update(params)
            row.update(feats)
            metadata_rows[inst].append(row)
            pbar.update(1)
        pbar.close()
        df = pd.DataFrame(metadata_rows[inst])
        df.to_csv(inst_dir / 'metadata.csv', index=False)
        print(f"Saved {len(df)} {inst} loops and metadata to {inst_dir}/metadata.csv")

# ----------------------
# Main
# ----------------------
if __name__=='__main__':
    generate_many_loops_per_instrument('dnb_presets.json','pattern_presets.json',out_dir='dnb_loops',n_loops=1000)
