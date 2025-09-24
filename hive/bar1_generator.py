import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

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

def adsr_envelope(y, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    N = y.shape[0]
    env = torch.ones(N)
    A = int(sr*attack)
    D = int(sr*decay)
    R = int(sr*release)
    S = N - (A+D+R)
    if S < 0:  # scale segments proportionally if total > N
        total = A+D+R
        scale = N/total
        A = int(A*scale)
        D = int(D*scale)
        R = N - A - D
        S = 0
    pointer = 0
    if A>0:
        env[pointer:pointer+A] = torch.linspace(0,1,A)
        pointer += A
    if D>0:
        env[pointer:pointer+D] = torch.linspace(1,sustain,D)
        pointer += D
    if S>0:
        env[pointer:pointer+S] = sustain
        pointer += S
    if R>0:
        env[pointer:pointer+R] = torch.linspace(env[pointer-1],0,R)
    return y*env

def sine_wave(freq,duration,sr=44100):
    t = torch.arange(0,duration,1/sr)
    return torch.sin(2*np.pi*freq*t)

def white_noise(duration,sr=44100):
    samples = int(duration*sr)
    return torch.rand(samples)*2-1

def lowpass(y,cutoff,sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.lowpass_biquad(y,sr,cutoff)
    return y.squeeze()

def bandpass(y, center_freq, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.bandpass_biquad(y, sr, center_freq, Q=1.0)
    return y.squeeze()

# ----------------------
# Instrument generators
# ----------------------
def build_kick(params,sr=44100):
    freq = params.get('sub_freq',60)
    duration = params.get('duration',0.5)
    y = sine_wave(freq,duration,sr)
    y = adsr_envelope(y,sr,attack=0.01,decay=params.get('sub_decay',0.2),sustain=0.0,release=0.05)
    y = lowpass(y, cutoff=150, sr=sr)
    return y

def build_bass(params,sr=44100):
    freq = params.get('freq',100)
    duration = params.get('duration',1.0)
    y = sine_wave(freq,duration,sr)
    y = adsr_envelope(y,sr,attack=0.01,decay=params.get('decay',0.2),sustain=params.get('sustain',0.8),release=0.1)
    cutoff = params.get('filter_cut',400)
    y = lowpass(y, cutoff=cutoff, sr=sr)
    return y

def build_hat(params,sr=44100):
    duration = params.get('duration',0.2)
    y = white_noise(duration,sr)
    y = adsr_envelope(y,sr,attack=0.01,decay=params.get('decay',0.05),sustain=params.get('sustain',0.3),release=0.05)
    cutoff = params.get('filter_cut',8000)
    y = bandpass(y, center_freq=cutoff, sr=sr)
    return y

def build_snare(params,sr=44100):
    duration = params.get('duration',0.3)
    y = white_noise(duration,sr)
    y = adsr_envelope(y,sr,attack=params.get('attack',0.005),decay=params.get('decay',0.1),sustain=0.0,release=params.get('release',0.01))
    cutoff = params.get('tone_freq',2000)
    y = bandpass(y, center_freq=cutoff,sr=sr)
    if params.get('distortion',0) > 0:
        y = torch.tanh(y*(1 + params['distortion']*5))
    return y

# ----------------------
# Feature extraction
# ----------------------
def compute_features(y):
    Y = y.numpy()
    return {
        'rms': float(np.sqrt(np.mean(Y**2))),
        'max': float(np.max(Y)),
        'min': float(np.min(Y)),
        'mean': float(np.mean(Y))
    }

# ----------------------
# Generate one bar for a single instrument
# ----------------------
def generate_one_bar_instrument(sound_presets, pattern_presets, instrument, bpm=174, sr=44100):
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
            bar, params = generate_one_bar_instrument(sound_presets, pattern_presets, inst, bpm, sr)
            if bar.abs().max() < 1e-6:
                continue
            file_path = inst_dir / f"{inst}_loop_{i:05d}.wav"
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
