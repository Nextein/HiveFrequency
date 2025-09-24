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
def adsr_envelope(y, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.05):
    """Exponential-like ADSR envelope for punchy sounds"""
    N = y.shape[0]
    env = torch.ones(N)
    A = int(sr*attack)
    D = int(sr*decay)
    R = int(sr*release)
    S = N - (A+D+R)
    if S < 0:
        scale = N/(A+D+R)
        A = int(A*scale)
        D = int(D*scale)
        R = N-A-D
        S = 0
    pointer = 0
    if A>0:
        env[pointer:pointer+A] = torch.linspace(0,1,A)**2  # exponential attack
        pointer += A
    if D>0:
        env[pointer:pointer+D] = torch.linspace(1,sustain,D)**1.5
        pointer += D
    if S>0:
        env[pointer:pointer+S] = sustain
        pointer += S
    if R>0:
        env[pointer:pointer+R] = torch.linspace(env[pointer-1],0,R)**1.2
    return y*env

def sine_wave(freq, duration, sr=44100):
    t = torch.arange(0,duration,1/sr)
    return torch.sin(2*np.pi*freq*t)

def white_noise(duration, sr=44100):
    samples = int(duration*sr)
    return torch.rand(samples)*2-1

def lowpass(y, cutoff, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.lowpass_biquad(y, sr, cutoff)
    return y.squeeze()

def highpass(y, cutoff, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.highpass_biquad(y, sr, cutoff)
    return y.squeeze()

def bandpass(y, center_freq, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.bandpass_biquad(y, sr, center_freq, Q=1.5)
    return y.squeeze()

# Instrument generators
# ----------------------
def build_kick(params, sr=44100):
    """
    Pro-level generic kick synthesizer:
    - Pitch envelope for body
    - Amplitude envelope
    - Click transient
    - Optional distortion & filtering
    """

    duration = params.get("duration", 0.5)
    n_samples = int(sr * duration)
    t = torch.linspace(0, duration, n_samples)

    # ----------------------------
    # 1. Pitch-swept sub oscillator
    # ----------------------------
    base_freq = params.get("sub_freq", 50)
    start_freq = params.get("start_freq", base_freq * 3)   # Vital-like sweep
    pitch_decay = params.get("pitch_decay", 30.0)

    pitch_curve = torch.exp(-t * pitch_decay)
    freqs = base_freq + (start_freq - base_freq) * pitch_curve
    phase = 2 * np.pi * torch.cumsum(freqs, dim=0) / sr
    sub = torch.sin(phase) * params.get("sub_level", 1.0)
    sub = adsr_envelope(sub, sr, attack=0.001,
                        decay=params.get("sub_decay", 0.2),
                        sustain=0.0, release=0.01)

    # ----------------------------
    # 2. Click / transient
    # ----------------------------
    click = torch.zeros_like(sub)
    if params.get("click_amp", 0) > 0:
        click_freq = params.get("click_tone_freq", 4000)
        click_decay = params.get("click_decay", 0.01)
        click_wave = torch.sin(2 * np.pi * click_freq * t)
        click_env = torch.exp(-t * (1 / max(click_decay, 1e-5)))
        click = click_wave * click_env * params["click_amp"]

    # ----------------------------
    # 3. Combine
    # ----------------------------
    y = sub + click

    # ----------------------------
    # 4. Distortion
    # ----------------------------
    drive = params.get("distortion", 0)
    if drive > 0:
        y = torch.tanh(y * (1 + drive * 5))

    # ----------------------------
    # 5. Optional tone shaping
    # ----------------------------
    if params.get("brightness", 0.5) > 0.6:
        y = highpass(y, cutoff=80 + params["brightness"] * 2000, sr=sr)
    else:
        y = lowpass(y, cutoff=200 + params["brightness"] * 4000, sr=sr)

    return y / (y.abs().max() + 1e-9)


def build_snare(params, sr=44100):
    """
    Snare: layered noise + optional tonal body
    """
    duration = params.get('duration',0.3)
    noise = white_noise(duration,sr)
    noise = bandpass(noise, center_freq=params.get('tone_freq',2000), sr=sr)
    noise = adsr_envelope(noise, sr, attack=params.get('attack',0.001), decay=params.get('decay',0.08), sustain=0.0, release=params.get('release',0.01))

    # Optional tonal layer
    freq = params.get('body_freq', 180)
    if freq>0:
        body = sine_wave(freq,duration,sr)
        body = adsr_envelope(body, sr, attack=0.001, decay=params.get('decay',0.08), sustain=0.0, release=0.01)
        noise += body

    # Optional distortion
    distortion = params.get('distortion',0)
    if distortion>0:
        noise = torch.tanh(noise*(1+distortion*3))

    snare = noise / (noise.abs().max() + 1e-9)
    return snare

def build_hat(params, sr=44100):
    """
    Hat: short, band-limited noise
    """
    duration = params.get('duration',0.05)
    noise = white_noise(duration,sr)
    noise = bandpass(noise, center_freq=params.get('tone_freq',8000), sr=sr)
    noise = adsr_envelope(noise, sr, attack=0.001, decay=params.get('decay',0.02), sustain=0.0, release=0.001)
    hat = noise / (noise.abs().max()+1e-9)
    return hat

def build_bass(params, sr=44100):
    """
    Bass: layered saw + sine + optional sub + filter envelope
    """
    duration = params.get('duration',1.0)
    freq = params.get('freq',50)
    sub_oct = params.get('sub_oct',True)
    filter_cut = params.get('filter_cut',200)

    # Main oscillator
    t = torch.arange(0,duration,1/sr)
    osc_type = params.get('osc_type','saw')
    if osc_type=='saw':
        main = 2*(t*freq - torch.floor(t*freq+0.5))
    else:
        main = torch.sin(2*np.pi*freq*t)

    # Optional sub layer
    if sub_oct:
        sub = torch.sin(2*np.pi*freq/2*t)*0.5
        main += sub

    # Filter envelope
    main = lowpass(main, cutoff=filter_cut, sr=sr)
    main = adsr_envelope(main, sr, attack=0.01, decay=params.get('decay',0.3), sustain=params.get('sustain',0.8), release=0.05)

    # Optional distortion
    drive = params.get('drive',0)
    if drive>0:
        main = torch.tanh(main*(1+drive*3))

    bass = main / (main.abs().max()+1e-9)
    return bass

def build_snare_fx(params, sr=44100):
    duration = params.get('duration', 0.3)
    y = white_noise(duration, sr)
    y = adsr_envelope(
        y, sr,
        attack=params.get('attack', 0.005),
        decay=params.get('decay', 0.1),
        sustain=0.0,
        release=params.get('release', 0.01)
    )
    cutoff = params.get('tone_freq', 2000)
    y = bandpass(y, center_freq=cutoff, sr=sr)
    if params.get('distortion', 0) > 0:
        y = torch.tanh(y * (1 + params['distortion']*5))
    return y


# ----------------------
# Feature extraction
# ----------------------
def compute_features(y, sr=44100):
    feats = {}
    Y = y.numpy().astype(float)
    feats['rms'] = float(np.sqrt(np.mean(Y**2)))
    feats['max'] = float(np.max(Y))
    feats['min'] = float(np.min(Y))
    feats['mean'] = float(np.mean(Y))
    return feats

# ----------------------
# Sample generation (flattened, ML-ready)
# ----------------------
def get_next_sample_name(instrument_folder, instrument_name):
    """
    Returns the next sequential sample name for a given instrument folder.
    E.g., kick_0.wav, kick_1.wav, etc.
    """
    if not os.path.exists(instrument_folder):
        os.makedirs(instrument_folder)

    existing_files = [f for f in os.listdir(instrument_folder) if f.lower().endswith(".wav")]
    
    # Find the highest existing index for this instrument
    indices = []
    for f in existing_files:
        name, ext = os.path.splitext(f)
        parts = name.split("_")
        if len(parts) == 2 and parts[0] == instrument_name and parts[1].isdigit():
            indices.append(int(parts[1]))
    
    next_index = max(indices) + 1 if indices else 0
    return f"{instrument_name}_{next_index}.wav"

def generate_samples(presets_json, out_dir, n_total=1000, sr=44100, seed=42):
    np.random.seed(seed)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    presets = load_presets(presets_json)  # this is a flat list
    rows = []
    pbar = tqdm(total=n_total, desc='Generating samples')
    for i in range(n_total):
        # pick a random flat preset
        preset = np.random.choice(presets)
        params = jitter_params(preset, jitter_fraction=0.05)
        inst = params['instrument']
        params['duration'] = params.get('duration', 1.0)

        inst_dir = out_dir / inst
        ensure_dir(inst_dir)

        # synthesize based on instrument
        if inst == 'kick':
            audio = build_kick(params, sr=sr)
        elif inst == 'bass':
            audio = build_bass(params, sr=sr)
        elif inst == 'hat':
            audio = build_hat(params, sr=sr)
        elif inst == 'snare':
            audio = build_snare(params, sr=sr)
        else:
            continue

        # save file
        wav_path = inst_dir / get_next_sample_name(inst_dir, inst)
        torchaudio.save(str(wav_path), audio.unsqueeze(0), sr)

        # compute metadata
        feats = compute_features(audio, sr=sr)
        row = {'file': str(wav_path.relative_to(out_dir))}
        row.update(params)  # keep instrument + style + all parameters
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
    presets_json = 'dnb_presets.json'  # flattened presets file
    out_dir = './dnb_samples'
    generate_samples(presets_json, out_dir)
