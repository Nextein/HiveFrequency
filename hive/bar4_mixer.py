# four_bar_mixer.py
import json
import math
from pathlib import Path
import numpy as np
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import random

# ----------------------
# Utilities
# ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def clamp(x, a=-1.0, b=1.0):
    return max(a, min(b, x))

def soft_limit(stereo: torch.Tensor, threshold=0.98):
    # stereo: [2, N]
    peak = stereo.abs().max().item()
    if peak <= threshold or peak == 0:
        return stereo
    # normalize close to threshold and apply gentle tanh shaping
    stereo = stereo / (peak / threshold)
    return torch.tanh(stereo)

def pan_mono_to_stereo(y: torch.Tensor, pan: float):
    """
    Equal-power panning.
    pan in [-1.0, 1.0] where -1 = hard left, 0 center, 1 = hard right.
    y: [N]
    returns: [2, N]
    """
    theta = (pan + 1.0) * (math.pi / 4.0)  # map -1..1 to 0..pi/2
    left_gain = math.cos(theta)
    right_gain = math.sin(theta)
    stereo = torch.stack([y * left_gain, y * right_gain], dim=0)
    return stereo

# ----------------------
# ADSR envelope (robust)
# ----------------------
def adsr_envelope(y: torch.Tensor, sr: int,
                  attack=0.01, decay=0.1, sustain=0.7, release=0.05):
    N = y.shape[0]
    A = max(0, int(sr * attack))
    D = max(0, int(sr * decay))
    R = max(0, int(sr * release))
    S = N - (A + D + R)
    if S < 0:
        # scale down A, D, R proportionally
        total = (A + D + R) or 1
        scale = N / total
        A = int(A * scale)
        D = int(D * scale)
        R = N - (A + D)
        S = 0
    env = torch.empty(N)
    pointer = 0
    if A > 0:
        env[pointer:pointer + A] = torch.linspace(0.0, 1.0, A)
        pointer += A
    if D > 0:
        env[pointer:pointer + D] = torch.linspace(1.0, sustain, D)
        pointer += D
    if S > 0:
        env[pointer:pointer + S] = sustain
        pointer += S
    if R > 0:
        # start value for release:
        start_val = env[pointer - 1].item() if pointer > 0 else sustain
        env[pointer:pointer + R] = torch.linspace(start_val, 0.0, R)
    return y * env

# ----------------------
# Basic synth building blocks (high-quality variants)
# ----------------------
def sine_wave(freq, duration, sr=44100):
    t = torch.arange(0, int(duration * sr), dtype=torch.float32) / sr
    return torch.sin(2.0 * math.pi * freq * t)

def white_noise(duration, sr=44100):
    n = int(duration * sr)
    return (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0)

def lowpass(y: torch.Tensor, cutoff, sr=44100):
    if cutoff <= 0:
        return torch.zeros_like(y)
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.lowpass_biquad(y, sr, cutoff)
    return y.squeeze()

def highpass(y: torch.Tensor, cutoff, sr=44100):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.highpass_biquad(y, sr, cutoff)
    return y.squeeze()

def bandpass(y: torch.Tensor, center, sr=44100, Q=1.5):
    y = y.unsqueeze(0).unsqueeze(0)
    y = torchaudio.functional.bandpass_biquad(y, sr, center, Q)
    return y.squeeze()

# ----------------------
# Instrument synths (improved)
# These are variants of the high-quality functions we've iterated on.
# Accepts 'params' dict (from sound presets). Returns mono torch.Tensor.
# ----------------------
def build_kick(params, sr=44100):
    duration = float(params.get("duration", 0.45))
    sub_freq = float(params.get("sub_freq", 50))
    sub_level = float(params.get("sub_level", 1.0))
    sub_decay = float(params.get("sub_decay", 0.18))
    click_freq = float(params.get("click_tone_freq", 4500))
    click_amp = float(params.get("click_amp", 0.7))
    click_decay = float(params.get("click_decay", 0.012))
    punch_freq = float(params.get("punch_freq", 110))
    punch_amp = float(params.get("punch_amp", 0.6))
    distortion = float(params.get("distortion", 0.0))

    n = int(duration * sr)
    t = torch.arange(0, n, dtype=torch.float32) / sr

    # Sub with pitch drop
    pitch_env = torch.exp(-12.0 * t)  # fast drop envelope
    sub = torch.sin(2.0 * math.pi * sub_freq * t * (1.0 + 0.5 * pitch_env))
    sub = adsr_envelope(sub * sub_level, sr, attack=0.001, decay=sub_decay, sustain=0.0, release=0.01)
    sub = lowpass(sub, cutoff=180, sr=sr)

    # Punch / mid transient
    punch = torch.sin(2.0 * math.pi * punch_freq * t) * punch_amp
    punch = adsr_envelope(punch, sr, attack=0.0005, decay=0.04, sustain=0.0, release=0.0)
    punch = lowpass(punch, cutoff=600, sr=sr)

    # Click layer
    click = torch.sin(2.0 * math.pi * click_freq * t) * click_amp
    click = adsr_envelope(click, sr, attack=0.0, decay=click_decay, sustain=0.0, release=0.0)
    click = highpass(click, cutoff=click_freq * 0.4, sr=sr)

    kick = sub + punch + click

    if distortion > 0:
        kick = torch.tanh(kick * (1.0 + distortion * 4.0))

    # normalize by peak to keep headroom
    peak = kick.abs().max().clamp(min=1e-9)
    kick = kick / peak * 0.95
    return kick

def build_snare(params, sr=44100):
    duration = float(params.get("duration", 0.28))
    noise_level = float(params.get("noise_level", 1.0))
    tone_freq = float(params.get("body_freq", params.get("tone_freq", 200.0)))
    body_level = float(params.get("body_level", 0.8))
    body_decay = float(params.get("body_decay", 0.12))
    crack_freq = float(params.get("crack_freq", 6000.0))
    crack_level = float(params.get("crack_level", 0.7))
    crack_decay = float(params.get("crack_decay", 0.03))
    distortion = float(params.get("distortion", 0.0))

    n = int(duration * sr)
    t = torch.arange(0, n, dtype=torch.float32) / sr

    # Noise body
    noise = white_noise(duration, sr) * noise_level
    noise = bandpass(noise, center=tone_freq * 4.0, sr=sr, Q=1.2)
    noise = adsr_envelope(noise, sr, attack=0.001, decay=body_decay, sustain=0.0, release=0.01)

    # tonal body
    body = torch.sin(2.0 * math.pi * tone_freq * t) * body_level
    body = adsr_envelope(body, sr, attack=0.002, decay=body_decay, sustain=0.0, release=0.01)
    body = lowpass(body, cutoff=max(200.0, tone_freq * 2.0), sr=sr)

    # crack transient
    crack = torch.sin(2.0 * math.pi * crack_freq * t) * crack_level
    crack = adsr_envelope(crack, sr, attack=0.0, decay=crack_decay, sustain=0.0, release=0.0)
    crack = highpass(crack, cutoff=crack_freq * 0.3, sr=sr)

    snare = noise + body + crack
    if distortion > 0:
        snare = torch.tanh(snare * (1.0 + distortion * 3.0))

    peak = snare.abs().max().clamp(min=1e-9)
    snare = snare / peak * 0.95
    return snare

def build_hat(params, sr=44100):
    duration = float(params.get("duration", 0.06))
    tone_freq = float(params.get("tone_freq", 8000.0))
    decay = float(params.get("decay", 0.02))
    noise_level = float(params.get("noise_level", 1.0))
    brightness = float(params.get("brightness", 0.6))

    noise = white_noise(duration, sr) * noise_level
    noise = bandpass(noise, center=tone_freq, sr=sr, Q=1.8)
    hat = adsr_envelope(noise * brightness, sr, attack=0.0005, decay=decay, sustain=0.0, release=0.0005)

    peak = hat.abs().max().clamp(min=1e-9)
    hat = hat / peak * 0.95
    return hat

def build_bass(params, sr=44100):
    duration = float(params.get("duration", 1.0))
    freq = float(params.get("freq", 50.0))
    sub_oct = bool(params.get("sub_oct", True))
    filter_cut = float(params.get("filter_cut", 200.0))
    decay = float(params.get("decay", 0.4))
    drive = float(params.get("drive", 0.0))
    osc_type = str(params.get("osc_type", "saw"))

    n = int(duration * sr)
    t = torch.arange(0, n, dtype=torch.float32) / sr

    # oscillator
    if osc_type == "saw":
        # naive saw via sawtooth formula
        main = 2.0 * (t * freq - torch.floor(t * freq + 0.5))
    else:
        main = torch.sin(2.0 * math.pi * freq * t)

    # sub layer
    if sub_oct:
        sub = 0.5 * torch.sin(2.0 * math.pi * (freq / 2.0) * t)
        main = main + sub

    main = lowpass(main, cutoff=filter_cut, sr=sr)
    main = adsr_envelope(main, sr, attack=0.01, decay=decay, sustain=0.8, release=0.05)

    if drive > 0:
        main = torch.tanh(main * (1.0 + drive * 3.0))

    peak = main.abs().max().clamp(min=1e-9)
    main = main / peak * 0.95
    return main

# ----------------------
# Pattern & preset selection helpers
# ----------------------
def select_pattern_for_instrument(pattern_presets, instrument, style, complexity):
    # Try exact style + complexity first
    candidates = [p for p in pattern_presets if p["instrument"] == instrument and p.get("style") == style and p.get("complexity") == complexity]
    if candidates:
        return random.choice(candidates)
    # Relax: same style any complexity <= requested
    candidates = [p for p in pattern_presets if p["instrument"] == instrument and p.get("style") == style and p.get("complexity", 3) <= complexity]
    if candidates:
        return random.choice(candidates)
    # Fallback: any instrument pattern matching instrument
    candidates = [p for p in pattern_presets if p["instrument"] == instrument]
    if candidates:
        return random.choice(candidates)
    return None

def select_sound_preset(sound_presets, instrument, style):
    candidates = [p for p in sound_presets if p.get("instrument") == instrument and p.get("style") == style]
    if candidates:
        return random.choice(candidates)
    candidates = [p for p in sound_presets if p.get("instrument") == instrument]
    if candidates:
        return random.choice(candidates)
    return {}

# ----------------------
# Build per-instrument 1-bar and 4-bar tracks
# ----------------------
def build_one_bar_instrument(pattern_presets, sound_presets, instrument, style, complexity, bpm=174, sr=44100, jitter=True):
    """
    Returns tuple: (mono_bar_tensor [N], chosen_pattern_dict, chosen_sound_preset_dict)
    """
    step_dur = 60.0 / bpm / 4.0
    bar_samples = int(round(16 * step_dur * sr))
    bar = torch.zeros(bar_samples, dtype=torch.float32)

    pattern = select_pattern_for_instrument(pattern_presets, instrument, style, complexity)
    if pattern is None:
        return bar, None, None
    sound_preset = select_sound_preset(sound_presets, instrument, style)
    if sound_preset is None:
        return bar, pattern, None

    # Possibly jitter numeric params for variation
    sound_params = dict(sound_preset)
    if jitter:
        for k, v in sound_params.items():
            if isinstance(v, (int, float)):
                sound_params[k] = float(v * (1.0 + np.random.uniform(-0.03, 0.03)))

    patt = pattern["pattern"]
    step_samples = bar_samples // 16

    for i, hit in enumerate(patt):
        if not hit:
            continue
        start = int(i * step_samples)
        if instrument == "kick":
            y = build_kick(sound_params, sr)
        elif instrument == "snare":
            y = build_snare(sound_params, sr)
        elif instrument == "hat":
            y = build_hat(sound_params, sr)
        elif instrument == "bass":
            # For bass, allow shorter notes (slice) or longer sustain; use sound_params duration
            y = build_bass(sound_params, sr)
        else:
            continue

        # trim or paste
        end = min(start + y.shape[0], bar_samples)
        bar[start:end] += y[: end - start]

    # keep relative level influenced by pattern loudness if present
    loudness = float(pattern.get("loudness", 1.0))
    bar = bar * loudness

    # normalize lightly to avoid silent bars
    peak = bar.abs().max()
    if peak > 0:
        bar = bar / (peak + 1e-9) * 0.95

    return bar, pattern, sound_params

def build_four_bar_instrument(pattern_presets, sound_presets, instrument, style, complexity, bpm=174, sr=44100):
    # Bars 1-3 identical; bar4 is variation
    bar1, pattern1, sound_preset = build_one_bar_instrument(pattern_presets, sound_presets, instrument, style, complexity, bpm, sr, jitter=True)
    if pattern1 is None or sound_preset is None:
        # empty 4 bars
        zeros = torch.zeros(bar1.shape[0] * 4, dtype=torch.float32)
        return zeros, [None, None, None, None], None
    # Prepare bars
    b1 = bar1
    b2 = b1.clone()
    b3 = b1.clone()

    # Create bar4: 50% chance new pattern (same style/complexity), 50% mutate (add ghost hits or rolls)
    if random.random() < 0.5:
        bar4, pattern4, _ = build_one_bar_instrument(pattern_presets, sound_presets, instrument, style, complexity, bpm, sr, jitter=True)
        if pattern4 is None:
            bar4 = b1.clone()
            pattern4 = pattern1
    else:
        # mutate base bar: add small ghost hits (hats/snare) near the end or random index
        bar4 = b1.clone()
        # choose candidate patterns for small insertions
        step_samples = b1.shape[0] // 16
        n_inserts = random.choice([1, 2])
        for _ in range(n_inserts):
            # pick a small transient to insert
            if instrument == "hat":
                small = build_hat(sound_preset, sr)
            elif instrument == "snare":
                small = build_snare(sound_preset, sr)
            elif instrument == "kick":
                small = build_kick(sound_preset, sr)
            elif instrument == "bass":
                small = build_bass(sound_preset, sr)
            else:
                small = build_hat(sound_preset, sr)
            # choose insertion point biased toward the end (bar 4 feel)
            pos_step = random.choice([12, 13, 14, 15, random.randint(0, 15)])
            start = pos_step * step_samples
            end = min(start + small.shape[0], b1.shape[0])
            bar4[start:end] += small[:end - start]
        pattern4 = dict(pattern1)  # not a real pattern, but keep a reference

    final = torch.cat([b1, b2, b3, bar4])
    # light normalization per instrument track so mixing can balance
    peak = final.abs().max()
    if peak > 0:
        final = final / (peak + 1e-9) * 0.95
    return final, [pattern1, pattern1, pattern1, pattern4], sound_preset

# ----------------------
# Mixer: combines instrument tracks into stereo
# ----------------------
def mix_tracks_to_stereo(tracks_mono: dict, mix_options: dict, sr=44100):
    """
    tracks_mono: dict of instrument -> mono tensor [N]
    mix_options: dict with per-instrument gain and pan, e.g. {'kick':{'gain':1.0,'pan':0.0},...}
    returns stereo tensor [2, N]
    """
    # find longest track
    lengths = [t.shape[0] for t in tracks_mono.values() if t is not None]
    if not lengths:
        return torch.zeros((2, 1), dtype=torch.float32)
    total_len = max(lengths)
    stereo = torch.zeros((2, total_len), dtype=torch.float32)

    for inst, mono in tracks_mono.items():
        if mono is None:
            continue
        # pad to total_len
        if mono.shape[0] < total_len:
            pad = torch.zeros(total_len - mono.shape[0], dtype=torch.float32)
            mono = torch.cat([mono, pad])

        opts = mix_options.get(inst, {})
        gain = float(opts.get("gain", 1.0))
        pan = float(opts.get("pan", 0.0))
        # apply instrument loudness scaling if present
        stereo_inst = pan_mono_to_stereo(mono * gain, pan)
        stereo = stereo + stereo_inst

    # apply master soft limiter
    stereo = soft_limit(stereo, threshold=0.98)
    # ensure float32
    return stereo.type(torch.float32)

# ----------------------
# Features for file metadata
# ----------------------
def compute_audio_features(stereo: torch.Tensor, sr=44100):
    # stereo: [2, N]
    mono = stereo.mean(dim=0).numpy()
    feats = {}
    feats['rms'] = float(np.sqrt(np.mean(mono**2)))
    feats['peak'] = float(np.max(np.abs(mono)))
    feats['duration_s'] = float(mono.shape[0] / sr)

    # spectral centroid (simple)
    magn = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0/sr)
    if magn.sum() > 0:
        feats['spectral_centroid'] = float((magn * freqs).sum() / magn.sum())
    else:
        feats['spectral_centroid'] = 0.0
    return feats

# ----------------------
# Top-level batch generator
# ----------------------
def generate_many_mixed_four_bar_loops(sound_json: str,
                                      pattern_json: str,
                                      out_dir: str,
                                      n_loops: int = 100,
                                      style: str = "jungle",
                                      complexity: int = 3,
                                      bpm: int = 174,
                                      sr: int = 44100,
                                      mix_profile: dict = None):
    """
    mix_profile: optional dict with per-instrument gain/pan defaults, e.g.:
      {"kick":{"gain":1.0,"pan":0.0}, "snare":{"gain":0.9,"pan":0.0}, "hat":{"gain":0.7,"pan":-0.3}, "bass":{"gain":1.0,"pan":0.0}}
    """
    sound_presets = load_json(sound_json)
    pattern_presets = load_json(pattern_json)

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    if mix_profile is None:
        # sensible defaults
        mix_profile = {
            "kick": {"gain": 1.0, "pan": 0.0},
            "snare": {"gain": 0.9, "pan": 0.0},
            "hat": {"gain": 0.6, "pan": -0.25},
            "hat2": {"gain": 0.6, "pan": 0.25},
            "bass": {"gain": 1.0, "pan": 0.0}
        }

    rows = []
    pbar = tqdm(total=n_loops, desc="Generating mixed 4-bar loops")

    for i in range(n_loops):
        # Build instrument tracks
        instruments = ["kick", "snare", "hat", "bass"]
        # choose per-loop pan variation for hats to stereo spread
        hat_pan = np.random.uniform(-0.4, -0.1)
        hat2_pan = np.random.uniform(0.1, 0.4)

        tracks = {}
        per_instrument_info = {}

        for inst in instruments:
            track, patterns_used, sound_used = build_four_bar_instrument(pattern_presets, sound_presets, inst, style, complexity, bpm, sr)
            if inst == "hat":
                # we will split hat into two stereo hat tracks by duplicating and panning
                tracks["hat_L"] = track.clone()
                tracks["hat_R"] = track.clone()
                # We'll set them into mix with different pans below
                per_instrument_info["hat"] = {"patterns": patterns_used, "preset": sound_used}
            else:
                tracks[inst] = track
                per_instrument_info[inst] = {"patterns": patterns_used, "preset": sound_used}

        # Now define mix options per track key
        mix_opts = {}
        mix_opts["kick"] = mix_profile.get("kick", {"gain": 1.0, "pan": 0.0})
        mix_opts["snare"] = mix_profile.get("snare", {"gain": 0.9, "pan": 0.0})
        mix_opts["bass"] = mix_profile.get("bass", {"gain": 1.0, "pan": 0.0})
        # hats left/right
        mix_opts["hat_L"] = {"gain": mix_profile.get("hat", {}).get("gain", 0.6), "pan": hat_pan}
        mix_opts["hat_R"] = {"gain": mix_profile.get("hat", {}).get("gain", 0.6), "pan": hat2_pan}

        # Mix to stereo
        stereo = mix_tracks_to_stereo(tracks, mix_opts, sr=sr)  # [2, N]

        # Final trim (avoid super-long tail): find last non-near-zero and trim small tail
        mono_mean = stereo.mean(dim=0)
        last_idx = int((mono_mean.abs() > 1e-4).nonzero().max().item()) if (mono_mean.abs() > 1e-4).any() else stereo.shape[1]-1
        stereo = stereo[:, :max(last_idx + 1, int(sr * (60.0 / bpm * 4.0) * 4))]

        # final peak normalize & limiter
        stereo = stereo / (stereo.abs().max() + 1e-9) * 0.98
        stereo = soft_limit(stereo, threshold=0.98)

        # save file
        filename = f"{style}_c{complexity}_{i:05d}.wav"
        out_path = out_dir / filename
        torchaudio.save(str(out_path), stereo, sr)

        # metadata
        feats = compute_audio_features(stereo, sr=sr)
        meta_row = {
            "file": str(out_path.name),
            "style": style,
            "complexity": complexity,
            "bpm": bpm,
            "rms": feats["rms"],
            "peak": feats["peak"],
            "spectral_centroid": feats["spectral_centroid"],
            "duration_s": feats["duration_s"]
        }
        # attach per-instrument preset + pattern info as JSON strings (safe for CSV)
        for inst, info in per_instrument_info.items():
            meta_row[f"{inst}_preset"] = json.dumps(info.get("preset", {}))
            meta_row[f"{inst}_patterns"] = json.dumps(info.get("patterns", []))
        rows.append(meta_row)
        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Saved {n_loops} mixed 4-bar loops to {out_dir} and metadata.csv")

generate_many_mixed_four_bar_loops(
    sound_json="dnb_presets.json",
    pattern_json="pattern_presets.json",
    out_dir="dnb_mixed_4bar_loops",
    n_loops=200,
    style="neurofunk",
    complexity=4,
    bpm=174,
    sr=44100
)
