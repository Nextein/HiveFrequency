"""
DNB Sample & Feature Generator
File: dnb_sample_generator.py

Purpose:
- Generate high-quality parametric Drum & Bass one-shot samples (kick, bass, hat) using either
  *pyo* (recommended high-quality synthesis) or a portable NumPy-based synthesizer fallback.
- Extract a rich set of audio features and compute normalized attribute labels (snappy, brightness,
  distortion, pitch, decay, loudness, etc.).
- Save WAV files and a CSV containing synthesis parameters, raw features, and normalized attribute labels.

Usage examples:
  python dnb_sample_generator.py --out_dir ./dnb_samples --n_total 1000 --instruments kick bass hat --use_pyo
  python dnb_sample_generator.py --out_dir ./dnb_samples --n_total 1000 --instruments kick --no_pyo

Design notes:
- The script prefers the pyo engine (higher fidelity) if --use_pyo is set and pyo is available.
  If pyo is not available or you pass --no_pyo, a high-quality NumPy DSP fallback is used.
- Feature extraction tries to use librosa + pyloudnorm when available, else falls back to FFT-based approximations.

Dependencies (recommended):
  pip install numpy scipy soundfile pandas tqdm librosa pyloudnorm
  # pyo is optional and may require platform-specific install (conda or system packages). If you want pyo:
  conda install -c conda-forge pyo
  or follow pyo docs: https://ajaxsoundstudio.com/pyodoc/

Notes for your laptop:
- Your Ryzen 7 5800H + RTX 3050 is excellent for training and generation. The sample generator is CPU-bound
  for synthesis but will run fast locally. Training models benefits from the GPU; 4GB VRAM is enough for
  smaller VAEs and shallower neural vocoders. Keep batch sizes small (8-32) when training on your GPU.

Output:
- WAV files: OUT_DIR/<instrument>/...wav
- CSV: OUT_DIR/metadata.csv (contains synthesis params, raw features, normalized attributes)

"""

import os
import argparse
import math
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.signal import get_window, butter, sosfilt

# Optional dependencies: librosa, pyloudnorm, pyo
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import pyloudnorm as pyln
    HAS_PYLUFS = True
except Exception:
    HAS_PYLUFS = False

try:
    import pyo
    HAS_PYO = True
except Exception:
    HAS_PYO = False

# ----------------------------- Utilities -----------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# simple stable minmax scaler
def minmax_scale(arr):
    a = np.array(arr, dtype=float)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)

# ----------------------------- Synthesis (NumPy) ---------------------------
def synth_kick_numpy(sr=44100, duration=1.0,
                     sub_freq=50.0, sub_level=0.9, sub_decay=0.18,
                     click_level=0.6, click_decay=0.01,
                     pitch_drop=0.2, pitch_glide_time=0.06,
                     noise_level=0.02, distortion=0.0, brightness_db=0.0,
                     gain_db=0.0):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    # sub oscillator with exponential decay and pitch glide
    env_sub = np.exp(-t / max(1e-4, sub_decay))
    start_freq = sub_freq * (1.0 + 0.25 * pitch_drop)
    glide = np.clip(t / max(1e-6, pitch_glide_time), 0, 1.0)
    freq_t = start_freq * (1.0 - glide) + sub_freq * glide
    phase = 2 * np.pi * np.cumsum(freq_t) / sr
    sub = np.sin(phase) * env_sub * sub_level

    # click element: shaped noise + high-frequency sine impulse
    click_env = np.exp(-t / max(1e-5, click_decay))
    click_env = click_env * np.exp(-((t) / (max(1e-5, click_decay)*2))**2)
    noise = np.random.randn(len(t)) * noise_level
    click = noise * click_env * click_level
    click_pulse = np.sin(2*np.pi*4000*t) * (np.exp(-t/(max(1e-6, click_decay*0.5)))) * click_level*0.6
    click = click + click_pulse

    x = sub + click

    # soft saturation (tanh-based)
    if distortion and distortion > 0:
        drive = 1.0 + float(distortion) * 10.0
        x = np.tanh(x * drive) / np.tanh(drive)

    # brightness: tiny high-frequency boost through subtractive LP
    if abs(brightness_db) > 0.01:
        lp_len = int(sr * 0.002)
        if lp_len >= 1:
            kernel = np.ones(lp_len) / lp_len
            lp = np.convolve(x, kernel, mode='same')
            hp = x - lp
            gain = 10**(brightness_db / 20.0)
            x = x + hp * (gain - 1.0)

    # global gain and normalization
    x = x * (10**(gain_db / 20.0))
    maxv = np.max(np.abs(x)) + 1e-9
    if maxv > 0:
        x = x / maxv * 0.95
    return x.astype(np.float32)


def synth_hat_numpy(sr=44100, duration=0.25, noise_level=0.7, body_level=0.2, tone_freq=8000.0,
                    decay=0.02, brightness_db=0.0, gain_db=0.0):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    env = np.exp(-t / decay)
    noise = np.random.randn(len(t)) * noise_level
    # tone component for metallic character
    tone = np.sin(2*np.pi*tone_freq * t) * body_level
    x = (noise + tone) * env
    if abs(brightness_db) > 1e-3:
        # small HF boost
        lp_len = int(sr * 0.001)
        if lp_len >= 1:
            kernel = np.ones(lp_len) / lp_len
            lp = np.convolve(x, kernel, mode='same')
            hp = x - lp
            gain = 10**(brightness_db / 20.0)
            x = x + hp * (gain - 1.0)
    x = x * (10**(gain_db / 20.0))
    maxv = np.max(np.abs(x)) + 1e-9
    if maxv > 0:
        x = x / maxv * 0.95
    # pad to 1s for consistent length in dataset
    y = np.zeros(int(sr * 1.0), dtype=np.float32)
    y[:len(x)] = x
    return y


def synth_bass_numpy(sr=44100, duration=1.0, freq=55.0, osc_type='saw', amp=0.9, decay=0.5,
                     filter_cut=200.0, filter_q=0.7, sub_octave=False, drive=0.0, gain_db=0.0):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    if osc_type == 'saw':
        # naive band-limited saw approximation via additive
        n_harm = max(4, int(sr / (2*freq)))
        x = np.zeros_like(t)
        for h in range(1, min(60, n_harm)):
            x += (1.0/h) * np.sin(2*np.pi*freq*h*t)
        x = x * (2/np.pi)
    else:
        x = np.sin(2*np.pi*freq*t)
    if sub_octave:
        x = x + 0.5 * np.sin(2*np.pi*(freq/2)*t)
    env = np.exp(-t/ max(1e-4, decay))
    x = x * env * amp
    # lowpass via 2nd-order butterworth
    if filter_cut < (sr/2):
        sos = butter(2, filter_cut/(sr/2), btype='low', output='sos')
        x = sosfilt(sos, x)
    if drive and drive > 0:
        x = np.tanh(x * (1.0 + drive*5.0))
    x = x * (10**(gain_db/20.0))
    maxv = np.max(np.abs(x)) + 1e-9
    if maxv > 0:
        x = x / maxv * 0.95
    return x.astype(np.float32)

# ----------------------------- PYO synthesis (optional) --------------------
# Note: pyo must be installed on the system. Installing pyo is platform-specific.
# The pyo branch uses offline rendering to write high-quality audio. If pyo is unavailable
# the code falls back to the NumPy synthesizers above.

def synth_kick_pyo(sr=44100, duration=1.0, **kwargs):
    # Using pyo to construct a richer kick synth. This function returns a numpy array.
    s = pyo.Server(duplex=0).boot()
    s.recordOptions(dur=duration, filename="/tmp/pyo_tmp_render.wav", fileformat=0)
    s.start()
    # build pyo objects: sine for sub, FM for click, noise
    subfreq = kwargs.get('sub_freq', 50.0)
    subenv = pyo.Fader(fadein=0.0001, fadeout=kwargs.get('sub_decay',0.18), dur=duration).play()
    sub = pyo.Sine(freq=subfreq, mul=kwargs.get('sub_level',0.9))*subenv
    # click
    clickenv = pyo.Fader(fadein=0.0001, fadeout=kwargs.get('click_decay',0.01), dur=duration).play()
    click = pyo.Noise(mul=kwargs.get('noise_level',0.02))*clickenv
    # simple distortion
    if kwargs.get('distortion',0.0) > 0:
        sub = pyo.Disto(sub, drive=kwargs.get('distortion',0.2), slope=0.5)
    mix = sub + click
    mix.out()

    s.stop()
    s.shutdown()
    # read rendered file
    import soundfile as sf
    y, sr = sf.read('/tmp/pyo_tmp_render.wav')
    if y.ndim>1:
        y = y.mean(axis=1)
    return y.astype(np.float32)

# ----------------------------- Feature extraction -------------------------
def compute_features_librosa(y, sr=44100):
    feats = {}
    y = y.astype(float)
    # RMS
    feats['rms'] = float(np.mean(librosa.feature.rms(y=y)[0]))
    # LUFS if pyloudnorm available
    if HAS_PYLUFS:
        meter = pyln.Meter(sr)
        feats['lufs'] = float(meter.integrated_loudness(y))
    else:
        feats['lufs'] = float(20.0 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    feats['centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))
    feats['flatness'] = float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))
    feats['rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)[0]))
    feats['hnr'] = float(np.mean(librosa.effects.hpss(y)[0]**2) / (np.mean(librosa.effects.hpss(y)[1]**2)+1e-9))
    # pitch with yin (robust for pitched bass)
    try:
        pitch = librosa.yin(y, fmin=20, fmax=200, sr=sr)
        pitch_vals = pitch[~np.isnan(pitch)]
        feats['pitch'] = float(np.median(pitch_vals) if len(pitch_vals)>0 else 0.0)
    except Exception:
        feats['pitch'] = 0.0
    # snappiness: early RMS ratio
    n_early = int(sr * 0.05)
    early_rms = float(np.mean(librosa.feature.rms(y=y[:n_early])[0])) if n_early>0 else 0.0
    feats['snappiness'] = early_rms / (feats['rms'] + 1e-9)
    # decay estimate: time from peak to 10% of peak
    peak_idx = int(np.argmax(np.abs(y)))
    threshold = 0.1 * np.max(np.abs(y))
    below = np.where(np.abs(y[peak_idx:]) < threshold)[0]
    feats['decay_time_s'] = float(below[0]/sr) if len(below)>0 else float(len(y)/sr)
    Sfull = S
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    high_idx = np.where(freqs>3000)[0]
    feats['highband_ratio'] = float(np.mean(np.sum(Sfull[high_idx,:],axis=0))/ (np.mean(np.sum(Sfull,axis=0))+1e-9)) if len(high_idx)>0 else 0.0
    feats['crest'] = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2))+1e-9))
    # spectral contrast if available
    try:
        sc = librosa.feature.spectral_contrast(S=S, sr=sr)
        feats['spectral_contrast_mean'] = float(np.mean(sc))
    except Exception:
        feats['spectral_contrast_mean'] = 0.0
    return feats


def compute_features_fallback(y, sr=44100):
    y = y.astype(np.float32)
    feats = {}
    rms = float(np.sqrt(np.mean(y**2)))
    feats['rms'] = rms
    # approx loudness dB
    feats['lufs'] = float(20.0 * np.log10(rms + 1e-9))
    # FFT frame spectral features
    frame_len = 2048
    hop = 256
    # simple single-window FFT on signal
    N = 1 << ((len(y)-1).bit_length())
    X = np.fft.rfft(y * get_window('hann', len(y)), n=N)
    mags = np.abs(X)
    freqs = np.fft.rfftfreq(N, 1/sr)
    centroid = float(np.sum(mags * freqs) / (np.sum(mags) + 1e-9))
    feats['centroid'] = centroid
    # flatness: geometric mean over whole spectrum
    geo = np.exp(np.mean(np.log(mags + 1e-12)))
    arith = np.mean(mags + 1e-12)
    feats['flatness'] = float(geo / (arith + 1e-12))
    # rolloff 85%
    cumsum = np.cumsum(mags)
    total = cumsum[-1] + 1e-9
    idx = np.argmax(cumsum >= 0.85 * total)
    feats['rolloff'] = float(freqs[idx])
    # hnr approx
    median = np.median(mags)
    peaks = mags > (median * 3.0)
    harmonic_energy = np.sum(mags[peaks])
    noise_energy = np.sum(mags[~peaks]) + 1e-9
    feats['hnr'] = float(harmonic_energy / noise_energy)
    # pitch coarse via lowband FFT
    low_idx = np.where((freqs >= 20) & (freqs <= 200))[0]
    if len(low_idx) > 0:
        peak = low_idx[np.argmax(mags[low_idx])]
        feats['pitch'] = float(freqs[peak])
    else:
        feats['pitch'] = 0.0
    # snappiness
    n_early = int(sr * 0.05)
    early_rms = float(np.sqrt(np.mean(y[:n_early]**2))) if n_early < len(y) else float(np.sqrt(np.mean(y**2)))
    feats['snappiness'] = early_rms / (rms + 1e-9)
    # decay
    peak_idx = int(np.argmax(np.abs(y)))
    threshold = 0.1 * np.max(np.abs(y))
    below = np.where(np.abs(y[peak_idx:]) < threshold)[0]
    feats['decay_time_s'] = float(below[0]/sr) if len(below)>0 else float(len(y)/sr)
    # highband ratio >3k
    hb_idx = np.where(freqs > 3000)[0]
    if len(hb_idx)>0:
        hb_energy = np.sum(mags[hb_idx])
        feats['highband_ratio'] = float(hb_energy / (np.sum(mags) + 1e-9))
    else:
        feats['highband_ratio'] = 0.0
    feats['crest'] = float(np.max(np.abs(y)) / (rms + 1e-9))
    return feats

# Choose compute_features wrapper
def compute_features(y, sr=44100):
    if HAS_LIBROSA:
        return compute_features_librosa(y, sr)
    else:
        return compute_features_fallback(y, sr)

# ----------------------------- Dataset generation -------------------------
DEFAULT_FEATURES = ['rms','lufs','centroid','flatness','rolloff','hnr','pitch','snappiness','decay_time_s','highband_ratio','crest','spectral_contrast_mean']

def generate_dataset(out_dir: Path, n_total=1000, instruments=['kick'], sr=44100, duration=1.0, use_pyo=False, seed=None):
    random.seed(seed)
    np.random.seed(seed if seed is not None else 0)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    rows = []
    per_instrument = max(1, n_total // len(instruments))

    idx = 0
    pbar = tqdm(total=n_total, desc='Generating samples')
    for inst in instruments:
        inst_dir = out_dir / inst
        ensure_dir(inst_dir)
        for i in range(per_instrument):
            if idx >= n_total:
                break
            if inst == 'kick':
                # sample DnB-oriented distribution of parameters
                sub_freq = float(np.clip(np.random.normal(50, 7), 28, 90))
                sub_level = float(np.clip(np.random.beta(2,2), 0.5, 1.4))
                sub_decay = float(np.clip(np.random.normal(0.18, 0.06), 0.06, 0.7))
                click_level = float(np.clip(np.random.beta(2,1.5)*1.2, 0.05, 1.5))
                click_decay = float(np.clip(np.random.normal(0.012, 0.006), 0.002, 0.08))
                pitch_drop = float(np.clip(np.random.beta(2,5), 0.0, 0.95))
                pitch_glide_time = float(np.clip(np.random.normal(0.05,0.02), 0.005, 0.18))
                noise_level = float(np.clip(np.random.beta(1,6)*0.8, 0.0, 0.9))
                distortion_param = float(np.clip(np.random.beta(1.2,5)*0.8, 0.0, 1.4))
                brightness_db = float(np.clip(np.random.normal(0.0, 3.0), -8.0, 10.0))
                gain_db = float(np.clip(np.random.normal(-6.0, 4.0), -12.0, 6.0))
                if use_pyo and HAS_PYO:
                    y = synth_kick_pyo(sr=sr, duration=duration, sub_freq=sub_freq, sub_level=sub_level,
                                       sub_decay=sub_decay, click_level=click_level, click_decay=click_decay,
                                       pitch_drop=pitch_drop, pitch_glide_time=pitch_glide_time, noise_level=noise_level,
                                       distortion=distortion_param, brightness_db=brightness_db, gain_db=gain_db)
                else:
                    y = synth_kick_numpy(sr=sr, duration=duration, sub_freq=sub_freq, sub_level=sub_level,
                                         sub_decay=sub_decay, click_level=click_level, click_decay=click_decay,
                                         pitch_drop=pitch_drop, pitch_glide_time=pitch_glide_time, noise_level=noise_level,
                                         distortion=distortion_param, brightness_db=brightness_db, gain_db=gain_db)
                params = dict(sub_freq=sub_freq, sub_level=sub_level, sub_decay=sub_decay, click_level=click_level,
                              click_decay=click_decay, pitch_drop=pitch_drop, pitch_glide_time=pitch_glide_time,
                              noise_level=noise_level, distortion_param=distortion_param, brightness_db=brightness_db, gain_db=gain_db)
            elif inst == 'hat':
                noise_level = float(np.clip(np.random.normal(0.6, 0.2), 0.05, 1.5))
                body_level = float(np.clip(np.random.beta(2,5)*0.4, 0.0, 0.8))
                tone_freq = float(np.clip(np.random.normal(6000, 1500), 3000, 12000))
                decay = float(np.clip(np.random.normal(0.02, 0.01), 0.005, 0.15))
                brightness_db = float(np.clip(np.random.normal(1.0, 2.5), -4.0, 8.0))
                gain_db = float(np.clip(np.random.normal(-6.0, 3.0), -12.0, 3.0))
                y = synth_hat_numpy(sr=sr, duration=1.0, noise_level=noise_level, body_level=body_level, tone_freq=tone_freq,
                                    decay=decay, brightness_db=brightness_db, gain_db=gain_db)
                params = dict(noise_level=noise_level, body_level=body_level, tone_freq=tone_freq, decay=decay, brightness_db=brightness_db, gain_db=gain_db)
            elif inst == 'bass':
                freq = float(np.clip(np.random.normal(55, 10), 30, 120))
                amp = float(np.clip(np.random.beta(2,1.5), 0.4, 1.4))
                decay = float(np.clip(np.random.normal(0.5, 0.3), 0.05, 2.0))
                drive = float(np.clip(np.random.beta(1.5,3.0)*0.8, 0.0, 1.6))
                sub_octave = random.choice([True, False])
                y = synth_bass_numpy(sr=sr, duration=duration, freq=freq, amp=amp, decay=decay, drive=drive, sub_octave=sub_octave)
                params = dict(freq=freq, amp=amp, decay=decay, drive=drive, sub_octave=sub_octave)
            else:
                raise ValueError(f'unknown instrument {inst}')

            # compute features
            feats = compute_features(y, sr)

            # save
            fname = f"{inst}_{idx:05d}.wav"
            sf.write(str(out_dir / inst / fname), y.astype(np.float32), sr)

            # store row
            row = dict(file=fname, instrument=inst)
            row.update(params)
            row.update(feats)
            rows.append(row)
            idx += 1
            pbar.update(1)
            if idx >= n_total:
                break
    pbar.close()

    # build dataframe and normalized attributes
    df = pd.DataFrame(rows)
    # choose canonical features to normalize to create human-facing attributes
    # 'snappy' ~ snappiness, 'brightness' ~ centroid, 'distortion' ~ distortion_param or highband_ratio/flatness
    if 'snappiness' in df.columns:
        df['attr_snappy'] = minmax_scale(df['snappiness'].fillna(0).values)
    else:
        df['attr_snappy'] = 0.0
    if 'centroid' in df.columns:
        df['attr_brightness'] = minmax_scale(df['centroid'].fillna(0).values)
    else:
        df['attr_brightness'] = 0.0
    # distortion label: use distortion_param when present, else approximate with highband_ratio + flatness
    if 'distortion_param' in df.columns:
        df['attr_distortion'] = minmax_scale(df['distortion_param'].fillna(0).values)
    else:
        proxy = df.get('highband_ratio', np.zeros(len(df))) + df.get('flatness', np.zeros(len(df)))
        df['attr_distortion'] = minmax_scale(proxy)

    # additional normalizations for other attributes
    if 'hnr' in df.columns:
        df['attr_harmonicity'] = minmax_scale(df['hnr'].fillna(0).values)
    if 'decay_time_s' in df.columns:
        # long decay -> less snappy; invert it for 'snappy_decay' metric if desired
        df['attr_decay'] = minmax_scale(df['decay_time_s'].fillna(0).values)

    # write CSV
    csv_path = out_dir / 'metadata.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples and metadata to: {out_dir}")
    return out_dir, df

# ----------------------------- CLI ----------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='./dnb_samples', help='output folder')
    p.add_argument('--n_total', type=int, default=1000, help='total number of samples to generate')
    p.add_argument('--sr', type=int, default=44100)
    p.add_argument('--duration', type=float, default=1.0)
    p.add_argument('--instruments', nargs='+', default=['kick'], help='instruments to synth: kick hat bass')
    p.add_argument('--use_pyo', action='store_true', help='use pyo engine if available (higher quality)')
    p.add_argument('--no_pyo', action='store_true', help='force disable pyo fallback')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    use_pyo = args.use_pyo and HAS_PYO and (not args.no_pyo)
    if args.use_pyo and not HAS_PYO:
        print('pyo requested but not available; falling back to numpy synth. Install pyo for higher fidelity.')
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    generate_dataset(out_dir, n_total=args.n_total, instruments=args.instruments, sr=args.sr, duration=args.duration, use_pyo=use_pyo, seed=args.seed)

if __name__ == '__main__':
    main()
