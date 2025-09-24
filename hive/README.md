# Drum & Bass One-Shot Generator (TorchFX + TorchAudio)

This project is a **GPU-accelerated Drum & Bass one-shot generator** built in Python using **TorchFX** for audio synthesis and **TorchAudio** for audio processing. It allows you to generate high-quality kicks, basses, hi-hats, and more with preset-driven parameters, and export features for machine learning workflows.

---

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Preset Configuration](#preset-configuration)
* [Running the Generator](#running-the-generator)
* [Output](#output)
* [Feature Extraction](#feature-extraction)
* [Customizing Presets](#customizing-presets)
* [Adding New Instruments](#adding-new-instruments)

---

## Features

* High-quality **kick, bass, and hi-hat synthesis** with GPU acceleration.
* Preset-driven sound design for quick and consistent results.
* Batch generation of large datasets.
* Metadata CSV export with parameters and basic audio features (RMS).
* Easily extendable with new instruments or preset variations.

---

## Requirements

* Python **3.11**
* GPU recommended for TorchFX synthesis (CUDA-enabled)
* Python libraries:

  * `torch` (PyTorch 2.1.0 recommended)
  * `torchaudio` (2.1.0 recommended)
  * `torchfx`
  * `numpy`
  * `pandas`
  * `tqdm`

---

## Installation

```bash
pip install torch==2.1.0 torchaudio==2.1.0 torchfx numpy pandas tqdm
```

---

## Preset Configuration

Presets are stored in a JSON file, e.g., `dnb_presets.json`:

```json
{
  "kick": [
    {"sub_freq":48,"sub_level":0.95,"sub_decay":0.15,"click_amp":0.7,"click_tone_freq":4000,"click_decay":0.012,"distortion":0.4,"brightness":0.3,"style":"neuro"}
  ],
  "bass": [
    {"freq":55,"amp":0.9,"decay":0.6,"drive":0.3,"sub_oct":true,"filter_cut":200,"osc_type":"saw","style":"jumpup"}
  ],
  "hat": [
    {"noise_level":0.7,"tone_freq":7000,"body":0.2,"decay":0.02,"brightness":0.2,"style":"liquid"}
  ]
}
```

Each instrument can have multiple preset entries for variety. The `"style"` field allows you to filter presets by DnB subgenre.

---

## Running the Generator

1. Place the generator script `dnb_torchfx_generator.py` and your `dnb_presets.json` in the same directory.
2. Run the generator:

```bash
python dnb_torchfx_generator.py
```

* Default parameters:

  * Output directory: `./dnb_samples`
  * Total samples: 1000 (can be modified in the script)
  * Sample rate: 44100 Hz
  * Duration: 1.0 second per one-shot

---

## Output

* WAV files are stored by instrument:

```
dnb_samples/
    kick/
        kick_00001.wav
    bass/
        bass_00001.wav
    hat/
        hat_00001.wav
```

* Metadata CSV (`metadata.csv`) contains:

  * File path
  * Instrument type
  * Preset parameters
  * Extracted audio features (RMS, etc.)

---

## Feature Extraction

The generator currently extracts:

* RMS (Root Mean Square) energy
* You can extend feature extraction with **Librosa** or **TorchAudio** (e.g., spectral centroid, flatness, rolloff, MFCCs).

---

## Customizing Presets

* Modify or add presets in the JSON file for desired sound characteristics.
* Allow small parameter variations (5–10%) for dataset diversity.
* Filter generation by `"style"` to produce one-shots for specific subgenres (Neuro, Jungle, Jump-Up, Liquid, Techstep).

---

## Adding New Instruments

You can add new instruments (e.g., **snare**, **clap**, **ride**) to expand your generator. Steps:

1. **Add presets to JSON**:

```json
"snare": [
  {"noise_level":0.8,"tone_freq":1800,"body":0.3,"decay":0.08,"distortion":0.2,"style":"neuro"},
  {"noise_level":0.7,"tone_freq":1500,"body":0.25,"decay":0.1,"distortion":0.1,"style":"jungle"}
]
```

2. **Implement TorchFX generator function**:

```python
def build_snare_fx(params):
    import torchfx as fx
    chain = (
        fx.Noise(level=params['noise_level'])
        | fx.filter.BandPass(freq=params['tone_freq'], q=5)
        | fx.envelope.ADSR(attack=0.005, decay=params['decay'], sustain=0.0, release=0.01)
    )
    if params.get('distortion', 0) > 0:
        chain = chain | fx.effect.Overdrive(drive=params['distortion'])
    return chain
```

3. **Integrate into main loop**:

```python
elif inst == 'snare':
    fx_chain = build_snare_fx(params)
```

4. **Optional**: add `"style"` field to filter snare presets by subgenre.

---

### Tips for Advanced Presets

* **Kick**: fast decay for Neuro/Jump-Up, slower for Jungle/Liquid. Adjust click frequency and distortion for aggression.
* **Bass**: detuned saws for Reese (Neuro), sub-heavy sine for Liquid/Techstep.
* **Hats**: short decay + bright for Neuro, longer + airy for Jungle/Liquid.
* **Snares/Claps**: combine noise + tone; subtle distortion for aggressive styles.

---

This ensures your generator is **fully extendable** for multiple DnB subgenres and instruments.

