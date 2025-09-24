Here’s a fully detailed `README.md` you can include with your TorchFX + TorchAudio Drum & Bass one-shot generator project:

---

# Drum & Bass One-Shot Generator (TorchFX + TorchAudio)

This project is a **GPU-accelerated Drum & Bass one-shot generator** built in Python using **TorchFX** for audio synthesis and **TorchAudio** for audio processing. It allows you to generate high-quality kicks, basses, and hi-hats with preset-driven parameters, and export features for machine learning workflows.

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

Ensure your system’s GPU drivers are up to date for GPU acceleration.

---

## Preset Configuration

Presets are stored in a JSON file, e.g., `dnb_presets.json`:

```json
{
  "kick": [
    {"sub_freq": 50, "sub_level": 0.9, "sub_decay": 0.15, "click_amp": 0.7, "click_tone_freq": 4000, "click_decay": 0.01, "distortion": 0.4}
  ],
  "bass": [
    {"freq": 55, "amp": 0.9, "decay": 0.6, "drive": 0.3, "sub_oct": true, "filter_cut": 200, "osc_type": "saw"}
  ],
  "hat": [
    {"noise_level": 0.7, "tone_freq": 7000, "body": 0.2, "decay": 0.02}
  ]
}
```

Each instrument can have multiple preset entries for variety.

* `kick`: sub frequency, level, decay, click amplitude/frequency, distortion.
* `bass`: oscillator frequency/type, amplitude, decay, drive, filter cutoff.
* `hat`: noise level, tone frequency, body, decay.

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

3. Optionally, you can modify the script to change `n_total`, `duration`, or `sr`.

---

## Output

* WAV files are stored by instrument:

```
dnb_samples/
    kick/
        kick_00001.wav
        kick_00002.wav
    bass/
        bass_00001.wav
        bass_00002.wav
    hat/
        hat_00001.wav
        hat_00002.wav
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
* Optionally, you can extend feature extraction using **Librosa** or **TorchAudio** transforms, e.g., spectral centroid, flatness, rolloff, MFCCs.

---

## Customizing Presets

* Add or modify presets in the JSON file for your desired sound characteristics.
* Parameters can be **slightly jittered** during batch generation for dataset diversity.
* Recommended workflow:

  1. Create a high-quality base preset.
  2. Allow small parameter variations (5–10%) to generate multiple samples.
  3. Combine multiple presets for style diversity (Neuro, Jungle, Jump-Up, Liquid, etc.).

---

## Notes

* GPU usage is optional but highly recommended for large batch synthesis.
* The system is fully offline; no real-time audio server is needed.
* You can extend this pipeline to generate **loops, sequences, or multi-bar arrangements** in the future.

---

Would you like me to also **write a section with example Python code for adding new instruments and presets**, so that your generator is fully extendable for future DnB sound design workflows?
