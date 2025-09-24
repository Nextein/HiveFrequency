import os
import glob
import numpy as np
import librosa
import soundfile as sf
from pandas import DataFrame
import scipy
import pyrubberband as pyrb
import resampy


def detect_drop(y, sr):
    """
    Detects the drop in a track using spectral flux, RMS energy, and low-frequency emphasis.
    Returns the drop frame and beat frames.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    rms = librosa.feature.rms(y=y)[0]
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    flux_diff = np.diff(onset_env)
    threshold = np.percentile(flux_diff, 90)
    
    peaks = librosa.util.peak_pick(flux_diff, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=threshold, wait=5)
    
    drop_candidate = peaks[0] if len(peaks) > 0 else np.argmax(flux_diff)
    drop_frame = min(beat_frames, key=lambda x: abs(x - drop_candidate))
    
    return drop_frame, beat_frames, tempo

def detect_drop_pro(y, sr):
    """
    Detects the drop in a drum and bass track using a combination of:
    - Spectral flux (onset strength)
    - RMS energy
    - Low-frequency energy emphasis
    - Pre-drop breakdown detection
    
    Parameters:
    - y: Audio time series
    - sr: Sampling rate
    
    Returns:
    - drop_frame: The frame index of the detected drop
    - beat_frames: The list of detected beat frames (to align the drop with beats)
    """
    # Compute spectral flux (onset strength)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Extract low-frequency energy (focus on sub-bass frequencies)
    S = librosa.stft(y)
    S_magnitude = np.abs(S)
    low_freq_energy = np.sum(S_magnitude[:10, :], axis=0)  # Sum first 10 frequency bins (low freqs)

    # Normalize features
    rms /= np.max(rms)
    onset_env /= np.max(onset_env)
    low_freq_energy /= np.max(low_freq_energy)

    # Compute beats and tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Identify strong onset peaks (potential drop points)
    flux_diff = np.diff(onset_env)
    
    # Use dynamic thresholding to avoid false peaks
    threshold = np.percentile(flux_diff, 90)  # Select top 10% strongest changes
    peaks, _ = scipy.signal.find_peaks(flux_diff, height=threshold, distance=5)

    # Identify breakdown (sudden drop in RMS before the peak)
    if len(peaks) > 0:
        for peak in peaks:
            if peak > 10:  # Avoid first few frames
                previous_energy = np.mean(rms[max(0, peak-10):peak])  # Avg energy before peak
                if previous_energy < 0.3:  # A breakdown likely occurred
                    drop_candidate = peak
                    break
        else:
            drop_candidate = peaks[0]  # Default to first peak if no breakdown detected
    else:
        drop_candidate = np.argmax(flux_diff)  # Fallback: strongest onset change

    # Align the drop to the nearest beat
    drop_frame = min(beat_frames, key=lambda x: abs(x - drop_candidate))
    
    return drop_frame, beat_frames

def extract_spectrogram(y_segment, sr, feature_type="mel", n_mels=128, n_mfcc=20):
    """
    Extracts the spectrogram (mel, MFCC, or chroma).
    """
    if feature_type == "mel":
        S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=n_mels)
    elif feature_type == "mfcc":
        S = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc)
    elif feature_type == "chroma":
        S = librosa.feature.chroma_stft(y=y_segment, sr=sr)
    else:
        raise ValueError("Invalid feature_type. Choose 'mel', 'mfcc', or 'chroma'.")

    return librosa.power_to_db(S, ref=np.max)

def estimate_bpm(y, sr, target_bpm=174):
    """
    Estimates BPM more accurately by considering multiple BPM candidates
    and choosing the one closest to the target BPM.
    """
    # Compute BPM with different estimation methods
    bpm_candidates = librosa.feature.tempo(y=y, sr=sr, aggregate=None)

    # Pick the BPM closest to the expected range (e.g., 165-185 BPM for DnB)
    bpm_candidates = bpm_candidates[(bpm_candidates > 150) & (bpm_candidates < 190)]
    
    if len(bpm_candidates) == 0:
        bpm_candidates = librosa.beat.tempo(y=y, sr=sr)  # Fallback to default estimation
    
    estimated_bpm = min(bpm_candidates, key=lambda x: abs(x - target_bpm))
    
    return estimated_bpm

def correct_bpm(estimated_bpm, target_bpm=174):
    """
    Corrects the estimated BPM if it's detected as half or double.
    """
    if estimated_bpm < target_bpm / 1.5:  # If BPM is less than 116, likely half
        return estimated_bpm * 2
    elif estimated_bpm > target_bpm * 1.5:  # If BPM is greater than 261, likely double
        return estimated_bpm / 2
    return estimated_bpm

def process_track_with_drop(file_path, target_bpm=174, sr=22050, n_mels=128, save_audio=True, feature_type="mel", output_dir="output_audio"):
    """
    Processes a track, extracts the drop region, saves metadata, and validates output.
    """
    y, orig_sr = librosa.load(file_path, sr=None)
    # y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr) if orig_sr != sr else y
    y = resampy.resample(y, orig_sr, sr, filter='kaiser_best') if orig_sr != sr else y


    estimated_bpm = estimate_bpm(y, sr, target_bpm)
    estimated_bpm = correct_bpm(estimated_bpm, target_bpm)

    # y = librosa.effects.time_stretch(y=y, rate=target_bpm / estimated_bpm)
    y = pyrb.time_stretch(y, sr, target_bpm / estimated_bpm)

    # drop_frame, beat_frames = detect_drop_pro(y, sr)
    drop_frame, beat_frames, tempo = detect_drop(y, sr)

    required_beats = 4 * 32
    drop_index = np.where(beat_frames == drop_frame)[0][0]
    start_idx = max(0, drop_index - required_beats // 2)
    end_idx = min(len(beat_frames) - 1, drop_index + required_beats // 2)

    start_sample = librosa.frames_to_samples(beat_frames[start_idx])
    end_sample = librosa.frames_to_samples(beat_frames[end_idx])
    y_segment = y[start_sample:end_sample]

    if save_audio:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, os.path.basename(file_path).replace(".mp3", "").replace(".wav", "") + "_drop_segment.wav")
        sf.write(output_filename, y_segment, sr)
        print(f"Saved segmented audio: {output_filename}")

    S_db = extract_spectrogram(y_segment, sr, feature_type, n_mels)

    # Validate output dimensions
    if S_db.shape[1] < 50:  # Arbitrary lower bound for too-short clips
        print(f"Warning: {file_path} may have been cut incorrectly (spectrogram too short).")

    # Log metadata
    metadata = {
        "file": file_path,
        "drop_frame": drop_frame,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "tempo": tempo,
        "spectrogram_shape": S_db.shape
    }

    return S_db, metadata

def process_folder_with_drop(folder_path, target_bpm=174, sr=22050, n_mels=128, save_audio=True, feature_type="mel", output_dir="output_audio"):
    """
    Processes an entire folder, logs metadata, and saves extracted segments.
    """
    audio_files = glob.glob(os.path.join(folder_path, '*.mp3')) + glob.glob(os.path.join(folder_path, '*.wav'))
    dataset, metadata_list = [], []

    os.makedirs(output_dir, exist_ok=True)

    for file_path in audio_files:
        print(f"Processing {file_path}...")
        S_db, metadata = process_track_with_drop(file_path, target_bpm, sr, n_mels, save_audio, feature_type, output_dir)
        dataset.append(S_db)
        metadata_list.append(metadata)

    # Save metadata to CSV
    df_metadata = DataFrame(metadata_list)
    df_metadata.to_csv(METADATA_FILE, index=False)
    print(f"Metadata saved to {METADATA_FILE}")

    return dataset

# Example usage
folder = "data/raw_audio"
output_folder="data/raw_clips"
METADATA_FILE = "data/raw_clips/metadata.csv"
SPECTROGRAM_DATASET_FILE = "data/spectrograms/spectrogram_dataset.npz"

spectrogram_dataset = process_folder_with_drop(folder, feature_type="mel", output_dir=output_folder)

# Save the dataset as a compressed NumPy array
np.savez_compressed(SPECTROGRAM_DATASET_FILE, *spectrogram_dataset)
print(f"Spectrogram dataset saved to {SPECTROGRAM_DATASET_FILE}")