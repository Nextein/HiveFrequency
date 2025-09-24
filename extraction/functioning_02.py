import os
import glob
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

def detect_drop(y, sr):
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

def process_track_with_drop(file_path, target_bpm=174, sr=22050, n_mels=128, save_audio=True, output_dir="output_audio"):
    """
    Processes a track by detecting the drop and extracting a 32-bar segment around it.
    Saves the extracted segment as a WAV file for verification.
    
    Parameters:
    - file_path: Path to the audio file
    - target_bpm: Desired BPM for normalization (default: 174)
    - sr: Sampling rate for processing
    - n_mels: Number of mel frequency bins for spectrogram
    - save_audio: Whether to save the extracted audio clip
    - output_dir: Directory where extracted audio files will be stored
    
    Returns:
    - S_db: Mel spectrogram of the extracted segment (or None if an issue occurs)
    """
    # Load the audio file
    y, orig_sr = librosa.load(file_path, sr=None)
    
    # Resample if needed
    if orig_sr != sr:
        print(f"{orig_sr} != {sr} -> Resampling audio to {sr}Hz ...")
        y = librosa.resample(y=y, orig_sr=orig_sr, sr=sr)
    
    # Estimate BPM and normalize
    estimated_bpm = librosa.beat.tempo(y=y, sr=sr)[0]
    if estimated_bpm < 90:
        estimated_bpm *= 2
        estimated_bpm = int(estimated_bpm)
    rate_factor = target_bpm / estimated_bpm
    y = librosa.effects.time_stretch(y=y, rate=rate_factor)
    
    # Detect drop
    drop_frame, beat_frames = detect_drop(y, sr)
    
    # Ensure we have enough beats (128 surrounding the drop)
    required_beats = 32 * 4  # 128 beats for 32 bars
    drop_index = np.where(beat_frames == drop_frame)[0][0]  # Index of the drop in beat_frames
    
    # Extract beats before and after drop
    start_idx = max(0, drop_index - required_beats // 2)
    end_idx = min(len(beat_frames) - 1, drop_index + required_beats // 2)
    
    # Convert beat frames to sample indices
    start_sample = librosa.frames_to_samples(beat_frames[start_idx])
    end_sample = librosa.frames_to_samples(beat_frames[end_idx])
    
    y_segment = y[start_sample:end_sample]

    # Save extracted audio clip for verification
    if save_audio:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_filename = os.path.join(output_dir, os.path.basename(file_path).replace(".mp3", "").replace(".wav", "") + "_drop_clip.wav")
        sf.write(output_filename, y_segment, sr)
        print(f"Saved segmented audio: {output_filename}")
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    return S_db

def process_folder_with_drop(folder_path, target_bpm=174, sr=22050, n_mels=128, save_audio=True, output_dir="output_audio"):
    """
    Processes an entire folder of tracks, extracting 32-bar spectrograms around the drop.
    Saves extracted audio clips for verification.
    
    Parameters:
    - folder_path: Path to the folder containing drum and bass tracks
    - target_bpm: Desired BPM normalization
    - sr: Sampling rate
    - n_mels: Number of mel bins for spectrogram
    - save_audio: Whether to save extracted audio clips
    - output_dir: Directory where extracted audio files will be stored
    
    Returns:
    - List of spectrogram matrices
    """
    audio_files = glob.glob(os.path.join(folder_path, '*.mp3')) + \
                  glob.glob(os.path.join(folder_path, '*.wav'))
    
    dataset = []
    
    for file_path in audio_files:
        print(f"Processing {file_path}...")
        S_db = process_track_with_drop(file_path, target_bpm, sr, n_mels, save_audio, output_dir)
        if S_db is not None:
            dataset.append(S_db)
    
    return dataset

# Example usage:
folder = "data/raw_audio"
clip_folder = "data/raw_clips"
spectrogram_dataset = process_folder_with_drop(folder_path=folder, sr=44100, output_dir=clip_folder)
