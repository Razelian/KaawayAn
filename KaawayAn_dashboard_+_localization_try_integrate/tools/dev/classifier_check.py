"""Run the DroneClassifier over WAVs in staging/dumps and staging/false,

and print statistical distributions of probabilities and RMS dBFS levels.
Uses batching for fast CPU inference.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_wav_16k(path: Path) -> np.ndarray:
    """Read a mono or stereo 16-bit WAV file and return float32 samples (16 kHz mono)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif sampwidth == 4:
        samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 65536.0
    else:
        samples = np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]

    if len(samples) > 16000:
        samples = samples[:16000]
    elif len(samples) < 16000:
        samples = np.pad(samples, (0, 16000 - len(samples)))

    return samples


def compute_rms_dbfs(samples: np.ndarray) -> float:
    norm = samples / 32768.0
    rms = float(np.sqrt(np.mean(norm ** 2)))
    if rms > 1e-12:
        return float(20.0 * np.log10(rms))
    return -120.0


def evaluate_folder(name: str, folder_path: Path, model, batch_size: int = 32) -> dict:
    wav_files = sorted(folder_path.glob("*.wav"))
    if not wav_files:
        print(f"[{name}] No .wav files found in {folder_path}", flush=True)
        return {}

    probs: list[float] = []
    rms_levels: list[float] = []

    print(f"[{name}] Evaluating {len(wav_files)} files...", flush=True)

    for i in range(0, len(wav_files), batch_size):
        batch_files = wav_files[i : i + batch_size]
        batch_samples = [read_wav_16k(p) for p in batch_files]

        for s in batch_samples:
            rms_levels.append(compute_rms_dbfs(s))

        # Batch tensor inference
        arr = np.array(batch_samples, dtype=np.float32) / 32768.0
        waveforms = torch.from_numpy(arr)  # (B, N)
        with torch.no_grad():
            log_mels = (T.AmplitudeToDB()(model.mel_transform(waveforms)) + 40) / 40
            x = log_mels.unsqueeze(1)  # (B, 1, 64, T)
            logits = model(x)
            batch_probs = torch.sigmoid(logits).squeeze(-1).tolist()
            if isinstance(batch_probs, float):
                batch_probs = [batch_probs]
            probs.extend(batch_probs)

    probs_arr = np.array(probs)
    rms_arr = np.array(rms_levels)

    print(f"\n=======================================================", flush=True)
    print(f"Results for: {name} ({len(wav_files)} files)", flush=True)
    print(f"Folder: {folder_path}", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Probability Distribution:", flush=True)
    print(f"  Min:    {np.min(probs_arr):.4f}", flush=True)
    print(f"  25%:    {np.percentile(probs_arr, 25):.4f}", flush=True)
    print(f"  Median: {np.median(probs_arr):.4f}", flush=True)
    print(f"  Mean:   {np.mean(probs_arr):.4f}", flush=True)
    print(f"  75%:    {np.percentile(probs_arr, 75):.4f}", flush=True)
    print(f"  Max:    {np.max(probs_arr):.4f}", flush=True)
    print(f"  Prob >= 0.60 (Detection): {np.sum(probs_arr >= 0.60)} / {len(probs_arr)} ({np.mean(probs_arr >= 0.60)*100:.1f}%)", flush=True)
    print(f"  Prob >= 0.85 (High Conf): {np.sum(probs_arr >= 0.85)} / {len(probs_arr)} ({np.mean(probs_arr >= 0.85)*100:.1f}%)", flush=True)
    print(f"RMS Levels (dBFS):", flush=True)
    print(f"  Min:    {np.min(rms_arr):.2f} dBFS", flush=True)
    print(f"  25%:    {np.percentile(rms_arr, 25):.2f} dBFS", flush=True)
    print(f"  Median: {np.median(rms_arr):.2f} dBFS", flush=True)
    print(f"  Mean:   {np.mean(rms_arr):.2f} dBFS", flush=True)
    print(f"  75%:    {np.percentile(rms_arr, 75):.2f} dBFS", flush=True)
    print(f"  Max:    {np.max(rms_arr):.2f} dBFS", flush=True)

    return {
        "count": len(wav_files),
        "prob_median": float(np.median(probs_arr)),
        "prob_mean": float(np.mean(probs_arr)),
        "prob_min": float(np.min(probs_arr)),
        "prob_max": float(np.max(probs_arr)),
        "rms_median": float(np.median(rms_arr)),
        "rms_mean": float(np.mean(rms_arr)),
    }


def main():
    try:
        from model.model import DroneClassifier
        model = DroneClassifier()
        print("[classifier_check] DroneClassifier loaded successfully.", flush=True)
    except Exception as exc:
        print(f"[classifier_check] Failed to load DroneClassifier: {exc}", flush=True)
        return 1

    dumps_dir = ROOT / "staging" / "dumps"
    false_dir = ROOT / "staging" / "false"

    res_dumps = evaluate_folder("STAGING / DUMPS (Drone / Flight)", dumps_dir, model)
    res_false = evaluate_folder("STAGING / FALSE (Ambient / False Alarms)", false_dir, model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
