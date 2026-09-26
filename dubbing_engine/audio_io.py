"""Wrappers de ffmpeg para extraer audio, armar la pista final y
remultiplexar con el video original."""

import json
import subprocess
from typing import List, Tuple

import numpy as np
import soundfile as sf

DEFAULT_SR = 22050


def get_duration(media_path: str) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", media_path,
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    return float(data["format"]["duration"])


def extract_audio(video_path: str, out_wav: str, sr: int = DEFAULT_SR) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1",
            out_wav,
        ],
        check=True, capture_output=True,
    )


def build_timeline(
    segment_wavs: List[Tuple[float, str]],
    total_duration: float,
    out_wav: str,
    sr: int = DEFAULT_SR,
) -> None:
    """Arma una pista de `total_duration` segundos, colocando cada
    segment_wavs[i] = (start_time, wav_path) en su offset original."""

    total_samples = max(1, int(round(total_duration * sr)))
    timeline = np.zeros(total_samples, dtype="float32")

    for start_time, wav_path in segment_wavs:
        y, wav_sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if wav_sr != sr:
            import librosa

            y = librosa.resample(y, orig_sr=wav_sr, target_sr=sr)

        start_sample = int(round(start_time * sr))
        end_sample = start_sample + len(y)
        if start_sample >= total_samples:
            continue
        end_sample = min(end_sample, total_samples)
        usable = end_sample - start_sample
        if usable <= 0:
            continue
        # Suma en vez de sobreescribir, por si dos frases se pisan
        # levemente (poco común con VAD, pero no debe romper nada).
        timeline[start_sample:end_sample] += y[:usable]

    # Evita clipping si hubo solapamientos.
    peak = np.max(np.abs(timeline)) if timeline.size else 0.0
    if peak > 1.0:
        timeline = timeline / peak

    sf.write(out_wav, timeline, sr)


def mux(video_path: str, audio_wav: str, out_video: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            out_video,
        ],
        check=True, capture_output=True,
    )
