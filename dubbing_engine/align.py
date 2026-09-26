"""Ajuste de duración: hace que el audio sintetizado dure lo mismo que
la frase original, para que coincida en tiempo con el video.

Estrategia: time-stretch (cambia velocidad sin cambiar tono) acotado a
un rango razonable; si hace falta estirar/comprimir más que eso, se
recorta o rellena con silencio para no distorsionar demasiado la voz.

El estiramiento usa `pedalboard.time_stretch` (motor Rubber Band, el
mismo que usa Spotify): con voz suena mucho más nítido que el phase
vocoder de librosa, sobre todo en estiramientos grandes (>1.3x), donde
librosa produce un artefacto "metálico"/lejano notorio.
"""

import numpy as np
import soundfile as sf

MIN_RATE = 0.6   # no estirar más de ~1.7x
MAX_RATE = 1.7   # no comprimir más de ~1.7x


def stretch_to_duration(in_wav: str, out_wav: str, target_duration: float) -> float:
    """Lee in_wav, lo ajusta a target_duration segundos, y lo guarda en
    out_wav. Devuelve la duración final real (debería ser == target_duration,
    salvo casos límite con target_duration <= 0)."""

    y, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    current_duration = len(y) / sr if sr else 0.0

    if target_duration <= 0.01 or current_duration <= 0.01:
        # Frase original sin duración útil: no hay a qué ajustar.
        sf.write(out_wav, y, sr)
        return current_duration

    rate = current_duration / target_duration
    rate_clamped = min(max(rate, MIN_RATE), MAX_RATE)

    if abs(rate_clamped - 1.0) > 1e-3:
        import pedalboard

        # pedalboard espera/devuelve forma (canales, samples).
        stretched = pedalboard.time_stretch(
            y.reshape(1, -1), sr, stretch_factor=rate_clamped
        )
        y = stretched.reshape(-1)

    target_samples = int(round(target_duration * sr))
    if len(y) < target_samples:
        pad = np.zeros(target_samples - len(y), dtype=y.dtype)
        y = np.concatenate([y, pad])
    elif len(y) > target_samples:
        y = y[:target_samples]

    sf.write(out_wav, y, sr)
    return len(y) / sr
