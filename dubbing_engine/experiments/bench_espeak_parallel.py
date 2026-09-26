"""Benchmark: síntesis espeak secuencial vs. paralela (pipeline.py).

Mide el tiempo del loop de síntesis por frase de pipeline.run() con
DUBBING_TTS_WORKERS=1 (secuencial) vs. el default paralelo, sobre un
video sintético con N frases fijas. No requiere red ni descargar
modelos (usa un ASR fake con segmentos fijos, igual que
tests/test_pipeline_parallel.py).

Uso:
    python -m dubbing_engine.experiments.bench_espeak_parallel [N_FRASES]
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

from dubbing_engine import pipeline
from dubbing_engine.asr import ASRBackend
from dubbing_engine.registry import register
from dubbing_engine.segments import Segment


def _make_segments(n: int):
    segments = []
    t = 0.2
    for i in range(n):
        dur = 1.0
        segments.append(Segment(start=t, end=t + dur, text=f"Esta es la frase número {i + 1} de la prueba."))
        t += dur + 0.1
    return segments


def _register_fixed_asr(segments):
    @register("asr", "bench_fixed")
    class _FixedASR(ASRBackend):
        def __init__(self, **kwargs):
            pass

        def transcribe(self, audio_path, language=None, on_progress=None):
            return list(segments)

    return _FixedASR


def _make_test_video(path, duration):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=160x120:r=10:d={duration}",
            "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:d={duration}",
            "-c:v", "libx264", "-c:a", "aac",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _run_once(video_path: str, workers_env: str) -> float:
    old = os.environ.get("DUBBING_TTS_WORKERS")
    os.environ["DUBBING_TTS_WORKERS"] = workers_env
    try:
        with tempfile.TemporaryDirectory(prefix="bench_espeak_") as tmpdir:
            output_path = os.path.join(tmpdir, "out.mp4")
            start = time.perf_counter()
            pipeline.run(
                video_path=video_path,
                output_path=output_path,
                voice="es-f1",
                language="es",
                tts_backend="espeak",
                asr_backend="bench_fixed",
            )
            return time.perf_counter() - start
    finally:
        if old is None:
            os.environ.pop("DUBBING_TTS_WORKERS", None)
        else:
            os.environ["DUBBING_TTS_WORKERS"] = old


def main():
    if shutil.which("ffmpeg") is None or shutil.which("espeak-ng") is None:
        print("Requiere ffmpeg y espeak-ng en PATH.")
        sys.exit(1)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    segments = _make_segments(n)
    _register_fixed_asr(segments)

    total_duration = segments[-1].end + 0.5
    with tempfile.TemporaryDirectory(prefix="bench_espeak_video_") as tmpdir:
        video_path = os.path.join(tmpdir, "input.mp4")
        _make_test_video(video_path, total_duration)

        print(f"Benchmark: {n} frases, video de {total_duration:.1f}s")
        print("Corriendo secuencial (DUBBING_TTS_WORKERS=1)...")
        t_seq = _run_once(video_path, "1")
        print(f"  -> {t_seq:.2f}s")

        print("Corriendo paralelo (default, sin override)...")
        t_par = _run_once(video_path, "0")
        print(f"  -> {t_par:.2f}s")

    speedup = t_seq / t_par if t_par > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
