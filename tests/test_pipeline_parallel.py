"""Cubre el camino paralelo de síntesis (pipeline.py) que se agrega para
backends TTS marcados THREAD_SAFE (hoy, espeak): confirma que
pipeline.run() no pierde ni duplica frases aunque terminen desordenadas,
y que un backend no-thread-safe (como XTTS) sigue corriendo secuencial
en un único thread. No descarga el modelo Whisper real: se registra un
ASR fake que devuelve segmentos fijos, igual de espíritu que
test_pipeline_wiring.py.
"""

import shutil
import subprocess
import threading

import pytest

from dubbing_engine import pipeline
from dubbing_engine.asr import ASRBackend
from dubbing_engine.registry import register
from dubbing_engine.segments import Segment
from dubbing_engine.tts import TTSBackend

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("espeak-ng") is None,
    reason="requiere ffmpeg y espeak-ng instalados",
)

FIXED_SEGMENTS = [
    Segment(start=0.2, end=1.0, text="Frase número uno."),
    Segment(start=1.1, end=1.9, text="Frase número dos."),
    Segment(start=2.0, end=2.8, text="Frase número tres."),
    Segment(start=2.9, end=3.7, text="Frase número cuatro."),
    Segment(start=3.8, end=4.6, text="Frase número cinco."),
    Segment(start=4.7, end=5.5, text="Frase número seis."),
]


@register("asr", "fixed_fake")
class FixedFakeASR(ASRBackend):
    """Devuelve siempre FIXED_SEGMENTS, sin transcribir nada de verdad."""

    def __init__(self, **kwargs):
        pass

    def transcribe(self, audio_path, language=None, on_progress=None):
        return list(FIXED_SEGMENTS)


@register("tts", "recording_fake")
class RecordingFakeTTSBackend(TTSBackend):
    """No es THREAD_SAFE (default False, igual que XTTSBackend). Registra
    en qué thread ocurrió cada synth() para confirmar que el camino
    secuencial nunca usa más de un thread. El registro es a nivel de
    clase porque pipeline.run() crea la instancia internamente vía
    get_backend(), fuera del alcance directo del test."""

    thread_ids: list = []
    _lock = threading.Lock()

    def __init__(self, **kwargs):
        pass

    def list_voices(self):
        return ["fake"]

    def synth(self, text, language, voice, out_path):
        with RecordingFakeTTSBackend._lock:
            RecordingFakeTTSBackend.thread_ids.append(threading.get_ident())
        # Genera un wav real y corto vía espeak-ng, para que
        # align.stretch_to_duration tenga algo válido con qué trabajar.
        with open(out_path, "wb") as f:
            subprocess.run(
                ["espeak-ng", "-v", "es", "-s", "165", "--stdout"],
                input=text.encode("utf-8"),
                stdout=f,
                check=True,
            )


def _make_test_video(path, duration=6.0):
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


def test_parallel_espeak_preserves_all_segments(tmp_path):
    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _make_test_video(video, duration=6.0)

    events = []
    lock = threading.Lock()

    def on_progress(stage, **data):
        with lock:
            events.append((stage, data))

    result = pipeline.run(
        video_path=str(video),
        output_path=str(output),
        voice="es-f1",
        language="es",
        tts_backend="espeak",
        asr_backend="fixed_fake",
        on_progress=on_progress,
    )

    assert output.exists()

    synth_events = [d for stage, d in events if stage == "synth_segment"]
    seen_indexes = {d["index"] for d in synth_events}
    assert seen_indexes == set(range(len(FIXED_SEGMENTS)))
    assert len(synth_events) == len(FIXED_SEGMENTS)
    assert len(result.segments) == len(FIXED_SEGMENTS)


def test_non_thread_safe_backend_stays_sequential(tmp_path):
    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _make_test_video(video, duration=6.0)

    assert RecordingFakeTTSBackend.THREAD_SAFE is False
    RecordingFakeTTSBackend.thread_ids = []

    result = pipeline.run(
        video_path=str(video),
        output_path=str(output),
        voice="fake",
        language="es",
        tts_backend="recording_fake",
        asr_backend="fixed_fake",
    )

    assert output.exists()
    assert len(result.segments) == len(FIXED_SEGMENTS)
    assert len(RecordingFakeTTSBackend.thread_ids) == len(FIXED_SEGMENTS)
    assert len(set(RecordingFakeTTSBackend.thread_ids)) == 1, (
        "Un backend no thread-safe no debe recibir synth() desde más de un thread"
    )
