"""Test de integración de las partes que NO dependen de descargar un
modelo (TTS espeak real + align + timeline + mux). El ASR (whisper) se
reemplaza por segmentos fijos, porque bajar el modelo requiere red y
en algunos entornos (CI, sandboxes) el acceso a huggingface.co puede
estar bloqueado. En una máquina normal, `pipeline.run(...)` hace todo
esto más la transcripción real.
"""

import shutil
import subprocess

import pytest

from dubbing_poc import audio_io, align
from dubbing_poc.registry import get as get_backend
from dubbing_poc.segments import Segment

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("espeak-ng") is None,
    reason="requiere ffmpeg y espeak-ng instalados",
)


def _make_test_video(path, duration=4.0):
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


def test_full_wiring_without_asr(tmp_path):
    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _make_test_video(video, duration=4.0)

    segments = [
        Segment(start=0.2, end=1.5, text="Hola, esto es una prueba."),
        Segment(start=1.7, end=3.6, text="Y esta es la segunda frase."),
    ]

    total_duration = audio_io.get_duration(str(video))
    tts = get_backend("tts", "espeak", device="cpu")

    seg_wavs = []
    for i, seg in enumerate(segments):
        raw = tmp_path / f"raw_{i}.wav"
        aligned = tmp_path / f"aligned_{i}.wav"
        tts.synth(seg.text, "es", "es-f1", str(raw))
        final_dur = align.stretch_to_duration(str(raw), str(aligned), seg.duration)
        assert abs(final_dur - seg.duration) < 0.05
        seg_wavs.append((seg.start, str(aligned)))

    timeline_wav = tmp_path / "timeline.wav"
    audio_io.build_timeline(seg_wavs, total_duration, str(timeline_wav))
    audio_io.mux(str(video), str(timeline_wav), str(output))

    assert output.exists()
    out_duration = audio_io.get_duration(str(output))
    assert abs(out_duration - total_duration) < 0.1

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    stream_types = probe.stdout.split()
    assert "video" in stream_types
    assert "audio" in stream_types
