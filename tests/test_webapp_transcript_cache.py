"""Test del auto-cache de transcripción: si ya existe un .dubbing_poc.json
para el mismo nombre de archivo + idioma + modelo ASR, _find_cached_transcript
debe encontrarlo, para que _process_one no vuelva a transcribir."""

import json

from dubbing_poc.webapp.app import TRANSCRIPT_EXTENSION, _find_cached_transcript


def _write_transcript(dir_path, filename, original_filename, language, asr_model_size):
    path = dir_path / filename
    data = {
        "source_video": str(dir_path / "video.mp4"),
        "original_filename": original_filename,
        "language": language,
        "asr_backend": "whisper",
        "asr_model_size": asr_model_size,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_finds_cached_transcript_by_filename_and_config(tmp_path):
    _write_transcript(tmp_path, f"job1_out{TRANSCRIPT_EXTENSION}", "mi_video.mp4", "es", "medium")

    found = _find_cached_transcript("mi_video.mp4", "es", "medium", str(tmp_path))

    assert found is not None
    assert found["original_filename"] == "mi_video.mp4"
    assert len(found["segments"]) == 1


def test_does_not_match_different_language(tmp_path):
    _write_transcript(tmp_path, f"job1_out{TRANSCRIPT_EXTENSION}", "mi_video.mp4", "es", "medium")

    found = _find_cached_transcript("mi_video.mp4", "en", "medium", str(tmp_path))

    assert found is None


def test_does_not_match_different_asr_model_size(tmp_path):
    _write_transcript(tmp_path, f"job1_out{TRANSCRIPT_EXTENSION}", "mi_video.mp4", "es", "medium")

    found = _find_cached_transcript("mi_video.mp4", "es", "large-v3", str(tmp_path))

    assert found is None


def test_does_not_match_different_filename(tmp_path):
    _write_transcript(tmp_path, f"job1_out{TRANSCRIPT_EXTENSION}", "mi_video.mp4", "es", "medium")

    found = _find_cached_transcript("otro_video.mp4", "es", "medium", str(tmp_path))

    assert found is None


def test_prefers_most_recent_when_multiple_match(tmp_path):
    import os
    import time

    _write_transcript(tmp_path, f"old{TRANSCRIPT_EXTENSION}", "mi_video.mp4", "es", "medium")
    time.sleep(0.01)
    newer = _write_transcript(tmp_path, f"new{TRANSCRIPT_EXTENSION}", "mi_video.mp4", "es", "medium")
    # Asegura mtime distinguible en filesystems con resolución gruesa.
    now = time.time()
    os.utime(newer, (now + 10, now + 10))

    found = _find_cached_transcript("mi_video.mp4", "es", "medium", str(tmp_path))

    assert found is not None


def test_no_transcripts_returns_none(tmp_path):
    found = _find_cached_transcript("mi_video.mp4", "es", "medium", str(tmp_path))
    assert found is None
