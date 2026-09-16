"""Tests de personalización de voz por backend (speed/pitch/volume en
espeak): verifica que los parámetros custom se apliquen al comando real
de espeak-ng, y que los defaults se preserven si no se pasa nada."""

import shutil
from unittest.mock import patch

import pytest

from dubbing_poc.tts import EspeakTTSBackend

pytestmark = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="requiere espeak-ng instalado"
)


def _fake_run_capturing(calls):
    def _run(cmd, input, stdout, check):  # noqa: A002 - firma de subprocess.run
        calls.append(cmd)
        stdout.write(b"")
    return _run


def test_espeak_uses_default_voice_params(tmp_path):
    tts = EspeakTTSBackend()
    calls = []
    with patch("dubbing_poc.tts.subprocess.run", side_effect=_fake_run_capturing(calls)):
        tts.synth("hola", "es", "es-f1", str(tmp_path / "out.wav"))

    cmd = calls[0]
    assert "-s" in cmd and cmd[cmd.index("-s") + 1] == "165"
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "50"
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "100"


def test_espeak_applies_custom_voice_params(tmp_path):
    tts = EspeakTTSBackend(speed=220, pitch=80, volume=150)
    calls = []
    with patch("dubbing_poc.tts.subprocess.run", side_effect=_fake_run_capturing(calls)):
        tts.synth("hola", "es", "es-f1", str(tmp_path / "out.wav"))

    cmd = calls[0]
    assert cmd[cmd.index("-s") + 1] == "220"
    assert cmd[cmd.index("-p") + 1] == "80"
    assert cmd[cmd.index("-a") + 1] == "150"
