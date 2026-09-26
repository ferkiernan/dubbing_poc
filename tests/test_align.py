import numpy as np
import soundfile as sf

from dubbing_engine.align import stretch_to_duration


def _write_tone(path, duration, sr=22050, freq=220.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t).astype("float32")
    sf.write(path, y, sr)


def test_stretch_shorter_to_longer(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_tone(str(in_wav), duration=1.0)

    final_dur = stretch_to_duration(str(in_wav), str(out_wav), target_duration=2.0)

    assert abs(final_dur - 2.0) < 0.02
    y, sr = sf.read(str(out_wav))
    assert abs(len(y) / sr - 2.0) < 0.02


def test_stretch_longer_to_shorter(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_tone(str(in_wav), duration=3.0)

    final_dur = stretch_to_duration(str(in_wav), str(out_wav), target_duration=1.2)

    assert abs(final_dur - 1.2) < 0.02


def test_extreme_ratio_falls_back_to_pad_trim(tmp_path):
    # Un target muchísimo más corto que el audio original fuerza el
    # clamp de rate y después el recorte final: no debe romper.
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_tone(str(in_wav), duration=5.0)

    final_dur = stretch_to_duration(str(in_wav), str(out_wav), target_duration=0.5)

    assert abs(final_dur - 0.5) < 0.02
