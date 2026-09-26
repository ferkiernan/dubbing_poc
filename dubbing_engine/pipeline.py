"""Orquesta el flujo completo: extraer audio -> transcribir por frase ->
sintetizar cada frase con la voz elegida -> ajustar duración -> armar
la pista -> remultiplexar con el video."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from dubbing_engine import audio_io, align
from dubbing_engine.registry import get as get_backend
from dubbing_engine.segments import Segment, merge_sentence_segments

ProgressCallback = Callable[..., None]


def _notify(callback: Optional[ProgressCallback], stage: str, **data) -> None:
    if callback is not None:
        callback(stage, **data)


def _default_espeak_workers() -> int:
    override = int(os.environ.get("DUBBING_TTS_WORKERS", 0) or 0)
    if override > 0:
        return override
    return min(8, max(1, os.cpu_count() or 4))


def _synth_one_segment(
    tts, i: int, seg: Segment, language: str, voice: str, tmpdir: str,
    total: int, on_progress: Optional[ProgressCallback],
) -> Tuple[float, str]:
    _notify(
        on_progress, "synth_segment",
        index=i, total=total, text=seg.text,
        start=seg.start, end=seg.end,
    )
    raw_path = os.path.join(tmpdir, f"seg_{i:04d}_raw.wav")
    aligned_path = os.path.join(tmpdir, f"seg_{i:04d}_aligned.wav")

    tts.synth(seg.text, language, voice, raw_path)
    align.stretch_to_duration(raw_path, aligned_path, seg.duration)

    return (seg.start, aligned_path)


@dataclass
class DubResult:
    output_path: str
    segments: List[Segment]


def run(
    video_path: str,
    output_path: str,
    voice: str,
    language: str = "es",
    tts_backend: str = "espeak",
    asr_backend: str = "whisper",
    asr_model_size: str = "medium",
    device: str = "cpu",
    tts_kwargs: Optional[dict] = None,
    on_progress: Optional[ProgressCallback] = None,
    precomputed_segments: Optional[List[Segment]] = None,
    merge_sentences: bool = False,
) -> DubResult:
    """Si se pasa precomputed_segments (ej. leídos de un .json guardado
    en una corrida anterior), se salta extracción de audio y ASR por
    completo — útil para re-sintetizar el mismo video con otra voz sin
    pagar de nuevo el costo de transcripción.

    merge_sentences: fusiona segmentos consecutivos que Whisper cortó a
    mitad de una oración (por una pausa breve), para que el TTS reciba
    la frase entera en vez de pedazos. Ver segments.merge_sentence_segments.
    Se aplica también sobre precomputed_segments, para poder activarlo
    al reprocesar sin re-transcribir."""
    tts_kwargs = tts_kwargs or {}

    with tempfile.TemporaryDirectory(prefix="dubbing_engine_") as tmpdir:
        if precomputed_segments is not None:
            segments = precomputed_segments
            _notify(on_progress, "transcribed", count=len(segments), reused=True)
        else:
            _notify(on_progress, "extract_audio")
            extracted_wav = os.path.join(tmpdir, "extracted.wav")
            audio_io.extract_audio(video_path, extracted_wav)

            _notify(on_progress, "load_asr", backend=asr_backend, model_size=asr_model_size)
            asr = get_backend("asr", asr_backend, model_size=asr_model_size, device=device)

            _notify(on_progress, "transcribe")

            def _on_transcribe_progress(segment, pct):
                _notify(on_progress, "transcribe_progress", segment=segment, pct=pct)

            segments = asr.transcribe(
                extracted_wav, language=language, on_progress=_on_transcribe_progress
            )

            if not segments:
                raise RuntimeError(
                    "No se detectó habla en el video (o el idioma no coincide "
                    "con --lang). Probá con otro --lang o revisá el audio de entrada."
                )

            _notify(on_progress, "transcribed", count=len(segments), reused=False)

        if merge_sentences:
            before = len(segments)
            segments = merge_sentence_segments(segments)
            _notify(on_progress, "merged_sentences", before=before, after=len(segments))

        _notify(on_progress, "load_tts", backend=tts_backend)
        tts = get_backend("tts", tts_backend, device=device, **tts_kwargs)

        total_duration = audio_io.get_duration(video_path)

        if getattr(tts, "THREAD_SAFE", False):
            # Backend stateless (ej. espeak): cada frase es independiente
            # (build_timeline coloca cada wav por su start_time absoluto,
            # sin importar el orden en que se generaron), así que se
            # sintetizan en paralelo para aprovechar los cores de CPU.
            workers = _default_espeak_workers()
            results: List[Optional[Tuple[float, str]]] = [None] * len(segments)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(
                        _synth_one_segment, tts, i, seg, language, voice,
                        tmpdir, len(segments), on_progress,
                    ): i
                    for i, seg in enumerate(segments)
                }
                try:
                    for fut in as_completed(futures):
                        i = futures[fut]
                        results[i] = fut.result()
                except Exception as exc:
                    failed_index = futures[fut]
                    for f in futures:
                        f.cancel()
                    failed_seg = segments[failed_index]
                    raise RuntimeError(
                        f"Falló al sintetizar frase {failed_index}: {failed_seg.text!r}"
                    ) from exc
            segment_wavs = results
        else:
            # Backend con estado compartido (ej. XTTS: un único modelo
            # cargado en memoria) — se mantiene estrictamente secuencial.
            segment_wavs = []
            for i, seg in enumerate(segments):
                segment_wavs.append(
                    _synth_one_segment(
                        tts, i, seg, language, voice, tmpdir, len(segments), on_progress
                    )
                )

        _notify(on_progress, "build_timeline")
        timeline_wav = os.path.join(tmpdir, "timeline.wav")
        audio_io.build_timeline(segment_wavs, total_duration, timeline_wav)

        _notify(on_progress, "mux")
        audio_io.mux(video_path, timeline_wav, output_path)

        _notify(on_progress, "done")
        return DubResult(output_path=output_path, segments=segments)
