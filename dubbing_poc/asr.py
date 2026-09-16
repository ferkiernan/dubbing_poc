"""Transcripción con marcas de tiempo por frase (módulo intercambiable)."""

import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from dubbing_poc.registry import register
from dubbing_poc.segments import Segment

TranscribeProgressCallback = Callable[..., None]


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        on_progress: Optional[TranscribeProgressCallback] = None,
    ) -> List[Segment]:
        """Devuelve una lista de Segment (start, end, text) en orden."""
        raise NotImplementedError


@register("asr", "whisper")
class FasterWhisperASR(ASRBackend):
    """ASR local con faster-whisper (corre 100% en CPU o GPU, sin internet
    salvo la primera vez que descarga el modelo)."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: Optional[int] = None,
        num_workers: int = 1,
    ):
        from faster_whisper import WhisperModel  # import diferido: pesado

        # cpu_threads: paralelismo intra-op para UNA transcripción (ASR y
        # TTS no se solapan en el tiempo, así que puede tomar varios
        # cores sin competir con el resto del pipeline). num_workers
        # controla transcribe() *concurrentes*, que hoy nunca ocurren
        # (ni siquiera en modo batch, que procesa videos uno por uno) —
        # se deja en 1 a propósito; subirlo sin uso concurrente real solo
        # reservaría threads de más y podría competir con cpu_threads.
        resolved_threads = cpu_threads if cpu_threads is not None else min(16, os.cpu_count() or 4)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=resolved_threads,
            num_workers=num_workers,
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        on_progress: Optional[TranscribeProgressCallback] = None,
    ) -> List[Segment]:
        raw_segments, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
        )
        duration = getattr(info, "duration", 0) or 0
        result = []
        for s in raw_segments:
            text = s.text.strip()
            if text:
                result.append(Segment(start=s.start, end=s.end, text=text))
                if on_progress is not None:
                    pct = min(100, int(100 * s.end / duration)) if duration else 0
                    on_progress(segment=result[-1], pct=pct)
        return result
