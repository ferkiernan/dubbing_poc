"""Tipos de datos compartidos entre módulos."""

import re
from dataclasses import dataclass
from typing import List

_TERMINAL_PUNCT_RE = re.compile(r'[.!?…]["\')\]]*$')


@dataclass
class Segment:
    """Una frase/segmento de habla con su ventana de tiempo original."""

    start: float  # segundos
    end: float    # segundos
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _ends_sentence(text: str) -> bool:
    return bool(_TERMINAL_PUNCT_RE.search(text.strip()))


def merge_sentence_segments(
    segments: List[Segment],
    max_gap: float = 0.6,
    max_merged_duration: float = 12.0,
) -> List[Segment]:
    """Fusiona segmentos consecutivos que forman una sola oración cortada
    por una pausa breve (Whisper corta por VAD/respiración, no por
    puntuación). Heurística basada en reglas, sin modelo semántico:

    - Solo fusiona el segmento actual con el siguiente si el texto
      acumulado NO termina en puntuación terminal (. ! ? …), es decir,
      si "suena" a que la frase sigue.
    - Solo si el silencio entre ambos es corto (<= max_gap segundos):
      un gap largo suele ser un cambio de tema/hablante, no una pausa
      dentro de la misma frase.
    - Solo si la duración fusionada no supera max_merged_duration: un
      segmento demasiado largo fuerza a align.py a estirar/comprimir
      agresivamente (o a recortar/rellenar con silencio) para encajar
      en el tiempo original, sonando peor que dejarlo cortado.

    Si no hay match de puntuación al final de todo el audio, el último
    segmento se conserva tal cual (no hay nada más con qué fusionarlo).
    """
    if not segments:
        return []

    merged: List[Segment] = []
    buf = segments[0]

    for nxt in segments[1:]:
        gap = nxt.start - buf.end
        merged_duration = nxt.end - buf.start
        if (
            not _ends_sentence(buf.text)
            and gap <= max_gap
            and merged_duration <= max_merged_duration
        ):
            buf = Segment(
                start=buf.start,
                end=nxt.end,
                text=f"{buf.text.rstrip()} {nxt.text.lstrip()}",
            )
        else:
            merged.append(buf)
            buf = nxt

    merged.append(buf)
    return merged
