"""Síntesis de voz (módulo intercambiable).

Backends incluidos:
- "espeak" (default): usa espeak-ng por línea de comandos. Liviano,
  sin descargas de modelos, calidad robótica. Sirve para probar todo
  el pipeline (timing, corte de video, etc.) de entrada, en cualquier
  máquina, sin GPU.
- "xtts": usa Coqui XTTS-v2 (multilingüe, ~58 voces incorporadas,
  varias femeninas, y permite clonar una voz a partir de un wav de
  referencia). Mucho mejor calidad, pero pesado (~2GB de modelo,
  requiere torch) y lento en CPU. Recomendado para la siguiente etapa,
  no para la POC inicial.
- "kokoro": usa Kokoro-82M (paquete `kokoro`, pesos ~330MB vía Hugging
  Face). Punto medio entre espeak y xtts: mucho más natural que espeak,
  liviano y rápido en CPU (82M parámetros), licencia Apache 2.0 (sin
  restricciones de uso comercial, a diferencia de XTTS). Requiere
  espeak-ng instalado por fuera de pip para idiomas no ingleses (lo usa
  como fallback de fonemización).
"""

import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from dubbing_poc.registry import register
from dubbing_poc.voices import ESPEAK_VOICE_PRESETS, KOKORO_VOICE_PRESETS

KOKORO_SAMPLE_RATE = 24000

# Primera letra del código de voz Kokoro -> lang_code que espera KPipeline.
KOKORO_LANG_CODES = {"a": "a", "b": "b", "e": "e"}


class TTSBackend(ABC):
    # Si True, pipeline.py puede llamar a synth() concurrentemente desde
    # varios threads para distintas frases. Default seguro (False): un
    # backend nuevo, o uno que reuse un modelo cargado en memoria (como
    # XTTSBackend), corre secuencial salvo que declare lo contrario.
    THREAD_SAFE: bool = False

    @abstractmethod
    def synth(self, text: str, language: str, voice: str, out_path: str) -> None:
        """Genera un wav en out_path con el texto dado."""
        raise NotImplementedError

    @abstractmethod
    def list_voices(self) -> List[str]:
        raise NotImplementedError


@register("tts", "espeak")
class EspeakTTSBackend(TTSBackend):
    # Cada synth() lanza un subprocess independiente, sin estado
    # compartido entre llamadas: seguro de correr en paralelo.
    THREAD_SAFE = True

    def __init__(
        self,
        device: str = "cpu",
        sample_rate: int = 22050,
        speed: int = 165,
        pitch: int = 50,
        volume: int = 100,
    ):
        # device se ignora (espeak-ng no usa GPU); se acepta para que
        # la firma sea intercambiable con otros backends.
        if shutil.which("espeak-ng") is None:
            raise RuntimeError(
                "No se encontró 'espeak-ng'. Instalalo con: "
                "sudo apt-get install espeak-ng (Linux) o "
                "brew install espeak-ng (Mac)."
            )
        self.sample_rate = sample_rate
        self.speed = speed      # palabras por minuto, rango típico 80-260
        self.pitch = pitch      # 0-99 (espeak-ng), 50 = neutro
        self.volume = volume    # amplitud 0-200, 100 = neutro

    def list_voices(self) -> List[str]:
        return list(ESPEAK_VOICE_PRESETS.keys())

    def synth(self, text: str, language: str, voice: str, out_path: str) -> None:
        if voice not in ESPEAK_VOICE_PRESETS:
            raise ValueError(
                f"Voz '{voice}' no está en el catálogo de espeak. "
                f"Opciones: {self.list_voices()}"
            )
        lang_code, variant = ESPEAK_VOICE_PRESETS[voice]
        with open(out_path, "wb") as f:
            subprocess.run(
                [
                    "espeak-ng",
                    "-v", f"{lang_code}+{variant}",
                    "-s", str(self.speed),
                    "-p", str(self.pitch),
                    "-a", str(self.volume),
                    "--stdout",
                ],
                input=text.encode("utf-8"),
                stdout=f,
                check=True,
            )


@register("tts", "xtts")
class XTTSBackend(TTSBackend):
    """Backend de mayor calidad. Requiere `pip install coqui-tts`.

    Primer uso descarga el modelo (~2GB) desde Hugging Face. Con
    voice_sample_path se puede clonar una voz específica en vez de
    usar una de las incorporadas.
    """

    def __init__(
        self,
        device: str = "cpu",
        voice_sample_path: Optional[str] = None,
        speed: float = 1.0,
        temperature: float = 0.65,
    ):
        import torch  # import diferido: pesado
        from TTS.api import TTS

        self.voice_sample_path = voice_sample_path
        self.speed = speed              # 1.0 = normal; <1 más lento, >1 más rápido
        self.temperature = temperature  # variabilidad/expresividad del muestreo (0.01-1.0)
        resolved_device = device
        if device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(resolved_device)

    def list_voices(self) -> List[str]:
        return list(self._tts.speakers or [])

    def synth(self, text: str, language: str, voice: str, out_path: str) -> None:
        kwargs = dict(
            text=text, language=language, file_path=out_path,
            speed=self.speed, temperature=self.temperature,
        )
        if self.voice_sample_path:
            kwargs["speaker_wav"] = self.voice_sample_path
        else:
            kwargs["speaker"] = voice
        self._tts.tts_to_file(**kwargs)


@register("tts", "kokoro")
class KokoroTTSBackend(TTSBackend):
    """Backend de calidad intermedia. Requiere `pip install kokoro`.

    Un único KPipeline se mantiene cargado en memoria por lang_code (se
    crea uno nuevo la primera vez que se pide una voz de ese idioma), así
    que no es seguro llamar a synth() desde varios threads a la vez.
    """

    def __init__(self, device: str = "cpu", speed: float = 1.0):
        # import diferido: pesado (carga torch por debajo de kokoro/misaki).
        from kokoro import KPipeline

        self._KPipeline = KPipeline
        self._pipelines = {}
        self.device = None if device == "cpu" else device
        self.speed = speed  # 1.0 = normal; rango típico 0.5-2.0

    def list_voices(self) -> List[str]:
        return list(KOKORO_VOICE_PRESETS.keys())

    def _resolve_voice_code(self, voice: str) -> str:
        if voice in KOKORO_VOICE_PRESETS:
            return KOKORO_VOICE_PRESETS[voice]
        # Permite pasar directamente un código nativo de Kokoro (ej.
        # "af_sarah") aunque no esté en el catálogo curado.
        if "_" in voice and voice[0] in KOKORO_LANG_CODES:
            return voice
        raise ValueError(
            f"Voz '{voice}' no está en el catálogo de kokoro. "
            f"Opciones: {self.list_voices()}"
        )

    def _pipeline_for(self, lang_code: str):
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            pipeline = self._KPipeline(lang_code=lang_code, device=self.device)
            self._pipelines[lang_code] = pipeline
        return pipeline

    def synth(self, text: str, language: str, voice: str, out_path: str) -> None:
        import soundfile as sf

        voice_code = self._resolve_voice_code(voice)
        lang_code = KOKORO_LANG_CODES.get(voice_code[0], voice_code[0])
        pipeline = self._pipeline_for(lang_code)

        # KPipeline generator: divide texto largo en varios chunks: se
        # concatenan para dejar un único wav por frase, igual que los
        # demás backends (align.py ajusta la duración total después).
        chunks = [audio for _, _, audio in pipeline(text, voice=voice_code, speed=self.speed)]
        if not chunks:
            raise RuntimeError(f"Kokoro no generó audio para el texto: {text!r}")
        full_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(out_path, full_audio, KOKORO_SAMPLE_RATE)
