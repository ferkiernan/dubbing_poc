"""Herramienta de doblaje/cambio de voz local, sincronizada por frase.

POC modular: cada etapa (ASR, TTS, alineado de tiempo, audio/video) es
intercambiable. Ver README.md para el roadmap de módulos a agregar
(traducción, voice-conversion tipo RVC, separación de música de fondo, etc).
"""

__version__ = "0.1.0"

# Importar estos módulos registra sus backends (decorador @register).
# Los imports pesados (torch, faster-whisper, TTS) están adentro de
# cada __init__ de backend, así que esto es liviano.
from dubbing_engine import asr, tts  # noqa: E402,F401

