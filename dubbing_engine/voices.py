"""Catálogo de voces disponibles por backend de TTS.

Este archivo es el punto de extensión para "agregar más voces":
- Para 'espeak': agregar entradas nuevas a ESPEAK_VOICE_PRESETS.
- Para 'xtts': XTTSBackend.list_voices() ya expone las ~58 voces
  incorporadas del modelo (varias femeninas), y además acepta clonar
  una voz específica pasando un wav de referencia (ver tts.py).
- Para 'kokoro': agregar entradas nuevas a KOKORO_VOICE_PRESETS. El
  prefijo del código de voz define idioma+género para Kokoro (ver
  KOKORO_LANG_CODES en tts.py).
"""

# variante "+fN" = voz femenina número N; "+mN" = masculina. espeak-ng
# aplica estas variantes sobre el idioma base.
ESPEAK_VOICE_PRESETS = {
    "es-f1": ("es", "f2"),
    "es-f2": ("es", "f3"),
    "es-f3": ("es", "f4"),
    "es-m1": ("es", "m3"),
    "en-f1": ("en-us", "f3"),
    "en-f2": ("en-us", "f4"),
    "en-m1": ("en-us", "m3"),
}


def list_espeak_voices():
    return list(ESPEAK_VOICE_PRESETS.keys())


# Voces incorporadas de Kokoro-82M (checkpoint v1.0). El código sigue el
# esquema "<lang><genero>_<nombre>" de la librería `kokoro`: primera
# letra = idioma (a=en-US, b=en-GB, e=es), segunda = f/m (femenina/
# masculina). Subconjunto curado con buena calidad conocida; el modelo
# trae más voces (ver https://huggingface.co/hexgrad/Kokoro-82M).
KOKORO_VOICE_PRESETS = {
    "en-f1": "af_heart",
    "en-f2": "af_bella",
    "en-f3": "af_nicole",
    "en-m1": "am_adam",
    "en-m2": "am_michael",
    "en-gb-f1": "bf_emma",
    "en-gb-m1": "bm_george",
    "es-f1": "ef_dora",
    "es-m1": "em_alex",
}


def list_kokoro_voices():
    return list(KOKORO_VOICE_PRESETS.keys())
