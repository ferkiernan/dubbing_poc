# dubbing_poc

POC local para cambiar la voz de un video, manteniendo el tiempo de cada
frase. Corre 100% en tu máquina, sin mandar nada a la nube.

## Cómo funciona

1. Extrae el audio del video (ffmpeg).
2. Transcribe el audio frase por frase, con sus tiempos (Whisper local,
   vía `faster-whisper`).
3. Genera cada frase de nuevo con la voz elegida (TTS).
4. Estira o comprime cada frase generada para que dure lo mismo que la
   original (así el video no se desincroniza).
5. Arma la pista de audio completa y la vuelve a poner en el video
   (ffmpeg, sin recodificar el video).

## Instalación

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install -y ffmpeg espeak-ng

# Mac
brew install ffmpeg espeak-ng

pip install -r requirements.txt
```

### Windows + backend `kokoro`: usar Python 3.11

El backend `kokoro` depende de `spacy`/`blis` (paquetes con extensiones
en C), que todavía no publican wheels precompilados para Python 3.14
en Windows — instalarlo con esa versión falla al intentar compilar
`blis` desde código fuente. Solución: un entorno virtual con Python
3.11 (o 3.12) solo para este proyecto, sin tocar el Python del sistema.

```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\python.exe -m pip install --upgrade pip
.venv311\Scripts\python.exe -m pip install -r requirements.txt flask kokoro
```

`start.bat` ya detecta y usa `.venv311` automáticamente si existe. Si
no tenés Python 3.11 instalado, bajalo de python.org (marcando "Add to
PATH" es opcional, `py -3.11` lo encuentra igual vía el Python Launcher).

Nota sobre tu GPU: tenés una AMD, no NVIDIA. Los paquetes de este POC
(`faster-whisper`, `espeak-ng`) corren perfecto en CPU, no necesitan
GPU para esta primera etapa. Si más adelante sumás el backend `xtts`
(mejor calidad), en CPU va a ser lento; con GPU AMD la única vía
razonable es ROCm en Linux (soporte parcial de PyTorch), así que por
ahora conviene probar `xtts` en CPU o dejarlo para cuando lo necesites
de verdad.

## Uso

```bash
# Ver voces disponibles
python main.py list-voices

# Cambiar la voz de un video (misma frase, mismo idioma, otra voz)
python main.py run entrada.mp4 -o salida.mp4 --voice es-f2 --lang es

# Modelo de Whisper más grande = mejor transcripción, más lento
# (default ya es "medium"; "large-v3" es la opción de máxima precisión)
python main.py run entrada.mp4 -o salida.mp4 --voice es-f1 --lang es --asr-model-size large-v3

# Ajustar la voz (ver "Personalización de voz" más abajo)
python main.py run entrada.mp4 -o salida.mp4 --voice es-f1 --lang es --espeak-speed 190 --pitch 60
```

Voces incluidas en esta POC (motor `espeak-ng`, calidad "robótica" pero
gratis, liviano y sin descargas): `es-f1`, `es-f2`, `es-f3` (femeninas),
`es-m1` (masculina), y sus equivalentes en inglés (`en-f1`, `en-f2`,
`en-m1`). Se agregan nuevas en `dubbing_poc/voices.py`.

El modelo de transcripción por default es `medium` (mejor precisión que
`small` a costa de tardar más y descargar un modelo más pesado la
primera vez). Para transcripciones rápidas de prueba, usar
`--asr-model-size tiny` o `small`.

### Personalización de voz

Cada backend de TTS acepta ajustes finos vía `tts_kwargs` (CLI: flags
dedicadas; interfaz web: sliders que aparecen según la calidad elegida):

| Backend  | Parámetro     | CLI                | Rango típico | Default |
|----------|---------------|--------------------|--------------|---------|
| espeak   | velocidad     | `--espeak-speed`   | 80-260       | 165     |
| espeak   | tono          | `--pitch`          | 0-99         | 50      |
| espeak   | volumen       | `--volume`         | 0-200        | 100     |
| kokoro   | velocidad     | `--speed`          | 0.5-2.0      | 1.0     |
| xtts     | velocidad     | `--speed`          | 0.5-2.0      | 1.0     |
| xtts     | expresividad  | `--temperature`    | 0.1-1.0      | 0.65    |

`temperature` en XTTS controla la variabilidad del muestreo: valores
más altos suenan más expresivos/variados, más bajos más planos y
consistentes entre frases.

### Auto-cache de transcripción (interfaz web)

Al subir un video con el mismo nombre de archivo que uno ya procesado
antes (con el mismo idioma y modelo de Whisper), la interfaz web
detecta la transcripción guardada de esa corrida anterior y la reusa
automáticamente, sin volver a transcribir — útil para probar varias
voces/backends sobre el mismo video sin pagar el costo de ASR cada vez.
El log del job indica cuándo esto ocurre ("Transcripción reusada..."). Es
un match por nombre de archivo, no por contenido: si reemplazás el
archivo manteniendo el mismo nombre, se reusaría igual la transcripción
vieja (para forzar una re-transcripción, subilo con otro nombre o
borrá el `.dubbing_poc.json` correspondiente en la carpeta de salida).
Esto es adicional al modo "Reprocesar" manual, que sigue disponible
para apuntar a un `.dubbing_poc.json` específico.

### Carpeta de salida y salteo de videos ya procesados (interfaz web)

En el modo "Un video", el campo opcional "Carpeta destino" permite
elegir dónde se guarda el resultado (si se deja vacío, usa la carpeta
de salida por defecto de la app, `webapp/outputs/`). Todos los campos
del formulario —incluida esta carpeta destino, la de "Lote", voces,
sliders de ajuste de voz, checkboxes, etc.— se guardan en el
`localStorage` del navegador y se restauran automáticamente la próxima
vez que abrís la página.

Tanto en modo "Un video" como en "Lote", si el archivo de salida que se
generaría (mismo nombre base + sufijo de voz) ya existe en la carpeta
destino, ese video se saltea sin volver a procesarlo — se avisa en la
consola/log del job ("Salteado: ya existe...") y, en modo lote, se
continúa con el siguiente archivo. Es una comparación por nombre de
archivo esperado, no por contenido: para forzar el reprocesamiento,
cambiá la voz/idioma (cambia el sufijo del nombre) o borrá el archivo
de salida existente.

### Frases cortadas en la transcripción

Whisper a veces corta una misma oración en varios segmentos por una
pausa breve (coma, respiración), aunque semánticamente sea una sola
frase. Con `--merge-sentences` (CLI) o el checkbox "Mantener frases
completas" (interfaz web), se fusionan los segmentos consecutivos que
no terminan en puntuación terminal (`.`, `!`, `?`, `…`) y están
separados por un silencio corto (≤0.6s por defecto), siempre que la
duración fusionada no supere los 12s (ver
`dubbing_poc/segments.py::merge_sentence_segments`). Es una heurística
por reglas, no un modelo de lenguaje: rápida, determinística y sin
dependencias nuevas.

```bash
python main.py run entrada.mp4 -o salida.mp4 --voice es-f1 --lang es --merge-sentences
```

## Probar sin tener un video propio

```bash
espeak-ng -v es -s 150 --stdout "Hola, esto es una prueba de doblaje." > /tmp/speech.wav
ffmpeg -y -f lavfi -i color=c=blue:s=320x240:r=15 -i /tmp/speech.wav -shortest \
  -c:v libx264 -c:a aac /tmp/test.mp4
python main.py run /tmp/test.mp4 -o /tmp/test_out.mp4 --voice es-f2 --lang es --asr-model-size tiny
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Los tests cubren el ajuste de duración (`align.py`) y el pipeline
completo (TTS real con espeak + timeline + mux), sin depender de bajar
el modelo de Whisper (para no requerir red en CI).

## Arquitectura (para ampliar)

Todo backend se registra con un decorador y se selecciona por nombre
desde la CLI, sin tocar el resto del código:

- `dubbing_poc/asr.py` — transcripción (`whisper` registrado).
- `dubbing_poc/tts.py` — síntesis de voz (`espeak`, `xtts`, `kokoro` registrados).
- `dubbing_poc/voices.py` — catálogo de voces por backend.
- `dubbing_poc/segments.py` — tipo `Segment` y fusión opcional de frases cortadas.
- `dubbing_poc/align.py` — ajuste de duración por frase.
- `dubbing_poc/audio_io.py` — extracción, timeline, mux con ffmpeg.
- `dubbing_poc/pipeline.py` — orquesta todo lo anterior.

## Backend de voz: Kokoro-82M

Punto medio entre `espeak` (robótico, liviano) y `xtts` (muy natural,
pesado): calidad notablemente mejor que espeak, 82M de parámetros,
corre rápido en CPU, y licencia **Apache 2.0** (sin restricciones de
uso comercial, a diferencia de los pesos de XTTS).

```bash
pip install kokoro
python main.py run entrada.mp4 -o salida.mp4 --tts-backend kokoro --voice en-f1 --lang en
```

Requiere `espeak-ng` instalado (ya es dependencia del proyecto) como
fallback de fonemización para idiomas no ingleses. La primera vez
descarga ~330MB de pesos desde Hugging Face. Voces curadas en
`dubbing_poc/voices.py::KOKORO_VOICE_PRESETS` (inglés US/UK y español,
masculinas y femeninas); también acepta pasar directamente un código
nativo de Kokoro (ej. `--voice af_sarah`) aunque no esté en el catálogo.

## Roadmap de módulos a agregar

Ideas para cuando quieras ampliar esta POC, en orden sugerido:

1. **Voz de mejor calidad**: activar el backend `xtts` (ya está el
   código, falta `pip install coqui-tts`). Da ~58 voces incorporadas
   (varias femeninas) y permite clonar una voz específica pasando un
   wav de 6-10 segundos con `--voice-sample`.
2. **Traducción** (doblaje a otro idioma, no solo cambio de voz): un
   nuevo módulo `translate.py` con un `Translator` local (por ejemplo
   Argos Translate, offline) que traduzca `seg.text` antes del paso de
   TTS. El resto del pipeline no cambia.
3. **Conservar música/efectos de fondo**: hoy se reemplaza toda la
   pista de audio. Se puede agregar separación de voz/música (ej.
   Demucs) antes de extraer el habla, y mezclar la música original con
   la voz nueva al final.
4. **Cambio de voz preservando prosodia exacta** (en vez de
   transcribir y re-sintetizar): un backend de *voice conversion* tipo
   RVC, que convierte el timbre de la voz original directamente sobre
   el audio (conserva automáticamente el timing exacto, sin necesidad
   de estirar/comprimir). Requiere un modelo de voz entrenado o
   descargado para la voz destino.
5. **Alineado por palabra** en vez de por frase (más preciso, usando
   `word_timestamps=True` de faster-whisper o WhisperX).

## Licencias a tener en cuenta

- `espeak-ng`: GPL.
- `faster-whisper` / modelos Whisper: MIT / licencia abierta de OpenAI.
- Si activás `xtts` (Coqui): el código es MPL 2.0, pero los *pesos* del
  modelo XTTS-v2 tienen una licencia propia (Coqui Public Model
  License) con restricciones para uso comercial — revisá esa licencia
  antes de usar el resultado con fines comerciales.
- Si activás `kokoro`: código y pesos del modelo Kokoro-82M están bajo
  Apache 2.0, sin restricciones de uso comercial.
