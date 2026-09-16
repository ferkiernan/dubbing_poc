"""Interfaz web local para dubbing_poc.

Server Flask simple: subís un video, elegís calidad de voz (espeak
rápido/robótico o xtts natural), idioma y voz, y el pipeline corre en
un hilo de fondo. La página consulta el progreso y ofrece el video
final para descargar.

Correr con: python -m dubbing_poc.webapp.app
Abre en: http://127.0.0.1:5000
"""

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# XTTS-v2 pide confirmar la licencia CPML de Coqui (no comercial,
# https://coqui.ai/cpml) por input() interactivo la primera vez que se
# descarga el modelo. Como el pipeline corre en un hilo de fondo sin
# stdin, eso revienta con EOFError. Esto asume que ya la aceptaste.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from flask import Flask, jsonify, render_template, request, send_file

from dubbing_poc import pipeline
from dubbing_poc.segments import Segment
from dubbing_poc.voices import ESPEAK_VOICE_PRESETS, KOKORO_VOICE_PRESETS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Voces XTTS incorporadas conocidas por sonar con acento estadounidense
# claro (subconjunto curado; el modelo trae ~58 en total). No requieren
# wav de referencia: se seleccionan por nombre.
XTTS_US_VOICES = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammy Grit",
    "Andrew Chipper",
    "Craig Gutsy",
]

# Voces de Kokoro (subconjunto curado, ver dubbing_poc/voices.py).
KOKORO_VOICES = list(KOKORO_VOICE_PRESETS.keys())

app = Flask(__name__)

jobs: Dict[str, "Job"] = {}
jobs_lock = threading.Lock()

# Peso relativo de cada etapa para estimar el % de progreso. La síntesis
# por frase ("synth_segment") es la única que se subdivide por cantidad
# de frases, el resto son pasos puntuales.
STAGE_WEIGHTS = {
    "extract_audio": 3,
    "load_asr": 2,
    "transcribe": 15,
    "load_tts": 10,
    "synth_segment": 60,
    "build_timeline": 5,
    "mux": 5,
}
STAGE_LABELS = {
    "extract_audio": "Extrayendo audio del video (ffmpeg)...",
    "load_asr": "Cargando modelo de transcripción (si es la primera vez con "
                "este tamaño, puede tardar varios minutos descargándolo)...",
    "transcribe": "Transcribiendo habla...",
    "transcribed": "Transcripción completa.",
    "merged_sentences": "Frases fusionadas.",
    "load_tts": "Cargando motor de voz (si es la primera vez, descarga "
                "el modelo — puede tardar varios minutos)...",
    "synth_segment": "Sintetizando frase...",
    "build_timeline": "Armando la pista de audio final...",
    "mux": "Remultiplexando con el video (ffmpeg)...",
    "done": "Listo.",
}


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
TRANSCRIPT_EXTENSION = ".dubbing_poc.json"


def _sanitize_voice_id(voice: str) -> str:
    """'Claribel Dervla' -> 'ClaribelDervla', 'en-m1' -> 'en-m1'. Solo
    para uso en nombres de archivo: sin espacios ni caracteres raros."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", voice.replace(" ", ""))
    return cleaned or "voz"


def _output_filename(base_name: str, ext: str, voice: str) -> str:
    return f"{base_name}_voz_{_sanitize_voice_id(voice)}{ext or '.mp4'}"


def _transcript_path_for(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    # Si el output ya tenía extensión doble por _output_filename, splitext
    # solo saca la última (.mp4), lo cual es lo que queremos acá.
    return base + TRANSCRIPT_EXTENSION


def _save_transcript(
    output_path: str, source_video: str, original_filename: str,
    segments: List[Segment], opts: dict,
) -> str:
    transcript_path = _transcript_path_for(output_path)
    data = {
        "source_video": source_video,
        "original_filename": original_filename,
        "language": opts["language"],
        "asr_backend": "whisper",
        "asr_model_size": opts["asr_model_size"],
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ],
    }
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return transcript_path


def _load_transcript(transcript_path: str) -> dict:
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["segments"] = [
        Segment(start=s["start"], end=s["end"], text=s["text"]) for s in data["segments"]
    ]
    return data


def _find_cached_transcript(
    original_filename: str, language: str, asr_model_size: str,
    search_dir: str = OUTPUT_DIR,
) -> Optional[dict]:
    """Busca en search_dir una transcripción ya guardada (.dubbing_poc.json)
    para el mismo archivo de entrada (por nombre) y la misma config de ASR
    (idioma + tamaño de modelo). Si la encuentra, se reusa y se salta la
    transcripción por completo — solo matchea por nombre+config, no por
    contenido del video (si el archivo cambió mantiendo el nombre, esto
    reusaría una transcripción desactualizada)."""
    try:
        candidates = [
            name for name in os.listdir(search_dir)
            if name.endswith(TRANSCRIPT_EXTENSION)
        ]
    except OSError:
        return None

    # Más reciente primero: si hay varias corridas para el mismo archivo,
    # se prefiere la última.
    candidates.sort(
        key=lambda name: os.path.getmtime(os.path.join(search_dir, name)),
        reverse=True,
    )

    for name in candidates:
        path = os.path.join(search_dir, name)
        try:
            transcript = _load_transcript(path)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if (
            transcript.get("original_filename") == original_filename
            and transcript.get("language") == language
            and transcript.get("asr_model_size") == asr_model_size
        ):
            return transcript
    return None


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued -> running -> done -> error
    message: str = ""
    progress: int = 0  # 0-100
    output_path: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False  # True si se saltó por ya existir el output (modo single)
    logs: List[str] = field(default_factory=list)
    # Solo se usan en jobs de lote (batch): progreso agregado sobre la
    # lista de videos, además del progreso normal (0-100) del video
    # que se está procesando en ese momento.
    is_batch: bool = False
    batch_total: int = 0
    batch_index: int = 0  # video actual, 1-based
    batch_current_file: str = ""
    batch_results: List[dict] = field(default_factory=list)  # [{file, status, error}]

    def log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {line}")


def _stage_start_pct(stage: str) -> int:
    order = list(STAGE_WEIGHTS.keys())
    return sum(STAGE_WEIGHTS[s] for s in order[: order.index(stage)]) if stage in order else 0


def _make_progress_callback(job: Job):
    last_transcribe_pct = {"value": -1}

    def on_progress(stage: str, **data) -> None:
        if stage == "transcribed":
            job.log(f"Transcripción lista: {data['count']} frase(s) detectadas.")
            return

        if stage == "merged_sentences":
            job.log(f"Frases fusionadas: {data['before']} -> {data['after']}.")
            return

        if stage == "transcribe_progress":
            pct = data["pct"]
            # Solo logueamos cuando el % avanza, para no floodear con
            # segmentos muy cortos (silencios, palabras sueltas).
            if pct != last_transcribe_pct["value"]:
                last_transcribe_pct["value"] = pct
                seg = data["segment"]
                snippet = seg.text[:60] + ("..." if len(seg.text) > 60 else "")
                job.log(f"Transcribiendo... {pct}% [{seg.end:.1f}s]: \"{snippet}\"")
            return

        if stage == "synth_segment":
            i, total = data["index"], data["total"]
            base = _stage_start_pct("synth_segment")
            # Con backends thread-safe (espeak), las frases pueden terminar
            # desordenadas — max() evita que la barra retroceda visualmente.
            job.progress = max(job.progress, base + int(STAGE_WEIGHTS["synth_segment"] * (i / max(total, 1))))
            snippet = data["text"][:60] + ("..." if len(data["text"]) > 60 else "")
            job.message = f"Sintetizando frase {i + 1}/{total}..."
            job.log(f"Frase {i + 1}/{total} [{data['start']:.1f}s-{data['end']:.1f}s]: \"{snippet}\"")
            return

        job.progress = _stage_start_pct(stage) if stage != "done" else 100
        job.message = STAGE_LABELS.get(stage, stage)
        job.log(job.message)

    return on_progress


def _process_one(
    job: Job, input_path: str, output_path: str, opts: dict,
    original_filename: str,
    precomputed_segments: Optional[List[Segment]] = None,
) -> "pipeline.DubResult":
    """Corre el pipeline para un solo video, actualizando job.progress
    (0-100) vía el callback de progreso. No cambia job.status: eso lo
    maneja el caller (_run_job para un video suelto, _run_batch_job
    para cada ítem del lote). Guarda un .json con la transcripción junto
    al output, para poder re-sintetizar con otra voz sin re-transcribir
    (pasando ese archivo como precomputed_segments).

    Si no se pasa precomputed_segments explícitamente (ej. desde
    /reprocess), busca automáticamente una transcripción cacheada por
    nombre de archivo + idioma + modelo ASR (ver _find_cached_transcript)
    antes de transcribir de nuevo — así una segunda corrida sobre el
    mismo archivo (ej. para probar otra voz) no repite el ASR."""
    if precomputed_segments is None:
        search_dir = os.path.dirname(output_path) or OUTPUT_DIR
        cached = _find_cached_transcript(
            original_filename, opts["language"], opts["asr_model_size"], search_dir
        )
        if cached is not None:
            job.log(f"Transcripción reusada de una corrida anterior para '{original_filename}' (mismo idioma/modelo, no se vuelve a transcribir).")
            precomputed_segments = cached["segments"]

    result = pipeline.run(
        video_path=input_path,
        output_path=output_path,
        voice=opts["voice"],
        language=opts["language"],
        tts_backend=opts["tts_backend"],
        asr_backend="whisper",
        asr_model_size=opts["asr_model_size"],
        device="cpu",
        tts_kwargs=opts["tts_kwargs"],
        on_progress=_make_progress_callback(job),
        precomputed_segments=precomputed_segments,
        merge_sentences=opts.get("merge_sentences", False),
    )
    transcript_path = _save_transcript(output_path, input_path, original_filename, result.segments, opts)
    job.log(f"Transcripción guardada: {transcript_path}")
    return result


def _run_job(job: Job, input_path: str, output_path: str, opts: dict, original_filename: str) -> None:
    try:
        job.status = "running"
        if os.path.isfile(output_path):
            job.log(
                f"Salteado: ya existe un video de salida igual en '{output_path}' "
                f"(mismo nombre esperado). No se vuelve a procesar."
            )
            job.status = "done"
            job.progress = 100
            job.output_path = output_path
            job.message = "Ya existía: se saltó el procesamiento."
            job.skipped = True
            return
        job.log(f"Iniciando: backend TTS={opts['tts_backend']}, voz={opts['voice']}, idioma={opts['language']}")
        _process_one(job, input_path, output_path, opts, original_filename)
        job.status = "done"
        job.progress = 100
        job.output_path = output_path
        job.message = "Listo."
    except Exception as exc:  # noqa: BLE001 - se reporta al usuario en la UI
        job.status = "error"
        job.error = str(exc)
        job.message = "Falló el procesamiento."
        job.log(f"ERROR: {exc}")


def _run_batch_job(job: Job, files: List[str], output_dir: str, opts: dict) -> None:
    job.status = "running"
    job.is_batch = True
    job.batch_total = len(files)
    job.log(f"Iniciando lote: {len(files)} video(s), backend TTS={opts['tts_backend']}, voz={opts['voice']}")

    for i, input_path in enumerate(files, start=1):
        filename = os.path.basename(input_path)
        job.batch_index = i
        job.batch_current_file = filename
        job.progress = 0
        job.log(f"--- Video {i}/{len(files)}: {filename} ---")

        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, _output_filename(name, ext, opts["voice"]))

        if os.path.isfile(output_path):
            job.log(
                f"Salteado: '{filename}' ya tiene un video de salida igual en "
                f"'{output_path}' (mismo nombre esperado). Se pasa al siguiente."
            )
            job.batch_results.append({"file": filename, "status": "skipped", "output": output_path})
            continue

        try:
            _process_one(job, input_path, output_path, opts, filename)
            job.batch_results.append({"file": filename, "status": "done", "output": output_path})
            job.log(f"Listo: {filename} -> {output_path}")
        except Exception as exc:  # noqa: BLE001 - se reporta al usuario en la UI, se sigue con el resto
            job.batch_results.append({"file": filename, "status": "error", "error": str(exc)})
            job.log(f"ERROR en {filename}: {exc}")

    job.status = "done"
    job.progress = 100
    failed = sum(1 for r in job.batch_results if r["status"] == "error")
    skipped = sum(1 for r in job.batch_results if r["status"] == "skipped")
    done = len(files) - failed - skipped
    job.message = f"Lote terminado: {done}/{len(files)} exitosos, {skipped} salteados, {failed} con error."
    job.log(job.message)


@app.route("/")
def index():
    return render_template(
        "index.html",
        espeak_voices=list(ESPEAK_VOICE_PRESETS.keys()),
        xtts_voices=XTTS_US_VOICES,
        kokoro_voices=KOKORO_VOICES,
    )


def _form_float(name: str, default: float) -> float:
    raw = (request.form.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _form_int(name: str, default: int) -> int:
    raw = (request.form.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _build_opts_from_form(job_id: str, voice_sample) -> dict:
    quality = request.form.get("quality", "fast")  # "fast" (espeak) | "kokoro" | "hq" (xtts)
    language = request.form.get("language", "en")

    tts_kwargs = {}
    if quality == "hq":
        tts_backend = "xtts"
        if voice_sample and voice_sample.filename:
            sample_path = os.path.join(UPLOAD_DIR, f"{job_id}_voice.wav")
            voice_sample.save(sample_path)
            tts_kwargs["voice_sample_path"] = sample_path
            voice = "cloned"
        else:
            voice = request.form.get("xtts_voice", XTTS_US_VOICES[0])
        tts_kwargs["speed"] = _form_float("xtts_speed", 1.0)
        tts_kwargs["temperature"] = _form_float("xtts_temperature", 0.65)
    elif quality == "kokoro":
        tts_backend = "kokoro"
        voice = request.form.get("kokoro_voice", KOKORO_VOICES[0])
        tts_kwargs["speed"] = _form_float("kokoro_speed", 1.0)
    else:
        tts_backend = "espeak"
        voice = request.form.get("espeak_voice", "en-m1")
        tts_kwargs["speed"] = _form_int("espeak_speed", 165)
        tts_kwargs["pitch"] = _form_int("espeak_pitch", 50)
        tts_kwargs["volume"] = _form_int("espeak_volume", 100)

    return {
        "voice": voice,
        "language": language,
        "tts_backend": tts_backend,
        "asr_model_size": request.form.get("asr_model_size", "medium"),
        "tts_kwargs": tts_kwargs,
        "merge_sentences": request.form.get("merge_sentences") == "on",
    }


@app.route("/submit", methods=["POST"])
def submit():
    video = request.files.get("video")
    if not video or video.filename == "":
        return jsonify({"error": "Subí un archivo de video."}), 400

    output_dir = (request.form.get("single_output_dir") or "").strip() or OUTPUT_DIR
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        return jsonify({"error": f"No se pudo crear/usar la carpeta destino: {exc}"}), 400

    job_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(video.filename)[1] or ".mp4"
    original_filename = os.path.basename(video.filename)
    original_name = os.path.splitext(original_filename)[0]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    video.save(input_path)

    opts = _build_opts_from_form(job_id, request.files.get("voice_sample"))
    # Sin prefijo de job_id: así el nombre de salida es estable entre
    # corridas del mismo video+voz, y se puede detectar que ya existe
    # (ver _run_job / skip por archivo ya procesado).
    output_path = os.path.join(output_dir, _output_filename(original_name, ext, opts["voice"]))

    job = Job(id=job_id)
    jobs[job_id] = job

    thread = threading.Thread(
        target=_run_job, args=(job, input_path, output_path, opts, original_filename), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/reprocess", methods=["POST"])
def reprocess():
    """Re-sintetiza un video ya procesado con otra voz, reusando la
    transcripción guardada (.dubbing_poc.json junto al output anterior)
    en vez de transcribir de nuevo."""
    transcript_path = (request.form.get("transcript_path") or "").strip()
    if not transcript_path:
        return jsonify({"error": "Indicá la ruta del archivo .dubbing_poc.json a reusar."}), 400
    if not os.path.isfile(transcript_path):
        return jsonify({"error": f"No se encontró el archivo: {transcript_path}"}), 400

    try:
        transcript = _load_transcript(transcript_path)
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        return jsonify({"error": f"No se pudo leer la transcripción: {exc}"}), 400

    source_video = transcript["source_video"]
    if not os.path.isfile(source_video):
        return jsonify({
            "error": f"El video original ya no está en: {source_video} "
                     "(se necesita para tomar duración y remultiplexar)."
        }), 400

    job_id = uuid.uuid4().hex[:12]
    opts = _build_opts_from_form(job_id, request.files.get("voice_sample"))
    # El idioma y modelo ASR quedan fijados por la transcripción original,
    # no por el form (no tendría sentido cambiarlos sin re-transcribir).
    opts["language"] = transcript["language"]
    opts["asr_model_size"] = transcript["asr_model_size"]

    output_dir = os.path.dirname(transcript_path)
    base_name = os.path.basename(transcript_path)[: -len(TRANSCRIPT_EXTENSION)]
    # base_name puede venir con un sufijo _voz_X de la corrida anterior;
    # se lo saco para no encadenar sufijos en re-reprocesos sucesivos.
    base_name = re.sub(r"_voz_[A-Za-z0-9_-]+$", "", base_name)
    ext = os.path.splitext(source_video)[1] or ".mp4"
    output_path = os.path.join(output_dir, _output_filename(base_name, ext, opts["voice"]))

    job = Job(id=job_id)
    jobs[job_id] = job

    def _run_reprocess():
        try:
            job.status = "running"
            job.log(f"Reprocesando con voz={opts['voice']} (transcripción reusada, sin re-transcribir)")
            original_filename = transcript.get("original_filename", os.path.basename(source_video))
            _process_one(
                job, source_video, output_path, opts, original_filename,
                precomputed_segments=transcript["segments"],
            )
            job.status = "done"
            job.progress = 100
            job.output_path = output_path
            job.message = "Listo."
        except Exception as exc:  # noqa: BLE001 - se reporta al usuario en la UI
            job.status = "error"
            job.error = str(exc)
            job.message = "Falló el procesamiento."
            job.log(f"ERROR: {exc}")

    thread = threading.Thread(target=_run_reprocess, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/submit_batch", methods=["POST"])
def submit_batch():
    source_dir = (request.form.get("source_dir") or "").strip()
    output_dir = (request.form.get("output_dir") or "").strip()

    if not source_dir or not output_dir:
        return jsonify({"error": "Indicá carpeta origen y carpeta destino."}), 400
    if not os.path.isdir(source_dir):
        return jsonify({"error": f"La carpeta origen no existe: {source_dir}"}), 400

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        return jsonify({"error": f"No se pudo crear/usar la carpeta destino: {exc}"}), 400

    files = sorted(
        os.path.join(source_dir, name)
        for name in os.listdir(source_dir)
        if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
        and os.path.isfile(os.path.join(source_dir, name))
    )
    if not files:
        return jsonify({"error": f"No se encontraron videos en {source_dir}."}), 400

    job_id = uuid.uuid4().hex[:12]
    opts = _build_opts_from_form(job_id, request.files.get("voice_sample"))

    job = Job(id=job_id)
    jobs[job_id] = job

    thread = threading.Thread(
        target=_run_batch_job, args=(job, files, output_dir, opts), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "file_count": len(files)})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job no encontrado."}), 404
    return jsonify(
        {
            "status": job.status,
            "message": job.message,
            "progress": job.progress,
            "error": job.error,
            "download_url": (
                f"/download/{job_id}" if job.status == "done" and not job.is_batch else None
            ),
            "skipped": job.skipped,
            "is_batch": job.is_batch,
            "batch_total": job.batch_total,
            "batch_index": job.batch_index,
            "batch_current_file": job.batch_current_file,
            "batch_results": job.batch_results,
        }
    )


@app.route("/logs/<job_id>")
def logs(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job no encontrado."}), 404
    since = request.args.get("since", default=0, type=int)
    new_lines = job.logs[since:]
    return jsonify({"lines": new_lines, "next": since + len(new_lines)})


@app.route("/download/<job_id>")
def download(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.status != "done" or not job.output_path:
        return jsonify({"error": "El resultado no está listo."}), 404
    return send_file(job.output_path, as_attachment=True)


def main():
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
