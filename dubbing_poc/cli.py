import argparse
import sys

from dubbing_poc import pipeline
from dubbing_poc.registry import available
from dubbing_poc.voices import list_espeak_voices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubbing_poc",
        description=(
            "Cambia la voz de un video local, manteniendo el tiempo de "
            "cada frase. POC: transcribe -> re-sintetiza con otra voz -> "
            "ajusta duración -> remultiplexa."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Procesa un video")
    run_p.add_argument("input", help="Video de entrada (mp4, mkv, etc.)")
    run_p.add_argument("-o", "--output", required=True, help="Video de salida")
    run_p.add_argument("--voice", default="es-f1", help="Voz destino (ver 'list-voices')")
    run_p.add_argument("--lang", default="es", help="Idioma del habla (código, ej: es, en)")
    run_p.add_argument("--tts-backend", default="espeak", choices=available("tts") or ["espeak", "xtts"])
    run_p.add_argument("--asr-backend", default="whisper", choices=available("asr") or ["whisper"])
    run_p.add_argument("--asr-model-size", default="medium", help="tiny/base/small/medium/large-v3")
    run_p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    run_p.add_argument(
        "--voice-sample",
        default=None,
        help="(solo --tts-backend xtts) wav de referencia para clonar una voz específica",
    )
    run_p.add_argument(
        "--merge-sentences",
        action="store_true",
        help=(
            "Fusiona segmentos que Whisper cortó a mitad de una oración "
            "(por una pausa breve), para que el TTS reciba la frase "
            "entera en vez de pedazos separados."
        ),
    )
    run_p.add_argument(
        "--speed", type=float, default=None,
        help="(kokoro, xtts) velocidad de habla, 1.0 = normal (típico 0.5-2.0)",
    )
    run_p.add_argument(
        "--espeak-speed", type=int, default=None,
        help="(solo espeak) palabras por minuto, default 165 (típico 80-260)",
    )
    run_p.add_argument(
        "--pitch", type=int, default=None,
        help="(solo espeak) tono de voz 0-99, default 50",
    )
    run_p.add_argument(
        "--volume", type=int, default=None,
        help="(solo espeak) volumen 0-200, default 100",
    )
    run_p.add_argument(
        "--temperature", type=float, default=None,
        help="(solo xtts) variabilidad/expresividad del muestreo 0.01-1.0, default 0.65",
    )

    sub.add_parser("list-voices", help="Lista voces disponibles (backend espeak)")
    sub.add_parser("list-backends", help="Lista los backends de ASR/TTS registrados")

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-voices":
        print("Voces disponibles (backend 'espeak'):")
        for v in list_espeak_voices():
            print(f"  {v}")
        print(
            "\nCon --tts-backend xtts hay ~58 voces incorporadas de mayor "
            "calidad (correr 'list-backends' o ver README), o clonar una "
            "voz propia con --voice-sample archivo.wav"
        )
        return 0

    if args.command == "list-backends":
        print("ASR:", available("asr"))
        print("TTS:", available("tts"))
        return 0

    if args.command == "run":
        tts_kwargs = {}
        if args.voice_sample:
            tts_kwargs["voice_sample_path"] = args.voice_sample

        if args.tts_backend in ("kokoro", "xtts") and args.speed is not None:
            tts_kwargs["speed"] = args.speed
        if args.tts_backend == "espeak":
            if args.espeak_speed is not None:
                tts_kwargs["speed"] = args.espeak_speed
            if args.pitch is not None:
                tts_kwargs["pitch"] = args.pitch
            if args.volume is not None:
                tts_kwargs["volume"] = args.volume
        if args.tts_backend == "xtts" and args.temperature is not None:
            tts_kwargs["temperature"] = args.temperature

        result = pipeline.run(
            video_path=args.input,
            output_path=args.output,
            voice=args.voice,
            language=args.lang,
            tts_backend=args.tts_backend,
            asr_backend=args.asr_backend,
            asr_model_size=args.asr_model_size,
            device=args.device,
            tts_kwargs=tts_kwargs,
            merge_sentences=args.merge_sentences,
        )
        print(f"Listo: {result.output_path}")
        print(f"Frases procesadas: {len(result.segments)}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
