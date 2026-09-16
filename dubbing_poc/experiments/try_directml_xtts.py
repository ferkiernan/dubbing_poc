"""EXPERIMENTO AISLADO. No se importa desde pipeline.py ni ningún módulo
productivo. No modifica tts.py ni requirements.txt.

Objetivo: determinar si XTTS-v2 puede correr sobre DirectML (única vía
teórica de aceleración GPU en Windows con una GPU AMD, ya que ni ROCm ni
CUDA son viables acá) y si da una mejora de tiempo real vs. CPU antes de
considerar integrarlo al backend XTTSBackend (tts.py).

IMPORTANTE — requiere un virtualenv SEPARADO del entorno productivo:
    torch-directml suele exigir una versión de `torch` distinta a la
    que ya usa XTTSBackend en este proyecto (torch 2.13.0+cpu). Instalar
    torch-directml en el entorno productivo puede romper el backend XTTS
    que ya funciona en CPU.

    python -m venv .venv-directml-experiment
    # Windows:
    .venv-directml-experiment\\Scripts\\activate
    pip install torch-directml coqui-tts
    python -m dubbing_poc.experiments.try_directml_xtts

Este script NO instala nada por sí mismo. Si falta torch_directml,
imprime instrucciones y termina. Cualquier falla durante la carga o
síntesis con DirectML se captura y reporta sin dejar el proceso en
estado raro — XTTS usa operadores de PyTorch que pueden no estar
implementados en el backend DirectML (NotImplementedError, kernel no
soportado, etc.), eso es un resultado esperado del experimento, no un
bug del script.
"""

import sys
import tempfile
import time
import traceback

TEST_TEXT = "Esta es una frase de prueba para medir el tiempo de síntesis de voz."
TEST_LANGUAGE = "es"
REPEATS = 3


def _time_synth(tts, speaker, out_dir, label):
    times = []
    for i in range(REPEATS):
        out_path = f"{out_dir}/synth_{label}_{i}.wav"
        start = time.perf_counter()
        tts.tts_to_file(text=TEST_TEXT, language=TEST_LANGUAGE, speaker=speaker, file_path=out_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  [{label}] intento {i + 1}/{REPEATS}: {elapsed:.2f}s")
    # Se descarta el primer intento (posible warm-up/JIT) si hay más de uno.
    usable = times[1:] if len(times) > 1 else times
    avg = sum(usable) / len(usable)
    return avg, times


def _load_xtts(device_obj, label):
    print(f"\nCargando XTTS-v2 en {label}...")
    start = time.perf_counter()
    try:
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device_obj)
    except Exception:
        print(f"FALLÓ la carga de XTTS en {label}:")
        traceback.print_exc(limit=3)
        return None, None
    elapsed = time.perf_counter() - start
    print(f"  Carga OK en {elapsed:.2f}s")
    return tts, elapsed


def main():
    try:
        import torch_directml
    except ImportError:
        print(
            "torch_directml no está instalado en este entorno.\n\n"
            "Este experimento requiere un virtualenv SEPARADO del "
            "entorno productivo (torch-directml suele necesitar una "
            "versión de torch distinta a la instalada aquí):\n\n"
            "  python -m venv .venv-directml-experiment\n"
            "  .venv-directml-experiment\\Scripts\\activate\n"
            "  pip install torch-directml coqui-tts\n"
            "  python -m dubbing_poc.experiments.try_directml_xtts\n"
        )
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="directml_experiment_") as tmpdir:
        cpu_tts, cpu_load_time = _load_xtts("cpu", "CPU")
        cpu_avg = cpu_times = None
        if cpu_tts is not None:
            speaker = (cpu_tts.speakers or ["Claribel Dervla"])[0]
            print(f"\nMidiendo síntesis en CPU (voz: {speaker})...")
            cpu_avg, cpu_times = _time_synth(cpu_tts, speaker, tmpdir, "cpu")

        dml_tts = dml_load_time = dml_avg = dml_times = None
        try:
            dml_device = torch_directml.device()
        except Exception:
            print("\nFALLÓ al obtener el dispositivo DirectML:")
            traceback.print_exc(limit=3)
            dml_device = None

        if dml_device is not None:
            dml_tts, dml_load_time = _load_xtts(dml_device, "DirectML")
            if dml_tts is not None:
                speaker = (dml_tts.speakers or ["Claribel Dervla"])[0]
                print(f"\nMidiendo síntesis en DirectML (voz: {speaker})...")
                try:
                    dml_avg, dml_times = _time_synth(dml_tts, speaker, tmpdir, "dml")
                except Exception:
                    print("FALLÓ la síntesis en DirectML:")
                    traceback.print_exc(limit=3)
                    dml_avg = None

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    if cpu_load_time is not None:
        print(f"Carga CPU:       {cpu_load_time:.2f}s")
    if dml_load_time is not None:
        print(f"Carga DirectML:  {dml_load_time:.2f}s")
    if cpu_avg is not None:
        print(f"Síntesis CPU (promedio):      {cpu_avg:.2f}s  (intentos: {['%.2f' % t for t in cpu_times]})")
    if dml_avg is not None:
        print(f"Síntesis DirectML (promedio): {dml_avg:.2f}s  (intentos: {['%.2f' % t for t in dml_times]})")

    if cpu_avg is not None and dml_avg is not None:
        speedup = cpu_avg / dml_avg
        print(f"\nSpeedup DirectML vs CPU: {speedup:.2f}x")
        if speedup > 1.5:
            print("CONCLUSIÓN: mejora significativa. Vale la pena evaluar integrar "
                  "DirectML como backend opcional en XTTSBackend (paso posterior, "
                  "fuera de este experimento).")
        else:
            print("CONCLUSIÓN: mejora marginal o nula. No se recomienda integrar "
                  "DirectML para XTTS en este momento.")
    elif cpu_avg is not None:
        print("\nCONCLUSIÓN: DirectML no funcionó en este entorno (ver errores "
              "arriba). Se descarta por ahora — XTTS sigue en CPU.")
    else:
        print("\nCONCLUSIÓN: no se pudo medir ni siquiera el baseline de CPU. "
              "Revisar la instalación de coqui-tts en este virtualenv.")


if __name__ == "__main__":
    main()
