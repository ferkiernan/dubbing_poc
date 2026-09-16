"""Tests del salteo de videos cuyo archivo de salida ya existe: tanto en
modo single (_run_job) como en modo lote (_run_batch_job), si el output
esperado ya está en disco no se debe volver a procesar (ni llamar a
pipeline.run), y debe quedar registrado en el log / batch_results."""

from unittest.mock import patch

from dubbing_poc.webapp.app import Job, _run_batch_job, _run_job


def _base_opts():
    return {
        "voice": "en-m1",
        "language": "en",
        "tts_backend": "espeak",
        "asr_model_size": "medium",
        "tts_kwargs": {},
        "merge_sentences": False,
    }


def test_run_job_skips_when_output_already_exists(tmp_path):
    output_path = tmp_path / "existing_out.mp4"
    output_path.write_bytes(b"fake video bytes")
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake input")

    job = Job(id="job1")
    with patch("dubbing_poc.webapp.app.pipeline.run") as mock_run:
        _run_job(job, str(input_path), str(output_path), _base_opts(), "input.mp4")

    mock_run.assert_not_called()
    assert job.status == "done"
    assert job.skipped is True
    assert job.output_path == str(output_path)
    assert any("Salteado" in line for line in job.logs)


def test_run_job_processes_when_output_missing(tmp_path):
    output_path = tmp_path / "missing_out.mp4"
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake input")

    job = Job(id="job2")
    with patch("dubbing_poc.webapp.app.pipeline.run") as mock_run:
        mock_run.return_value.segments = []
        with patch("dubbing_poc.webapp.app._save_transcript", return_value="t.json"):
            _run_job(job, str(input_path), str(output_path), _base_opts(), "input.mp4")

    mock_run.assert_called_once()
    assert job.status == "done"
    assert job.skipped is False


def test_run_batch_job_skips_existing_and_continues(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    existing_input = src_dir / "already_done.mp4"
    existing_input.write_bytes(b"fake")
    pending_input = src_dir / "pending.mp4"
    pending_input.write_bytes(b"fake")

    opts = _base_opts()
    # El nombre de salida esperado debe coincidir con _output_filename.
    from dubbing_poc.webapp.app import _output_filename
    existing_output = out_dir / _output_filename("already_done", ".mp4", opts["voice"])
    existing_output.write_bytes(b"already rendered")

    job = Job(id="batch1")
    with patch("dubbing_poc.webapp.app.pipeline.run") as mock_run:
        mock_run.return_value.segments = []
        with patch("dubbing_poc.webapp.app._save_transcript", return_value="t.json"):
            _run_batch_job(job, [str(existing_input), str(pending_input)], str(out_dir), opts)

    # Solo se procesa el segundo (pending); el primero se saltea.
    assert mock_run.call_count == 1
    statuses = {r["file"]: r["status"] for r in job.batch_results}
    assert statuses["already_done.mp4"] == "skipped"
    assert statuses["pending.mp4"] == "done"
    assert any("Salteado" in line for line in job.logs)
    assert "salteados" in job.message
