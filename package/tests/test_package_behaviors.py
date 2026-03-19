from __future__ import annotations

from pathlib import Path

import numpy as np
from urllib.error import HTTPError
from urllib.error import URLError

import pytest


def _fake_download_one(_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"ok")


def test_import_surface() -> None:
    import dpdfnet

    assert hasattr(dpdfnet, "enhance")
    assert hasattr(dpdfnet, "enhance_file")
    assert hasattr(dpdfnet, "StreamEnhancer")
    assert hasattr(dpdfnet, "available_models")
    assert hasattr(dpdfnet, "download")


def test_model_resolution_auto_download(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    monkeypatch.setattr(models, "_download_one", _fake_download_one)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    model_name = models.supported_models()[0]

    resolved = models.resolve_model(
        model=model_name,
        auto_download=True,
        verbose=False,
    )

    assert resolved.info.name == model_name
    assert resolved.onnx_path.is_file()


def test_download_all_matches_registry(tmp_path, monkeypatch) -> None:
    import dpdfnet
    from dpdfnet import models

    monkeypatch.setattr(models, "_download_one", _fake_download_one)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    downloaded = dpdfnet.download(quiet=True)

    assert isinstance(downloaded, dict)
    assert sorted(downloaded) == models.supported_models()

    for model_name in models.supported_models():
        info = models.MODEL_REGISTRY[model_name]
        assert (tmp_path / info.onnx_filename).is_file()


def test_cli_download_all_matches_registry(tmp_path, monkeypatch, capsys) -> None:
    from dpdfnet import cli, models

    monkeypatch.setattr(models, "_download_one", _fake_download_one)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))
    exit_code = cli.main(["download", "--quiet"])

    assert exit_code == 0
    output = capsys.readouterr().out
    for model_name in models.supported_models():
        assert f"- {model_name}:" in output


def test_enhance_progress_callback_reports_frame_progress(monkeypatch) -> None:
    from dpdfnet import api

    class _FakeSession:
        def run(self, _outputs, _inputs):
            return np.zeros((1, 1, 3, 2), dtype=np.float32), np.zeros((1,), dtype=np.float32)

    class _FakeRuntime:
        session = _FakeSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _ResolvedInfo:
        sample_rate = 16000

    class _ResolvedModel:
        info = _ResolvedInfo()
        onnx_path = Path("fake.onnx")

    monkeypatch.setattr(api, "resolve_model", lambda **_kwargs: _ResolvedModel())

    import dpdfnet.audio as audio_mod
    import dpdfnet.onnx_backend as backend_mod

    monkeypatch.setattr(audio_mod, "to_mono", lambda x: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "ensure_sample_rate", lambda x, _a, _b: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "make_stft_config", lambda _win: type("Cfg", (), {"win_len": 4})())
    monkeypatch.setattr(audio_mod, "preprocess_waveform", lambda _w, _cfg: np.zeros((1, 4, 3, 2), dtype=np.float32))
    monkeypatch.setattr(audio_mod, "postprocess_spec", lambda _spec, _cfg: np.zeros((8,), dtype=np.float32))
    monkeypatch.setattr(audio_mod, "fit_length", lambda x, _n: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(backend_mod, "build_runtime_model", lambda _onnx: _FakeRuntime())
    monkeypatch.setattr(backend_mod, "infer_win_len", lambda _session, _sr: 4)

    updates = []
    api.enhance(
        audio=np.zeros((8,), dtype=np.float32),
        sample_rate=16000,
        progress_callback=lambda done, total: updates.append((done, total)),
    )

    assert updates[0] == (0, 4)
    assert updates[-1] == (4, 4)
    assert len(updates) == 5


def test_model_download_http_error_reports_status(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    def _raise_http(url: str, _destination: Path) -> None:
        raise HTTPError(url=url, code=403, msg="Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(models, "_download_one", _raise_http)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match=r"HTTP 403"):
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=True,
            verbose=False,
        )


def test_model_download_url_error_reports_network_issue(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    def _raise_url(_url: str, _destination: Path) -> None:
        raise URLError("unreachable")

    monkeypatch.setattr(models, "_download_one", _raise_url)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match=r"Network error"):
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=True,
            verbose=False,
        )


def test_model_download_unwritable_dir_reports_path(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    model_dir = tmp_path / "blocked"
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(model_dir))

    def _raise_permission(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(models.tempfile, "mkstemp", _raise_permission)

    with pytest.raises(RuntimeError, match=r"not writable"):
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=True,
            verbose=False,
        )


def test_model_download_retries_transient_network_errors(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    attempts = {"count": 0}

    def _flaky_download(_url: str, destination: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise URLError("[Errno 104] Connection reset by peer")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ok")

    monkeypatch.setattr(models, "_download_one", _flaky_download)
    monkeypatch.setattr(models.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    model_name = models.supported_models()[0]
    info = models.MODEL_REGISTRY[model_name]
    resolved = models.resolve_model(
        model=model_name,
        auto_download=True,
        verbose=False,
    )

    assert resolved.onnx_path == (tmp_path / info.onnx_filename).resolve()
    assert resolved.onnx_path.is_file()
    assert attempts["count"] >= 3


def test_missing_model_message_does_not_reference_unsupported_cli_flags(tmp_path, monkeypatch) -> None:
    from dpdfnet import models

    monkeypatch.setenv("DPDFNET_MODEL_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        models.resolve_model(
            model=models.supported_models()[0],
            auto_download=False,
            verbose=False,
        )

    message = str(exc_info.value)
    assert "--onnx" not in message
    assert "onnx_path" in message


# ---------------------------------------------------------------------------
# StreamEnhancer tests
# ---------------------------------------------------------------------------

def _make_fake_stream_enhancer(monkeypatch, win_len: int = 8, model_sr: int = 16000):
    """Return a StreamEnhancer backed by a fake zero-output ONNX session."""
    from dpdfnet import stream as stream_mod

    freq_bins = win_len // 2 + 1

    class _FakeSession:
        def get_inputs(self):
            class _Spec:
                name = "spec"
                shape = (1, 1, freq_bins, 2)

            class _State:
                name = "state"

            return [_Spec(), _State()]

        def get_outputs(self):
            class _OutSpec:
                name = "out_spec"

            class _OutState:
                name = "out_state"

            return [_OutSpec(), _OutState()]

        def run(self, _outputs, _inputs):
            return (
                np.zeros((1, 1, freq_bins, 2), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
            )

    class _FakeRuntime:
        session = _FakeSession()
        init_state = np.zeros((1,), dtype=np.float32)
        in_spec_name = "spec"
        in_state_name = "state"
        out_spec_name = "out_spec"
        out_state_name = "out_state"

    class _ResolvedInfo:
        sample_rate = model_sr

    class _ResolvedModel:
        info = _ResolvedInfo()
        onnx_path = Path("fake.onnx")

    monkeypatch.setattr(stream_mod, "resolve_model", lambda **_kw: _ResolvedModel())
    monkeypatch.setattr(stream_mod, "build_runtime_model", lambda _p: _FakeRuntime())
    monkeypatch.setattr(stream_mod, "infer_win_len", lambda _s, _sr: win_len)

    from dpdfnet import StreamEnhancer

    return StreamEnhancer(model="dpdfnet2")


def test_stream_enhancer_buffers_small_chunks(monkeypatch) -> None:
    """Small chunks are buffered; output appears only after win_len samples arrive."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)  # hop_size=4
    # 3 samples < win_len=8 → nothing to process yet
    out = e.process(np.zeros(3, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0
    # 5 more → total 8 = win_len → one frame committed → hop_size=4 output samples
    out = e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)
    assert len(out) == 4


def test_stream_enhancer_misaligned_block_size(monkeypatch) -> None:
    """Chunk sizes not aligned to hop_size must not drop or duplicate samples.

    Simulates sounddevice with BLOCK_SIZE=512 at 48 kHz against a 16 kHz model
    by feeding 171-sample chunks (≈ 512 * 16000 / 48000) at model SR directly.
    """
    WIN = 320
    HOP = 160
    e = _make_fake_stream_enhancer(monkeypatch, win_len=WIN)

    total_input = int(16000 * 1.0)  # 1 second at model SR
    CHUNK = 171  # not a multiple of HOP

    outputs = []
    fed = 0
    while fed < total_input:
        n = min(CHUNK, total_input - fed)
        out = e.process(np.zeros(n, dtype=np.float32), sample_rate=16000)
        outputs.append(out)
        fed += n

    total_output = sum(len(o) for o in outputs)
    expected_frames = max(0, (total_input - WIN) // HOP + 1)
    assert total_output == expected_frames * HOP
    for o in outputs:
        assert o.dtype == np.float32


def test_stream_enhancer_reset_clears_state(monkeypatch) -> None:
    """After reset(), the buffer is empty and a full win_len is needed again."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)  # partial buffer
    e.reset()
    out = e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0


def test_stream_enhancer_flush_drains_remainder(monkeypatch) -> None:
    """flush() zero-pads the last partial window and returns enhanced samples."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)  # 5 in buf, no output
    out = e.flush()
    assert len(out) > 0
    assert out.dtype == np.float32


def test_stream_enhancer_flush_on_empty_buffer_returns_empty(monkeypatch) -> None:
    """flush() on a fresh (or fully-drained) instance returns an empty array."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    out = e.flush()
    assert len(out) == 0
    assert out.dtype == np.float32
