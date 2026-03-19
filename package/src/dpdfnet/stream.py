from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np

from .audio import ensure_sample_rate, make_stft_config, to_mono
from .models import DEFAULT_MODEL, resolve_model
from .onnx_backend import RuntimeModel, build_runtime_model, infer_win_len


class StreamEnhancer:
    """Process audio chunk-by-chunk while preserving RNN state across calls.

    Maintains an internal input buffer and overlap-add output buffer so that
    arbitrary chunk sizes can be passed without introducing boundary artefacts.

    The STFT is causal (``center=False``), giving a fixed latency of one model
    window (~20 ms at 16 kHz) before the first enhanced samples are returned.
    Subsequent calls return enhanced audio with a delay of one hop (~10 ms).

    .. note::
        This path uses a hand-rolled causal STFT / ISTFT and will **not**
        produce bit-identical output to :func:`dpdfnet.enhance`, which uses
        ``center=True`` reflection padding.  Both are algorithmically correct;
        the causal variant is necessary for real-time use.

    Args:
        model: Model name (default: ``"dpdfnet2"``).
        onnx_path: Optional path to a custom ONNX file; overrides *model*.
        verbose: Print model resolution / download progress.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        onnx_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> None:
        resolved = resolve_model(
            model=model,
            onnx_path=onnx_path,
            auto_download=True,
            verbose=verbose,
        )
        self._runtime: RuntimeModel = build_runtime_model(resolved.onnx_path)
        self._model_sr: int = resolved.info.sample_rate
        self._win_len: int = infer_win_len(self._runtime.session, self._model_sr)
        cfg = make_stft_config(self._win_len)
        self._hop_size: int = cfg.hop_size
        self._window: np.ndarray = cfg.window
        self._freq_bins: int = self._win_len // 2 + 1

        self._input_sr: Optional[int] = None
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset RNN state and internal buffers.

        Call this between independent audio segments (e.g. between speakers or
        separate recordings) so that the RNN hidden state does not leak across
        streams.
        """
        self._state: np.ndarray = self._runtime.init_state.copy()
        self._in_buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._out_buf: np.ndarray = np.zeros(self._win_len, dtype=np.float32)
        self._input_sr = None

    def process(
        self,
        chunk: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Enhance a chunk of audio, returning enhanced samples.

        Returns an empty array when not enough data has accumulated for the
        first model frame.  Thereafter each call returns enhanced audio whose
        length (at the input sample rate) is proportional to the number of
        complete model frames processed.

        Args:
            chunk: Float32 audio samples.  Stereo input is averaged to mono.
            sample_rate: Sample rate of *chunk*.  May be different from the
                model's native rate; resampling is handled internally.  Omit
                (or pass ``None``) to use the model's native sample rate.
                Must be consistent across calls on the same stream; call
                :meth:`reset` before switching sample rates.

        Returns:
            Enhanced float32 mono samples at *sample_rate* (or model SR if
            *sample_rate* was ``None``).  May have length 0.
        """
        chunk = to_mono(np.asarray(chunk, dtype=np.float32))
        if chunk.size == 0:
            return np.zeros(0, dtype=np.float32)

        sr_in = sample_rate if sample_rate is not None else self._model_sr

        if self._input_sr is None:
            self._input_sr = sr_in
        elif self._input_sr != sr_in:
            raise ValueError(
                f"Sample rate changed from {self._input_sr} to {sr_in} between "
                "process() calls.  Call reset() before processing a new stream."
            )

        chunk_model = ensure_sample_rate(chunk, sr_in, self._model_sr)
        self._in_buf = np.concatenate([self._in_buf, chunk_model])

        output_frames: list[np.ndarray] = []

        while len(self._in_buf) >= self._win_len:
            # --- Analysis STFT (causal / center=False) ---
            windowed = self._in_buf[: self._win_len] * self._window
            spec_complex = np.fft.rfft(windowed, n=self._win_len)
            spec_ri = np.stack(
                [spec_complex.real.astype(np.float32),
                 spec_complex.imag.astype(np.float32)],
                axis=-1,
            )
            spec_t = spec_ri[np.newaxis, np.newaxis, :, :]  # (1,1,freq_bins,2)

            # --- ONNX inference ---
            spec_e_t, self._state = self._runtime.session.run(
                [self._runtime.out_spec_name, self._runtime.out_state_name],
                {
                    self._runtime.in_spec_name: spec_t,
                    self._runtime.in_state_name: self._state,
                },
            )

            # --- Per-frame ISTFT + overlap-add ---
            ri = spec_e_t[0, 0]  # (freq_bins, 2)
            complex_frame = ri[:, 0] + 1j * ri[:, 1]
            time_frame = (
                np.fft.irfft(complex_frame, n=self._win_len) * self._window
            ).astype(np.float32)

            self._out_buf += time_frame

            # The Vorbis window satisfies COLA at 50% overlap:
            #   w[n]^2 + w[n+hop]^2 == 1  for all n
            # so the first hop_size samples are fully committed after each frame.
            committed = self._out_buf[: self._hop_size].copy()
            self._out_buf[: self._win_len - self._hop_size] = (
                self._out_buf[self._hop_size :]
            )
            self._out_buf[self._win_len - self._hop_size :] = 0.0

            output_frames.append(committed)
            self._in_buf = self._in_buf[self._hop_size :]

        if not output_frames:
            return np.zeros(0, dtype=np.float32)

        enhanced_model_sr = np.concatenate(output_frames)

        if sr_in != self._model_sr:
            return ensure_sample_rate(enhanced_model_sr, self._model_sr, sr_in)
        return enhanced_model_sr

    def flush(self) -> np.ndarray:
        """Drain any remaining buffered audio by zero-padding to a full frame.

        Call this at the end of a stream to retrieve enhanced audio for the
        last partial window.  The output length equals one hop interval at the
        most-recently-used sample rate (or the model's native rate if
        :meth:`process` was never called).

        Does **not** call :meth:`reset`; call that explicitly if you want to
        reuse the instance for a new stream.

        Returns:
            Enhanced float32 samples for the final partial window, or an empty
            array if the input buffer is already empty.
        """
        if self._in_buf.size == 0:
            return np.zeros(0, dtype=np.float32)

        remainder = len(self._in_buf)
        sr_in = self._input_sr or self._model_sr

        # Zero-pad to a full window so process() can handle one more frame.
        pad = np.zeros(self._win_len - remainder, dtype=np.float32)
        out = self.process(pad, sample_rate=self._model_sr)

        # Trim to only the samples that came from real (non-padded) input.
        # One hop's worth of real audio maps to one hop's worth of output.
        real_out = min(self._hop_size, len(out))
        trimmed = out[:real_out] if len(out) > 0 else out

        if sr_in != self._model_sr:
            trimmed = ensure_sample_rate(trimmed, self._model_sr, sr_in)

        return trimmed.astype(np.float32)
