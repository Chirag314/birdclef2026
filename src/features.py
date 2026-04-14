"""
features.py — audio loading and log-mel spectrogram extraction.

Design notes:
- Uses torchaudio for mel computation (fast, GPU-compatible, matches inference).
- load_audio uses soundfile (already installed) for OGG decoding.
- Resampling via scipy if sample rate doesn't match target (fallback).
- LogMelExtractor is stateless after __init__; safe to use across DataLoader workers.
"""

from typing import Optional

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 32000,
               offset: float = 0.0,
               duration: Optional[float] = None) -> np.ndarray:
    """
    Load an audio file to a mono float32 numpy array at target_sr.

    Args:
        path:       Path to OGG/WAV/FLAC file.
        target_sr:  Target sample rate in Hz.
        offset:     Start position in seconds.
        duration:   Number of seconds to read (None = full file).

    Returns:
        waveform: float32 numpy array of shape (samples,)
    """
    info = sf.info(path)
    sr = info.samplerate

    start_frame = int(offset * sr)
    if duration is not None:
        n_frames = int(duration * sr)
    else:
        n_frames = -1  # read to end

    audio, _ = sf.read(
        path,
        start=start_frame,
        frames=n_frames if n_frames > 0 else -1,
        dtype="float32",
        always_2d=True,
    )
    # Stereo → mono by averaging channels
    audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import torchaudio
            import torch
            x = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(x).squeeze(0).numpy()
        except ImportError:
            # Fallback: scipy resampling (lower quality but always available)
            import scipy.signal
            n_out = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, n_out).astype(np.float32)

    return audio.astype(np.float32)


def pad_or_trim(waveform: np.ndarray, target_samples: int,
                pad_mode: str = "constant") -> np.ndarray:
    """Pad with zeros or trim to exactly target_samples."""
    n = len(waveform)
    if n >= target_samples:
        return waveform[:target_samples]
    # Pad
    pad_width = target_samples - n
    return np.pad(waveform, (0, pad_width), mode=pad_mode)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def peak_normalize(waveform: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """
    Scale waveform so the peak absolute value equals target_peak.
    Silent audio (peak < 1e-6) is returned unchanged.
    """
    peak = np.abs(waveform).max()
    if peak > 1e-6:
        return waveform * (target_peak / peak)
    return waveform


def compute_rms_db(waveform: np.ndarray) -> float:
    """
    Compute RMS level in dBFS (dB relative to full scale).
    Returns -120.0 for silence.
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20.0 * np.log10(rms))


def is_silent(waveform: np.ndarray, threshold_db: float = -45.0) -> bool:
    """Return True if RMS level is below threshold_db (energy gate)."""
    return compute_rms_db(waveform) < threshold_db


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------

class LogMelExtractor:
    """
    Converts a raw waveform (numpy float32) to a log-mel spectrogram tensor.

    Output shape: (1, n_mels, time_frames)
    For a 5-second clip at 32 kHz with hop_length=320:
        time_frames = 32000 * 5 / 320 = 500

    Usage:
        extractor = LogMelExtractor(config)
        log_mel = extractor(waveform_np)  # returns torch.Tensor
    """

    def __init__(self, config: dict):
        self.sr = config["audio"]["sample_rate"]
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]
        self.n_mels = config["audio"]["n_mels"]
        self.fmin = config["audio"]["fmin"]
        self.fmax = config["audio"]["fmax"]
        self.power = config["audio"].get("power", 2.0)
        self.top_db = config["audio"].get("top_db", 80.0)
        self._transform = None  # lazy init (safe for DataLoader fork)

    def _init_transform(self):
        import torchaudio.transforms as T
        self._mel = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=self.power,
        )
        self._to_db = T.AmplitudeToDB(top_db=self.top_db)
        self._transform = True

    def __call__(self, waveform: np.ndarray):
        """
        Args:
            waveform: float32 numpy array, shape (samples,)
        Returns:
            log_mel: torch.Tensor, shape (1, n_mels, time_frames), float32
        """
        import torch
        if self._transform is None:
            self._init_transform()

        x = torch.from_numpy(waveform).unsqueeze(0)  # (1, samples)
        mel = self._mel(x)                            # (1, n_mels, T)
        log_mel = self._to_db(mel)                    # (1, n_mels, T)
        return log_mel

    @property
    def expected_frames(self) -> int:
        """Expected number of time frames for a standard 5-second window."""
        # floor((n_samples - n_fft) / hop_length) + 1  ≈ sr*5/hop
        return int(self.sr * 5 / self.hop_length)
