"""
augment.py — audio augmentations for BirdCLEF 2026.

Phase 0: All augmentations are OFF by default. This file defines the full
         augmentation pipeline so Phase 1 can enable individual components
         by setting config flags — no code changes required.

Augmentation design principles (from EDA):
  P1 Must Have  — background_mixing, waveform_mixup (enabled in Phase 1)
  P2 High       — rare_species oversampling (handled in dataset/sampler)
  P3 Medium     — spec_augment, soft_saturation
  Excluded      — large pitch shift, time stretch, gaussian noise, CutMix

All augmentations operate on raw waveforms (numpy float32, shape (samples,))
and their corresponding label vectors (numpy float32, shape (n_classes,)).

The Augmenter.apply() method returns (augmented_waveform, augmented_label_vec).
"""

import numpy as np
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def background_mixing(waveform: np.ndarray,
                      background_pool: List[str],
                      snr_db_range: Tuple[float, float] = (-5.0, 10.0),
                      sr: int = 32000,
                      window_samples: int = 160000) -> np.ndarray:
    """
    Mix signal waveform with a random 5-second window from a Pantanal soundscape.

    SNR is drawn uniformly from snr_db_range.
    Background label is all-zeros (background carries no species label).

    Args:
        waveform:        Signal waveform (samples,)
        background_pool: List of soundscape file paths to sample from
        snr_db_range:    (min_snr_db, max_snr_db) for mixing level
        sr:              Sample rate (for loading if needed)
        window_samples:  Target length in samples

    Returns:
        Mixed waveform of same length as input.
    """
    import soundfile as sf
    from src.features import peak_normalize, pad_or_trim

    if not background_pool:
        return waveform

    # Sample a random background file
    bg_path = np.random.choice(background_pool)
    try:
        bg_info = sf.info(bg_path)
        bg_duration = bg_info.duration
        window_sec = window_samples / sr
        if bg_duration <= window_sec:
            offset = 0.0
        else:
            max_offset = bg_duration - window_sec
            offset = np.random.uniform(0.0, max_offset)

        bg_audio, bg_sr = sf.read(bg_path, dtype="float32", always_2d=True,
                                   start=int(offset * bg_info.samplerate),
                                   frames=int(window_sec * bg_info.samplerate))
        bg_audio = bg_audio.mean(axis=1)
        bg_audio = pad_or_trim(bg_audio, window_samples)
        bg_audio = peak_normalize(bg_audio, target_peak=0.9)
    except Exception:
        return waveform  # silently skip if background file fails

    # Compute current signal RMS
    sig_rms = np.sqrt(np.mean(waveform ** 2) + 1e-10)
    bg_rms = np.sqrt(np.mean(bg_audio ** 2) + 1e-10)

    # Apply target SNR
    snr_db = np.random.uniform(*snr_db_range)
    snr_linear = 10.0 ** (snr_db / 20.0)
    bg_scaled = bg_audio * (sig_rms / (snr_linear * bg_rms))

    mixed = waveform + bg_scaled
    # Re-normalise to avoid clipping
    peak = np.abs(mixed).max()
    if peak > 0.9:
        mixed = mixed * (0.9 / peak)
    return mixed.astype(np.float32)


def waveform_mixup(waveform1: np.ndarray, label1: np.ndarray,
                   waveform2: np.ndarray, label2: np.ndarray,
                   alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mix two waveforms and blend their label vectors (soft mixup).

    λ ~ Beta(alpha, alpha); mixed = λ*w1 + (1-λ)*w2; label = λ*l1 + (1-λ)*l2

    This trains the model to predict co-present species at the correct ratio,
    matching the 3–5 species per test window distribution.
    """
    lam = float(np.random.beta(alpha, alpha))
    mixed_waveform = lam * waveform1 + (1.0 - lam) * waveform2
    mixed_label = lam * label1 + (1.0 - lam) * label2
    return mixed_waveform.astype(np.float32), mixed_label.astype(np.float32)


def apply_spec_augment(log_mel: "torch.Tensor",
                       freq_mask_param: int = 13,
                       time_mask_param: int = 40,
                       n_freq_masks: int = 2,
                       n_time_masks: int = 2) -> "torch.Tensor":
    """
    Apply SpecAugment frequency and time masking to a log-mel tensor.

    freq_mask_param = 13 ≈ 10% of 128 mel bins (conservative).
    time_mask_param = 40 ≈ 8% of 500 time frames (conservative).

    Called after mel extraction, not on raw waveform.
    """
    import torchaudio.transforms as T
    x = log_mel.clone()
    for _ in range(n_freq_masks):
        x = T.FrequencyMasking(freq_mask_param=freq_mask_param)(x)
    for _ in range(n_time_masks):
        x = T.TimeMasking(time_mask_param=time_mask_param)(x)
    return x


def apply_soft_saturation(waveform: np.ndarray,
                           prob: float = 0.25,
                           drive: float = 3.0) -> np.ndarray:
    """
    Simulate soft clipping / tape saturation via tanh.
    Applied to ~25% of clips (matching the ~8% observed clipping rate with margin).

    drive: Amplification before tanh (higher = more saturation effect).
    """
    if np.random.random() > prob:
        return waveform
    return np.tanh(drive * waveform).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmenter — orchestrates the pipeline
# ---------------------------------------------------------------------------

class Augmenter:
    """
    Unified augmentation pipeline controlled by config.

    Phase 0: all augmentations disabled.
    Phase 1: background_mixing, waveform_mixup enabled.
    Phase 3: spec_augment, soft_saturation enabled.

    Usage:
        augmenter = Augmenter(config, background_pool=[...], window_df=train_df)
        waveform, label = augmenter(waveform, label)

    For spec_augment (post-mel), call augmenter.apply_spec(log_mel_tensor).
    """

    def __init__(self, config: dict,
                 background_pool: Optional[List[str]] = None,
                 window_df: Optional["pd.DataFrame"] = None):
        aug_cfg = config.get("augmentation", {})
        self.config = config
        self.background_pool = background_pool or []
        self.window_df = window_df  # used for mixup partner sampling

        # --- Phase 0 defaults (all off) ---
        bg_cfg = aug_cfg.get("background_noise", {})
        self.bg_enabled = bg_cfg.get("enabled", False)
        self.bg_snr_range = tuple(bg_cfg.get("snr_db_range", [-5.0, 10.0]))

        mx_cfg = aug_cfg.get("mixup", {})
        self.mixup_enabled = mx_cfg.get("enabled", False)
        self.mixup_alpha = float(mx_cfg.get("alpha", 0.4))
        self.mixup_prob = float(mx_cfg.get("prob", 0.5))

        sa_cfg = aug_cfg.get("spec_augment", {})
        self.spec_aug_enabled = sa_cfg.get("enabled", False)
        self.freq_mask_param = int(sa_cfg.get("freq_mask_param", 13))
        self.time_mask_param = int(sa_cfg.get("time_mask_param", 40))
        self.n_freq_masks = int(sa_cfg.get("n_freq_masks", 2))
        self.n_time_masks = int(sa_cfg.get("n_time_masks", 2))

        sat_cfg = aug_cfg.get("soft_saturation", {})
        self.saturation_enabled = sat_cfg.get("enabled", False)
        self.saturation_prob = float(sat_cfg.get("prob", 0.25))

        sr = config["audio"]["sample_rate"]
        win_sec = config["competition"]["window_seconds"]
        self.window_samples = int(sr * win_sec)
        self.sr = sr

    def __call__(self,
                 waveform: np.ndarray,
                 label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply waveform-level augmentations. Returns (waveform, label)."""

        # Soft saturation
        if self.saturation_enabled:
            waveform = apply_soft_saturation(waveform, prob=self.saturation_prob)

        # Background mixing (H1)
        if self.bg_enabled and self.background_pool:
            waveform = background_mixing(
                waveform, self.background_pool,
                snr_db_range=self.bg_snr_range,
                sr=self.sr,
                window_samples=self.window_samples,
            )

        # Waveform mixup (H2) — requires a second sample from the dataset
        if self.mixup_enabled and self.window_df is not None:
            if np.random.random() < self.mixup_prob:
                waveform, label = self._apply_mixup(waveform, label)

        return waveform, label

    def apply_spec(self, log_mel: "torch.Tensor") -> "torch.Tensor":
        """Apply spectrogram-level augmentations (call after mel extraction)."""
        if self.spec_aug_enabled:
            log_mel = apply_spec_augment(
                log_mel,
                freq_mask_param=self.freq_mask_param,
                time_mask_param=self.time_mask_param,
                n_freq_masks=self.n_freq_masks,
                n_time_masks=self.n_time_masks,
            )
        return log_mel

    def _apply_mixup(self,
                     waveform: np.ndarray,
                     label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random partner from window_df and mix."""
        from src.features import load_audio, pad_or_trim, peak_normalize
        idx2 = np.random.randint(0, len(self.window_df))
        row2 = self.window_df.iloc[idx2]
        try:
            w2 = load_audio(row2["filepath"], target_sr=self.sr,
                             offset=float(row2["start_sec"]),
                             duration=self.config["competition"]["window_seconds"])
            w2 = pad_or_trim(w2, self.window_samples)
            w2 = peak_normalize(w2)
            l2 = row2["label_vec"].copy()
            return waveform_mixup(waveform, label, w2, l2, alpha=self.mixup_alpha)
        except Exception:
            return waveform, label  # skip mixup on error
