"""
postprocess.py — inference-time post-processing for BirdCLEF 2026.

Site×Hour Prior
---------------
Species occurrence in soundscapes is strongly constrained by recording
site and hour of day.  A model trained on global focal clips has no
awareness of these factors.  Multiplying model predictions by an
empirical prior built from train_soundscapes_labels.csv suppresses
false positives that are spatially or temporally implausible.

The prior is a smoothed probability matrix P[site, hour, class] that
captures the observed relative frequency of each species across the 9
training sites and 24 hours.  At inference, each prediction window's
raw model output is element-wise multiplied by P[site, hour, :] before
the final ranking.

Building the prior
    from src.postprocess import SiteHourPrior
    prior = SiteHourPrior.build(
        soundscape_labels_csv="path/to/train_soundscapes_labels.csv",
        class_list=label_encoder_classes,   # list of 234 primary_label strings
    )
    prior.save("artifacts/site_hour_prior.npz")

Applying at inference
    prior = SiteHourPrior.load("artifacts/site_hour_prior.npz")
    # row_id looks like: BC2026_Test_0001_S05_20250227_010002_5
    scaled = prior.apply(raw_probs, row_ids)   # (N, 234) -> (N, 234)

Design notes
- Laplace smoothing (alpha=1.0) so unseen site/hour combinations fall
  back to the global class frequency rather than zero.
- The prior is applied multiplicatively then renormalised row-wise to
  [0, 1] so it acts as a soft mask, not a calibration.
- If a row_id cannot be parsed (unknown site or missing time field),
  a uniform prior (all ones) is used — no harm done.
- The prior is stored as float32 for inference efficiency.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Sequence


# Regex to extract site code and hour from BirdCLEF 2026 soundscape filenames.
# Matches both training and test patterns:
#   BC2026_Train_0039_S22_20211231_201500.ogg
#   BC2026_Test_0001_S05_20250227_010002_5     (row_id, no .ogg)
_FILENAME_RE = re.compile(r'_(S\d+)_\d{8}_(\d{2})\d{4}')


def _parse_site_hour(name: str):
    """Return (site_str, hour_int) from a filename or row_id, or (None, None)."""
    m = _FILENAME_RE.search(name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


class SiteHourPrior:
    """
    Empirical Site×Hour species occurrence prior built from
    train_soundscapes_labels.csv.

    Attributes
    ----------
    sites      : sorted list of site strings seen in training data
    hours      : sorted list of hour ints seen in training data (0-23)
    classes    : list of 234 primary_label strings in submission order
    matrix     : (n_sites, 24, n_classes) float32 — smoothed prior probabilities
    global_    : (n_classes,) float32 — fallback when site/hour unknown
    """

    def __init__(self,
                 sites: List[str],
                 hours: List[int],
                 classes: List[str],
                 matrix: np.ndarray,
                 global_: np.ndarray):
        self.sites   = sites
        self.hours   = hours
        self.classes = classes
        self.matrix  = matrix    # (n_sites, 24, n_classes)
        self.global_ = global_   # (n_classes,)
        self._site_idx = {s: i for i, s in enumerate(sites)}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls,
              soundscape_labels_csv: str,
              class_list: Sequence[str],
              alpha: float = 1.0) -> "SiteHourPrior":
        """
        Build the prior from train_soundscapes_labels.csv.

        Args:
            soundscape_labels_csv : path to the CSV file.
            class_list            : ordered list of 234 class labels matching
                                    the submission column order.
            alpha                 : Laplace smoothing pseudo-count (default 1.0).

        Returns:
            SiteHourPrior instance ready for saving or direct use.
        """
        df = pd.read_csv(soundscape_labels_csv)
        class_idx = {c: i for i, c in enumerate(class_list)}
        n_classes = len(class_list)

        # Parse site and hour
        df[["site", "hour"]] = df["filename"].apply(
            lambda f: pd.Series(_parse_site_hour(f))
        )
        df = df.dropna(subset=["site", "hour"])
        df["hour"] = df["hour"].astype(int)

        sites = sorted(df["site"].unique())
        n_sites = len(sites)
        site_idx = {s: i for i, s in enumerate(sites)}

        # Raw count matrix: (n_sites, 24, n_classes)
        counts = np.zeros((n_sites, 24, n_classes), dtype=np.float32)

        for _, row in df.iterrows():
            si = site_idx.get(row["site"])
            h  = int(row["hour"])
            if si is None or not (0 <= h < 24):
                continue
            # primary_label may be semicolon-separated multi-label
            for label in str(row["primary_label"]).split(";"):
                label = label.strip()
                ci = class_idx.get(label)
                if ci is not None:
                    counts[si, h, ci] += 1.0

        # Laplace smoothing: add alpha to every cell
        smoothed = counts + alpha

        # Normalise per (site, hour) independently so each row sums to 1
        # across classes — preserves relative species likelihood.
        row_sums = smoothed.sum(axis=-1, keepdims=True)   # (n_sites, 24, 1)
        matrix = (smoothed / row_sums).astype(np.float32)

        # Global fallback: marginal over sites and hours
        global_counts = counts.sum(axis=(0, 1)) + alpha
        global_ = (global_counts / global_counts.sum()).astype(np.float32)

        hours = list(range(24))

        prior = cls(sites=sites, hours=hours, classes=list(class_list),
                    matrix=matrix, global_=global_)

        # Log summary
        print(f"SiteHourPrior built:")
        print(f"  Sites: {sites}")
        print(f"  Classes with any soundscape observation: "
              f"{int((counts.sum(axis=(0,1)) > 0).sum())} / {n_classes}")
        top5 = np.argsort(global_counts)[-5:][::-1]
        print(f"  Top-5 global classes: "
              f"{[class_list[i] for i in top5]}")

        return prior

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to a compressed .npz file."""
        np.savez_compressed(
            path,
            matrix=self.matrix,
            global_=self.global_,
            sites=np.array(self.sites),
            classes=np.array(self.classes),
        )
        print(f"SiteHourPrior saved → {path}")

    @classmethod
    def load(cls, path: str) -> "SiteHourPrior":
        """Load from a .npz file."""
        d = np.load(path, allow_pickle=True)
        sites   = list(d["sites"])
        classes = list(d["classes"])
        hours   = list(range(24))
        return cls(
            sites=sites, hours=hours, classes=classes,
            matrix=d["matrix"].astype(np.float32),
            global_=d["global_"].astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Inference application
    # ------------------------------------------------------------------

    def apply(self,
              probs: np.ndarray,
              row_ids: Sequence[str],
              eps: float = 1e-7) -> np.ndarray:
        """
        Scale model probabilities by the site/hour prior.

        Args:
            probs   : (N, n_classes) float32 — raw sigmoid probabilities.
            row_ids : length-N sequence of submission row_id strings, e.g.
                      "BC2026_Test_0001_S05_20250227_010002_5".
            eps     : small value added before normalisation to avoid /0.

        Returns:
            (N, n_classes) float32 — scaled probabilities in [0, 1].
        """
        probs = np.array(probs, dtype=np.float32)
        out   = probs.copy()

        for i, rid in enumerate(row_ids):
            site, hour = _parse_site_hour(rid)
            si = self._site_idx.get(site)

            if si is not None and hour is not None and 0 <= hour < 24:
                prior_vec = self.matrix[si, hour]
            else:
                prior_vec = self.global_

            scaled = probs[i] * prior_vec
            # Normalise to [0, 1] by dividing by the maximum
            max_val = scaled.max()
            if max_val > eps:
                scaled = scaled / max_val
            out[i] = scaled

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame showing per-site, per-hour class counts."""
        rows = []
        for si, site in enumerate(self.sites):
            for h in range(24):
                n_active = int((self.matrix[si, h] > (1.0 / len(self.classes))).sum())
                rows.append({"site": site, "hour": h, "active_classes": n_active})
        return pd.DataFrame(rows)
