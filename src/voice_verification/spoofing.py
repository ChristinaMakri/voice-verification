"""Presentation-attack (replay / synthetic voice) detection.

Voice cloning makes a verification-only system easy to fool. This module
defines the extension point; no detector ships yet, and the API reports
``spoof_check: "none"`` so callers can see that nothing was checked.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpoofResult:
    is_bonafide: bool
    score: float | None
    detector: str


class SpoofDetector(Protocol):
    name: str

    def check(self, samples: np.ndarray, sample_rate: int) -> SpoofResult: ...


class NoSpoofDetector:
    name = "none"

    def check(self, samples: np.ndarray, sample_rate: int) -> SpoofResult:
        return SpoofResult(is_bonafide=True, score=None, detector=self.name)
