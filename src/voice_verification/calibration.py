"""Per-channel acceptance thresholds, calibrated on your own speakers.

The method is the one the threshold study recommended and confirmed on
held-out speakers (docs/threshold-study.md): AS-norm scores, empirical
quantile at the target false-accept rate, one threshold per channel.

    vv-calibrate calib_data/ --cohort cohort.npz --out calibration.json \\
        [--simulate-narrowband]

``calib_data/<speaker>/<recording>``: for each speaker the first ``n_enroll``
recordings (sorted by name) form the voiceprint and the rest are probes.
Use speakers who are neither users of the service nor in the cohort.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import load_audio
from .channel import CHANNELS, NARROWBAND, WIDEBAND, detect_channel, simulate_narrowband
from .cohort import file_sha256, load_cohort, speaker_folders
from .config import Settings
from .crypto import VoiceprintCipher
from .embedding import Embedder
from .metrics import compute_eer, error_rates, threshold_at_far
from .scoring import asnorm, centroid, cohort_stats

FORMAT = 1
DEFAULT_TOP_K = 50
# The study saw one calibration in ten exceed 3x the target FAR with 10
# speakers, and little further gain beyond about 40.
MIN_RECOMMENDED_SPEAKERS = 40


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Calibration:
    model_id: str
    score: str  # "asnorm" | "raw"
    target_far: float
    thresholds: dict[str, float]
    speakers: dict[str, int]
    sources: dict[str, str]  # channel -> "recorded" | "simulated" | "mixed"
    cohort_sha256: str | None = None
    cohort_top_k: int | None = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format: int = FORMAT

    def __post_init__(self):
        if self.score not in ("asnorm", "raw"):
            raise CalibrationError(f"unknown score type '{self.score}'")
        if not self.thresholds or not set(self.thresholds) <= set(CHANNELS):
            raise CalibrationError(f"thresholds must be keyed by {CHANNELS}")
        if self.score == "asnorm" and (not self.cohort_sha256 or not self.cohort_top_k):
            raise CalibrationError("asnorm calibration needs cohort_sha256 and cohort_top_k")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT:
            raise CalibrationError(f"unsupported calibration format {data.get('format')}")
        return cls(**data)


def load_scoring_config(
    settings: Settings, cipher: VoiceprintCipher, model_id: str
) -> tuple[np.ndarray | None, Calibration | None]:
    """Load calibration (and its cohort) for the service, checking they belong together."""
    if not settings.calibration_path:
        return None, None
    cal = Calibration.load(Path(settings.calibration_path))
    if cal.model_id != model_id:
        raise CalibrationError(f"calibration is for '{cal.model_id}', service uses '{model_id}'")
    if cal.score == "raw":
        return None, cal
    if not settings.cohort_path:
        raise CalibrationError("calibration uses AS-norm: set VV_COHORT_PATH")
    if file_sha256(Path(settings.cohort_path)) != cal.cohort_sha256:
        raise CalibrationError("cohort file differs from the one used for calibration")
    return load_cohort(Path(settings.cohort_path), model_id, cipher), cal


def calibrate(
    speakers: dict[str, list[Path]],
    embedder: Embedder,
    cohort: np.ndarray | None,
    cohort_sha256: str | None = None,
    target_far: float = 0.01,
    n_enroll: int = 3,
    max_files_per_speaker: int | None = 8,
    top_k: int = DEFAULT_TOP_K,
    simulate: bool = False,
) -> tuple[Calibration, dict]:
    """Return the calibration and a report with per-channel error rates."""
    sr = embedder.sample_rate
    models, probes = {}, []  # probes: (speaker, channel, source, embedding)
    for spk, files in speakers.items():
        files = files[:max_files_per_speaker] if max_files_per_speaker else files
        if len(files) <= n_enroll:
            continue
        audios = [load_audio(f.read_bytes(), sr, 600) for f in files]
        models[spk] = centroid([embedder.embed(a.samples) for a in audios[:n_enroll]])
        for a in audios[n_enroll:]:
            x = a.samples
            channel, _ = detect_channel(x, sr, a.source_sample_rate)
            probes.append((spk, channel, "recorded", embedder.embed(x)))
            if simulate and channel == WIDEBAND:
                probes.append(
                    (spk, NARROWBAND, "simulated", embedder.embed(simulate_narrowband(x, sr)))
                )
    if len(models) < 2:
        raise CalibrationError("need at least two speakers with more than n_enroll recordings")

    names = sorted(models)
    model_mat = np.stack([models[s] for s in names])
    probe_mat = np.stack([p[3] for p in probes])
    raw = probe_mat @ model_mat.T  # (n_probes, n_models)
    if cohort is not None:
        m_stats = cohort_stats(model_mat, cohort, top_k)
        p_stats = cohort_stats(probe_mat, cohort, top_k)
        p_rep = np.repeat(p_stats, len(names), axis=0)
        m_rep = np.tile(m_stats, (len(probes), 1))
        scores = asnorm(raw.ravel(), m_rep, p_rep).reshape(raw.shape)
    else:
        scores = raw
    target = np.array([[p[0] == s for s in names] for p in probes])

    thresholds, counts, sources, report = {}, {}, {}, {}
    for channel in CHANNELS:
        rows = [i for i, p in enumerate(probes) if p[1] == channel]
        spk = {probes[i][0] for i in rows}
        if len(spk) < 2:
            continue
        s, t = scores[rows], target[rows]
        tgt, non = s[t], s[~t]
        thr = threshold_at_far(non, target_far)
        op = error_rates(tgt, non, thr)
        eer = compute_eer(tgt, non)
        thresholds[channel] = round(thr, 6)
        counts[channel] = len(spk)
        kinds = {probes[i][2] for i in rows}
        sources[channel] = kinds.pop() if len(kinds) == 1 else "mixed"
        report[channel] = {
            "speakers": len(spk),
            "target_trials": int(tgt.size),
            "nontarget_trials": int(non.size),
            "threshold": round(thr, 4),
            "far_on_calibration": round(op.far, 4),
            "frr_on_calibration": round(op.frr, 4),
            "eer": round((eer.far + eer.frr) / 2, 4),
            "source": sources[channel],
        }
        if len(spk) < MIN_RECOMMENDED_SPEAKERS:
            report[channel]["warning"] = (
                f"only {len(spk)} speakers; at least {MIN_RECOMMENDED_SPEAKERS} recommended"
            )
    if not thresholds:
        raise CalibrationError("no channel had probes from at least two speakers")

    cal = Calibration(
        model_id=embedder.model_id,
        score="asnorm" if cohort is not None else "raw",
        target_far=target_far,
        thresholds=thresholds,
        speakers=counts,
        sources=sources,
        cohort_sha256=cohort_sha256 if cohort is not None else None,
        cohort_top_k=top_k if cohort is not None else None,
    )
    return cal, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--cohort", type=Path, help="cohort file from vv-build-cohort (recommended)")
    ap.add_argument("--out", type=Path, default=Path("calibration.json"))
    ap.add_argument("--far", type=float, default=0.01, help="target false-accept rate")
    ap.add_argument("--n-enroll", type=int, default=3)
    ap.add_argument("--max-files-per-speaker", type=int, default=8)
    ap.add_argument(
        "--simulate-narrowband",
        action="store_true",
        help="also derive a narrowband threshold from wideband recordings",
    )
    ap.add_argument("--cache-dir", default="./.models")
    args = ap.parse_args(argv)

    from .embedding import EcapaEmbedder

    embedder = EcapaEmbedder(args.cache_dir)
    cohort = sha = None
    if args.cohort:
        key = os.environ.get("VV_ENCRYPTION_KEY")
        if not key:
            sys.exit("set VV_ENCRYPTION_KEY to decrypt the cohort")
        cohort = load_cohort(args.cohort, embedder.model_id, VoiceprintCipher(key))
        sha = file_sha256(args.cohort)
    else:
        print(
            "warning: no --cohort; raw scores are less stable on new speakers "
            "(see docs/threshold-study.md)",
            file=sys.stderr,
        )

    cal, report = calibrate(
        speaker_folders(args.data_dir),
        embedder,
        cohort,
        sha,
        args.far,
        args.n_enroll,
        args.max_files_per_speaker,
        simulate=args.simulate_narrowband,
    )
    cal.save(args.out)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
