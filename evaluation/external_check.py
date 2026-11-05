"""External gross-failure check, implementing docs/external-check-cypriot.md.

    python -m evaluation.external_check \\
        --cv-root data/commonvoice-el/cv-corpus-27.0-2026-09-11/el \\
        --sps-root data/commonvoice-el-cy-spontaneous/sps-corpus-5.0-2026-09-11-el-CY

The system is frozen on Greek Common Voice (cohort and M3 thresholds) and
applied unchanged to the spontaneous Cypriot speakers. Writes aggregate
metrics to results/external_check_cypriot.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from voice_verification.audio import load_audio
from voice_verification.calibration import Calibration
from voice_verification.channel import NARROWBAND, WIDEBAND, detect_channel
from voice_verification.config import Settings
from voice_verification.crypto import VoiceprintCipher, generate_key
from voice_verification.scoring import centroid
from voice_verification.service import VerificationService
from voice_verification.storage import InMemoryStore

from .channels import CONDITIONS
from .embed import EmbeddingCache
from .manifest import load_cv_spontaneous
from .protocol import ProtocolConfig, build_protocol
from .scoring import bootstrap
from .threshold_study import ALPHA, K_COHORT, build_trial_arrays, load, threshold_at_far

GROSS_FAR = 0.05
B = 1000
PROTOCOL = ProtocolConfig(n_enroll=3, min_probes=2, max_probes=20, calib_fraction=0.0)
PROBE_CONDITIONS = ("clean", "narrowband")


def frozen_thresholds(cv_root: Path, work_dir: Path):
    """M3 and M1 thresholds fitted on the 118 Greek Common Voice development speakers."""
    protocol, enroll, probes, cohort = load(cv_root, work_dir)
    calib = build_trial_arrays(protocol, "calib", enroll, probes, cohort)
    non = ~calib.target
    m3 = {
        WIDEBAND: threshold_at_far(calib.scores[("clean", "asnorm")][non], ALPHA),
        NARROWBAND: threshold_at_far(calib.scores[("narrowband", "asnorm")][non], ALPHA),
    }
    m1 = {
        WIDEBAND: threshold_at_far(calib.scores[("clean", "raw")][non], ALPHA),
        NARROWBAND: threshold_at_far(calib.scores[("narrowband", "raw")][non], ALPHA),
    }
    return cohort, m3, m1, int(calib.speakers.size)


def verdict(far_low: float, far_high: float) -> str:
    if far_low > GROSS_FAR:
        return "gross failure"
    if far_high <= GROSS_FAR:
        return "no gross failure detected"
    return "inconclusive"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--sps-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--out", type=Path, default=Path("results/external_check_cypriot.json"))
    args = ap.parse_args(argv)

    cohort, m3, m1, n_cal = frozen_thresholds(args.cv_root, args.work_dir)
    print(f"frozen M3 thresholds {m3}", file=sys.stderr)

    from voice_verification.embedding import EcapaEmbedder

    manifest = load_cv_spontaneous(args.sps_root)
    protocol = build_protocol(manifest, PROTOCOL)
    embedder = EcapaEmbedder(str(args.work_dir / "models"))
    cache = EmbeddingCache(args.work_dir / "embeddings_cy_sps", embedder)
    path_of = dict(zip(manifest.utt_id, manifest.path, strict=True))
    enroll_emb = cache.get("clean", [(u, path_of[u]) for u in protocol.enroll.utt_id])
    probe_emb = {
        c: cache.get(c, [(u, path_of[u]) for u in protocol.probes.utt_id]) for c in PROBE_CONDITIONS
    }

    # channel as the service would detect it
    channel = {c: {} for c in PROBE_CONDITIONS}
    for utt in protocol.probes.utt_id:
        a = load_audio(Path(path_of[utt]).read_bytes(), 16000, 600)
        for c in PROBE_CONDITIONS:
            src = a.source_sample_rate if c == "clean" else None
            channel[c][utt] = detect_channel(CONDITIONS[c](a.samples), 16000, src)[0]

    cal = Calibration(
        model_id=embedder.model_id,
        score="asnorm",
        target_far=ALPHA,
        thresholds=m3,
        speakers={WIDEBAND: n_cal, NARROWBAND: n_cal},
        sources={WIDEBAND: "recorded", NARROWBAND: "simulated"},
        cohort_sha256="frozen-greek-cv-cohort",
        cohort_top_k=K_COHORT,
    )
    svc = VerificationService(
        Settings(encryption_key=generate_key(), api_key="x"),
        embedder,
        InMemoryStore(),
        VoiceprintCipher(generate_key()),
        calibration=cal,
        cohort=cohort,
    )
    models = {
        s: centroid([enroll_emb[u] for u in g.utt_id])
        for s, g in protocol.enroll.groupby("speaker")
    }
    trials = protocol.trials

    result = {
        "dataset": manifest.dataset.iat[0],
        "frozen_on": "Greek Common Voice 27.0, 118 development speakers",
        "thresholds": {
            "M3": {k: round(v, 4) for k, v in m3.items()},
            "M1": {k: round(v, 4) for k, v in m1.items()},
        },
        "protocol": {
            **vars(PROTOCOL),
            "speakers": int(protocol.enroll.speaker.nunique()),
            "probes": int(len(protocol.probes)),
            "target_trials": int(trials.target.sum()),
            "nontarget_trials": int((~trials.target).sum()),
        },
        "gross_far_bound": GROSS_FAR,
        "conditions": {},
    }
    for c in PROBE_CONDITIONS:
        ch = [channel[c][u] for u in trials.probe_utt]
        decisions = [
            svc.decide(models[m], probe_emb[c][u], k)
            for m, u, k in zip(trials.model, trials.probe_utt, ch, strict=True)
        ]
        accept_m3 = np.array([d.score >= d.threshold for d in decisions], dtype=float)
        accept_m1 = np.array(
            [d.raw_score >= m1[k] for d, k in zip(decisions, ch, strict=True)], dtype=float
        )
        out = {
            "detected_channels": pd.Series([channel[c][u] for u in protocol.probes.utt_id])
            .value_counts()
            .to_dict()
        }
        for name, acc in (("M3", accept_m3), ("M1_reference", accept_m1)):
            df = pd.DataFrame({"model": trials.model, "target": trials.target, "score": acc})
            far = bootstrap(df, lambda t, n: float(np.mean(n)), B)
            frr = bootstrap(df, lambda t, n: float(1 - np.mean(t)), B)
            out[name] = {"far": far.as_dict(), "frr": frr.as_dict()}
        out["verdict_M3"] = verdict(out["M3"]["far"]["low"], out["M3"]["far"]["high"])
        result["conditions"][c] = out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
