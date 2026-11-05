"""Calibrate the acceptance threshold on your own recordings.

Expected layout (one folder per speaker, any number of recordings each):

    data/
      speaker_a/ 001.wav 002.wav ...
      speaker_b/ 001.wav ...

Every same-speaker pair is a target trial, every cross-speaker pair a
non-target trial. Use recordings from the channel you will deploy on
(e.g. 8 kHz phone calls), and speakers that were not used for anything else.
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

from .audio import load_audio
from .metrics import compute_eer, error_rates, threshold_at_far

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg"}


def collect(data_dir: Path) -> dict[str, list[Path]]:
    speakers = {}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES)
        if files:
            speakers[d.name] = files
    return speakers


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--far", type=float, default=0.01, help="target false-accept rate")
    ap.add_argument("--max-nontarget", type=int, default=20000, help="subsample impostor trials")
    ap.add_argument("--cache-dir", default="./.models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    speakers = collect(args.data_dir)
    if len(speakers) < 2:
        sys.exit("need at least two speaker folders")

    from .embedding import EcapaEmbedder

    embedder = EcapaEmbedder(args.cache_dir)
    embs: list[tuple[str, np.ndarray]] = []
    for spk, files in speakers.items():
        for f in files:
            audio = load_audio(f.read_bytes(), embedder.sample_rate, max_seconds=600)
            embs.append((spk, embedder.embed(audio.samples)))
        print(f"embedded {spk}: {len(files)} files", file=sys.stderr)

    target, nontarget = [], []
    for (sa, ea), (sb, eb) in itertools.combinations(embs, 2):
        (target if sa == sb else nontarget).append(float(np.dot(ea, eb)))
    if not target:
        sys.exit("need at least one speaker with two or more recordings")

    rng = random.Random(args.seed)
    if len(nontarget) > args.max_nontarget:
        nontarget = rng.sample(nontarget, args.max_nontarget)

    t, n = np.array(target), np.array(nontarget)
    eer = compute_eer(t, n)
    at_far = error_rates(t, n, threshold_at_far(n, args.far))
    print(
        json.dumps(
            {
                "speakers": len(speakers),
                "recordings": len(embs),
                "target_trials": len(target),
                "nontarget_trials": len(nontarget),
                "eer": {"rate": round((eer.far + eer.frr) / 2, 4), "threshold": eer.threshold},
                f"at_far_{args.far}": {
                    "threshold": at_far.threshold,
                    "far": round(at_far.far, 4),
                    "frr": round(at_far.frr, 4),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
