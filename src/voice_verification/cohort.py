"""Score-normalisation cohort: build it from your own data, store it encrypted.

A cohort is one voiceprint per speaker of a set of "strangers". They are real
people's biometrics, so the file is encrypted with the service key and is
never shipped with the code. Build it from recordings you are licensed to use,
from speakers who are not your users and not in your calibration set.

    vv-build-cohort cohort_data/ --out cohort.npz      # one folder per speaker
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from .audio import load_audio
from .crypto import VoiceprintCipher
from .embedding import Embedder
from .scoring import centroid

FORMAT = 1
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


class CohortError(ValueError):
    pass


def _aad(model_id: str) -> str:
    return f"cohort:{model_id}"


def save_cohort(path: Path, cohort: np.ndarray, model_id: str, cipher: VoiceprintCipher) -> None:
    n, dim = cohort.shape
    blob = cipher.encrypt(_aad(model_id), cohort.astype("<f4").ravel())
    with open(path, "wb") as f:
        np.savez(
            f,
            format=FORMAT,
            model_id=model_id,
            n=n,
            dim=dim,
            blob=np.frombuffer(blob, dtype=np.uint8),
        )


def load_cohort(path: Path, model_id: str, cipher: VoiceprintCipher) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if int(z["format"]) != FORMAT:
            raise CohortError(f"unsupported cohort format {int(z['format'])}")
        stored_model = str(z["model_id"])
        if stored_model != model_id:
            raise CohortError(f"cohort was built with '{stored_model}', service uses '{model_id}'")
        n, dim, blob = int(z["n"]), int(z["dim"]), z["blob"].tobytes()
    return cipher.decrypt(_aad(model_id), blob).reshape(n, dim)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def speaker_folders(data_dir: Path) -> dict[str, list[Path]]:
    """``data_dir/<speaker>/<recording>`` -> {speaker: sorted recordings}."""
    out = {}
    for d in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES)
        if files:
            out[d.name] = files
    return out


def build_cohort(speakers: dict[str, list[Path]], embedder: Embedder) -> np.ndarray:
    rows = []
    for files in speakers.values():
        embs = [
            embedder.embed(load_audio(f.read_bytes(), embedder.sample_rate, 600).samples)
            for f in files
        ]
        rows.append(centroid(embs))
    return np.stack(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("cohort.npz"))
    ap.add_argument("--min-speakers", type=int, default=100)
    ap.add_argument("--cache-dir", default="./.models")
    args = ap.parse_args(argv)

    key = os.environ.get("VV_ENCRYPTION_KEY")
    if not key:
        sys.exit("set VV_ENCRYPTION_KEY: the cohort is encrypted with the service key")
    speakers = speaker_folders(args.data_dir)
    if len(speakers) < args.min_speakers:
        sys.exit(f"found {len(speakers)} speakers, need at least {args.min_speakers}")

    from .embedding import EcapaEmbedder

    embedder = EcapaEmbedder(args.cache_dir)
    cohort = build_cohort(speakers, embedder)
    save_cohort(args.out, cohort, embedder.model_id, VoiceprintCipher(key))
    print(f"wrote {args.out}: {cohort.shape[0]} speakers, sha256 {file_sha256(args.out)}")


if __name__ == "__main__":
    main()
