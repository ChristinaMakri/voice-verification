"""Run one evaluation experiment end to end.

    python -m evaluation.run \\
        --cv-root data/commonvoice-el/cv-corpus-27.0-2026-09-11/el \\
        --enroll-condition clean --probe-condition g711

Writes aggregate metrics (no speaker or clip ids) to results/<name>.json.
"""

import argparse
import json
import sys
from pathlib import Path

from .channels import CONDITIONS
from .embed import EmbeddingCache
from .manifest import load_commonvoice
from .protocol import ProtocolConfig, build_protocol
from .scoring import score_trials, summarise


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--enroll-condition", default="clean", choices=sorted(CONDITIONS))
    ap.add_argument("--probe-condition", default="clean", choices=sorted(CONDITIONS))
    ap.add_argument("--name", help="result name (default: <enroll>__<probe>)")
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    manifest = load_commonvoice(args.cv_root)
    config = ProtocolConfig()
    protocol = build_protocol(manifest, config)
    print(json.dumps(protocol.summary()), file=sys.stderr)

    from voice_verification.embedding import EcapaEmbedder

    cache = EmbeddingCache(
        args.work_dir / "embeddings", EcapaEmbedder(str(args.work_dir / "models"))
    )
    path_of = dict(zip(manifest.utt_id, manifest.path, strict=True))
    enroll_emb = cache.get(args.enroll_condition, [(u, path_of[u]) for u in protocol.enroll.utt_id])
    probe_emb = cache.get(args.probe_condition, [(u, path_of[u]) for u in protocol.probes.utt_id])

    scored = score_trials(protocol, enroll_emb, probe_emb)
    result = {
        "dataset": manifest.dataset.iat[0],
        "model": cache.embedder.model_id,
        "enroll_condition": args.enroll_condition,
        "probe_condition": args.probe_condition,
        "protocol": {**vars(config), "splits": protocol.summary()},
        "metrics": summarise(scored, args.target_far, args.n_boot),
    }
    name = args.name or f"{args.enroll_condition}__{args.probe_condition}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
