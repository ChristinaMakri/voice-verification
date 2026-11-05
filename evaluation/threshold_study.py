"""Threshold study, implementing docs/threshold-study.md.

    python -m evaluation.threshold_study develop --cv-root <locale dir>
    python -m evaluation.threshold_study final   --cv-root <locale dir>

``develop`` touches only the calib (development) speakers and writes
results/threshold_study/{e1_e2,e3,decision}.json.
``final`` refuses to run until decision.json exists, then evaluates once on
the test (lockbox) speakers and writes results/threshold_study/final.json.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from voice_verification.scoring import asnorm, centroid, cohort_stats

from .embed import EmbeddingCache
from .manifest import load_commonvoice
from .protocol import Protocol, build_protocol
from .scoring import bootstrap

# --- pre-registered parameters (docs/threshold-study.md) --------------------
ALPHA = 0.01
ALPHA_DESCRIPTIVE = 0.001
R = 200
SEED = 7
N_EVAL = 38
N_FIT = (10, 20, 40, 60, 80)
B_FIT = 200
B_FINAL = 1000
K_COHORT = 50
SAFETY_MARGIN = 0.10
SIZE_TOLERANCE = 0.05
EPS = 1e-9  # pre-registered bounds are inclusive; absorb float rounding
CONDITIONS = ("clean", "narrowband")
METHODS = {  # id -> (score kind, rule)
    "M1": ("raw", "empirical"),
    "M2": ("raw", "conservative"),
    "M3": ("asnorm", "empirical"),
    "M4": ("asnorm", "conservative"),
}


# --- thresholds ---------------------------------------------------------------
def threshold_at_far(nontarget: np.ndarray, alpha: float) -> float:
    """Same result as voice_verification.metrics.threshold_at_far, in O(n)."""
    n = nontarget.size
    allowed = int(np.floor(alpha * n))
    if allowed >= n:
        return float(nontarget.min())
    k = n - 1 - allowed  # ascending index of the (allowed+1)-th largest score
    return float(np.nextafter(np.partition(nontarget, k)[k], np.inf))


def conservative_threshold(
    nontarget: np.ndarray, model: np.ndarray, alpha: float, rng: np.random.Generator
) -> float:
    """95th percentile of empirical thresholds over a speaker-level bootstrap."""
    order = np.argsort(model, kind="stable")
    _, starts = np.unique(model[order], return_index=True)
    groups = np.split(nontarget[order], starts[1:])
    th = np.empty(B_FIT)
    for b in range(B_FIT):
        pick = rng.integers(0, len(groups), size=len(groups))
        th[b] = threshold_at_far(np.concatenate([groups[i] for i in pick]), alpha)
    return float(np.quantile(th, 0.95))


# --- trials as flat arrays ------------------------------------------------------
@dataclass
class TrialArrays:
    speakers: np.ndarray  # local index -> speaker id (never written out)
    model: np.ndarray  # local speaker index per trial
    probe_speaker: np.ndarray
    target: np.ndarray
    scores: dict  # (condition, kind) -> scores per trial

    def mask(self, speakers_in: np.ndarray) -> np.ndarray:
        return speakers_in[self.model] & speakers_in[self.probe_speaker]


def build_trial_arrays(
    protocol: Protocol,
    split: str,
    enroll_emb: dict,
    probe_emb: dict,  # condition -> {utt_id: emb}
    cohort: np.ndarray,
) -> TrialArrays:
    t = protocol.trials[protocol.trials.split == split].reset_index(drop=True)
    speakers = np.array(sorted(set(t.model)))
    index = {s: i for i, s in enumerate(speakers)}
    models = {
        s: centroid([enroll_emb[u] for u in g.utt_id])
        for s, g in protocol.enroll.groupby("speaker")
    }

    model_mat = np.stack([models[s] for s in speakers])
    model_stats = cohort_stats(model_mat, cohort, K_COHORT)
    m_idx = t.model.map(index).to_numpy()

    probe_utts = np.array(sorted(set(t.probe_utt)))
    p_index = {u: i for i, u in enumerate(probe_utts)}
    p_idx = t.probe_utt.map(p_index).to_numpy()

    scores = {}
    for cond, table in probe_emb.items():
        probe_mat = np.stack([table[u] for u in probe_utts])
        raw = np.einsum("ij,ij->i", model_mat[m_idx], probe_mat[p_idx])
        scores[(cond, "raw")] = raw
        scores[(cond, "asnorm")] = asnorm(
            raw, model_stats[m_idx], cohort_stats(probe_mat, cohort, K_COHORT)[p_idx]
        )

    return TrialArrays(
        speakers=speakers,
        model=m_idx,
        probe_speaker=t.probe_speaker.map(index).to_numpy(),
        target=t.target.to_numpy(),
        scores=scores,
    )


def fit(
    tr: TrialArrays, scores: np.ndarray, speakers_in: np.ndarray, rule: str, alpha: float, rng
) -> float:
    m = tr.mask(speakers_in) & ~tr.target
    if rule == "empirical":
        return threshold_at_far(scores[m], alpha)
    return conservative_threshold(scores[m], tr.model[m], alpha, rng)


def rates(tr: TrialArrays, scores: np.ndarray, speakers_in: np.ndarray, threshold: float):
    m = tr.mask(speakers_in)
    tgt, s = tr.target[m], scores[m]
    return float(np.mean(s[~tgt] >= threshold)), float(np.mean(s[tgt] < threshold))


# --- development (calib speakers only) ----------------------------------------
def develop(tr: TrialArrays) -> pd.DataFrame:
    n_spk = tr.speakers.size
    m0 = {c: threshold_at_far(tr.scores[(c, "raw")][~tr.target], ALPHA) for c in CONDITIONS}

    rows = []
    for r in range(R):
        perm = np.random.default_rng([SEED, r]).permutation(n_spk)
        eval_in = np.zeros(n_spk, dtype=bool)
        eval_in[perm[:N_EVAL]] = True
        rest = perm[N_EVAL:]

        for cond in CONDITIONS:  # M0 reference: fitted on all development speakers
            far, frr = rates(tr, tr.scores[(cond, "raw")], eval_in, m0[cond])
            rows.append(
                dict(
                    exp="E1",
                    fit_cond=cond,
                    eval_cond=cond,
                    method="M0",
                    n=n_spk,
                    alpha=ALPHA,
                    rep=r,
                    threshold=m0[cond],
                    far=far,
                    frr=frr,
                )
            )

        for n in N_FIT:
            fit_in = np.zeros(n_spk, dtype=bool)
            fit_in[rest[:n]] = True  # nested: smaller fit sets are subsets of larger ones
            alphas = (ALPHA, ALPHA_DESCRIPTIVE) if n == max(N_FIT) else (ALPHA,)
            for alpha in alphas:
                for mi, (method, (kind, rule)) in enumerate(METHODS.items()):
                    for ci, fit_cond in enumerate(CONDITIONS):
                        rng = np.random.default_rng([SEED, r, n, mi, ci, int(alpha * 1e4)])
                        thr = fit(tr, tr.scores[(fit_cond, kind)], fit_in, rule, alpha, rng)
                        eval_conds = (
                            CONDITIONS if (n == max(N_FIT) and alpha == ALPHA) else (fit_cond,)
                        )
                        for eval_cond in eval_conds:
                            far, frr = rates(tr, tr.scores[(eval_cond, kind)], eval_in, thr)
                            rows.append(
                                dict(
                                    exp="E1" if eval_cond == fit_cond else "E3",
                                    fit_cond=fit_cond,
                                    eval_cond=eval_cond,
                                    method=method,
                                    n=n,
                                    alpha=alpha,
                                    rep=r,
                                    threshold=thr,
                                    far=far,
                                    frr=frr,
                                )
                            )
        if (r + 1) % 20 == 0:
            print(f"  repetition {r + 1}/{R}", file=sys.stderr)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        a = g.name[-1]  # alpha is the last grouping key
        return pd.Series(
            {
                "violation_rate": float(np.mean(g.far > a)),
                "bias_median_far_over_alpha": float(np.median(g.far) / a),
                "median_far": float(np.median(g.far)),
                "far_p10": float(np.quantile(g.far, 0.1)),
                "far_p90": float(np.quantile(g.far, 0.9)),
                "median_frr": float(np.median(g.frr)),
                "threshold_median": float(np.median(g.threshold)),
                "threshold_p10": float(np.quantile(g.threshold, 0.1)),
                "threshold_p90": float(np.quantile(g.threshold, 0.9)),
            }
        )

    keys = ["exp", "fit_cond", "eval_cond", "method", "n", "alpha"]
    return df.groupby(keys).apply(agg, include_groups=False).reset_index()


def decide(summary: pd.DataFrame) -> dict:
    """Apply the pre-registered decision rules."""
    s = summary[(summary.alpha == ALPHA)]
    n_max = max(N_FIT)

    def viol(method, fit_cond, eval_cond, n=n_max):
        row = s[
            (s.method == method)
            & (s.fit_cond == fit_cond)
            & (s.eval_cond == eval_cond)
            & (s.n == n)
        ]
        return float(row.violation_rate.iat[0])

    def frr(method, cond):
        row = s[
            (s.method == method) & (s.fit_cond == cond) & (s.eval_cond == cond) & (s.n == n_max)
        ]
        return float(row.median_frr.iat[0])

    m0 = {
        c: float(s[(s.method == "M0") & (s.fit_cond == c)].violation_rate.iat[0])
        for c in CONDITIONS
    }
    safety = {
        m: {c: viol(m, c, c) <= m0[c] + SAFETY_MARGIN + EPS for c in CONDITIONS} for m in METHODS
    }
    safe = [m for m in METHODS if all(safety[m].values())]
    cost = {m: float(np.mean([frr(m, c) for c in CONDITIONS])) for m in METHODS}

    decision = {
        "m0_violation_rate": m0,
        "violation_rate_n80": {m: {c: viol(m, c, c) for c in CONDITIONS} for m in METHODS},
        "meets_safety": safety,
        "mean_median_frr_n80": cost,
        "recommended": None,
    }
    if not safe:
        decision["note"] = "no method meets the safety criterion in both conditions"
        return decision

    rec = min(safe, key=lambda m: cost[m])
    decision["recommended"] = rec

    min_n = {}
    for c in CONDITIONS:
        ref = viol(rec, c, c)
        min_n[c] = next(n for n in N_FIT if abs(viol(rec, c, c, n) - ref) <= SIZE_TOLERANCE + EPS)
    decision["min_calibration_speakers"] = max(min_n.values())
    decision["min_calibration_speakers_by_condition"] = min_n

    transfer, needs_recalibration = {}, False
    for a in CONDITIONS:
        for b in CONDITIONS:
            if a != b:
                transfer[f"{a}->{b}"] = viol(rec, a, b)
                needs_recalibration |= transfer[f"{a}->{b}"] > m0[b] + SAFETY_MARGIN + EPS
    decision["transfer_violation_rate"] = transfer
    decision["recalibrate_per_channel"] = bool(needs_recalibration)
    return decision


# --- final (lockbox) -----------------------------------------------------------
def final(calib: TrialArrays, test: TrialArrays, recommended: str) -> dict:
    out = {}
    methods = sorted({"M1", recommended})
    all_calib = np.ones(calib.speakers.size, dtype=bool)
    for method in methods:
        kind, rule = METHODS[method]
        for fit_cond in CONDITIONS:
            rng = np.random.default_rng(
                [SEED, 999, list(METHODS).index(method), CONDITIONS.index(fit_cond)]
            )
            thr = fit(calib, calib.scores[(fit_cond, kind)], all_calib, rule, ALPHA, rng)
            for eval_cond in CONDITIONS:
                df = pd.DataFrame(
                    {
                        "model": test.model,
                        "target": test.target,
                        "score": test.scores[(eval_cond, kind)],
                    }
                )
                far = bootstrap(df, lambda t, n, th=thr: float(np.mean(n >= th)), B_FINAL)
                frr = bootstrap(df, lambda t, n, th=thr: float(np.mean(t < th)), B_FINAL)
                out[f"{method}:{fit_cond}->{eval_cond}"] = {
                    "threshold": round(thr, 4),
                    "far": far.as_dict(),
                    "frr": frr.as_dict(),
                }
    matched = [out[f"{recommended}:{c}->{c}"]["far"]["value"] <= ALPHA for c in CONDITIONS]
    out["confirmed"] = bool(all(matched))
    return out


# --- CLI ------------------------------------------------------------------------
def load(cv_root: Path, work_dir: Path):
    manifest = load_commonvoice(cv_root)
    protocol = build_protocol(manifest)
    from voice_verification.embedding import EcapaEmbedder

    cache = EmbeddingCache(work_dir / "embeddings", EcapaEmbedder(str(work_dir / "models")))
    path_of = dict(zip(manifest.utt_id, manifest.path, strict=True))

    counts = manifest.groupby("speaker").size()
    cohort_spk = counts[counts < protocol.config.n_enroll + protocol.config.min_probes].index
    cohort_rows = manifest[manifest.speaker.isin(cohort_spk)]
    cohort_emb = cache.get("clean", list(zip(cohort_rows.utt_id, cohort_rows.path, strict=True)))
    cohort = np.stack(
        [centroid([cohort_emb[u] for u in g.utt_id]) for _, g in cohort_rows.groupby("speaker")]
    )
    print(f"cohort: {cohort.shape[0]} speakers", file=sys.stderr)

    enroll = cache.get("clean", [(u, path_of[u]) for u in protocol.enroll.utt_id])
    probes = {
        c: cache.get(c, [(u, path_of[u]) for u in protocol.probes.utt_id]) for c in CONDITIONS
    }
    return protocol, enroll, probes, cohort


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["develop", "final"])
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/threshold_study"))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    decision_file = args.out_dir / "decision.json"

    if args.stage == "final" and not decision_file.exists():
        sys.exit("run `develop` first: the lockbox opens only after decision.json exists")

    protocol, enroll, probes, cohort = load(args.cv_root, args.work_dir)
    calib = build_trial_arrays(protocol, "calib", enroll, probes, cohort)

    if args.stage == "develop":
        raw = develop(calib)
        raw.to_csv(args.work_dir / "threshold_study_repetitions.csv", index=False)
        summary = summarise(raw)
        for exp, name in [("E1", "e1_e2"), ("E3", "e3")]:
            part = summary[summary.exp == exp]
            (args.out_dir / f"{name}.json").write_text(
                part.to_json(orient="records", indent=1) + "\n"
            )
        decision = decide(summary)
        decision_file.write_text(json.dumps(decision, indent=2) + "\n")
        print(json.dumps(decision, indent=2))
        return

    decision = json.loads(decision_file.read_text())
    if not decision.get("recommended"):
        sys.exit("decision.json recommends no method; the pre-registered rules stop here")
    test = build_trial_arrays(protocol, "test", enroll, probes, cohort)
    result = final(calib, test, decision["recommended"])
    (args.out_dir / "final.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
