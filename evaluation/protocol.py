"""Build a deterministic, speaker-disjoint evaluation protocol.

1. Keep speakers with at least ``n_enroll + min_probes`` recordings.
2. Split *speakers* (not recordings) into ``calib`` and ``test``, stratified by
   declared gender so the few speakers of the minority group land in both.
3. Per speaker, pick ``n_enroll`` recordings for the voiceprint and up to
   ``max_probes`` other recordings as probes. The cap stops a handful of very
   prolific speakers from dominating the error rates.
4. Within each split, every voiceprint is tried against every probe:
   same speaker -> target trial, different speaker -> non-target trial.

The calib split is used only to choose thresholds; the test split only to
report them. Nothing about the test speakers influences any decision.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtocolConfig:
    n_enroll: int = 3
    min_probes: int = 2
    max_probes: int = 5
    calib_fraction: float = 0.5
    seed: int = 20261105


@dataclass(frozen=True)
class Protocol:
    config: ProtocolConfig
    enroll: pd.DataFrame  # split, speaker, utt_id
    probes: pd.DataFrame  # split, speaker, utt_id
    trials: pd.DataFrame  # split, model, probe_utt, probe_speaker, target

    def summary(self) -> dict:
        out = {}
        for split, t in self.trials.groupby("split"):
            spk = self.enroll[self.enroll.split == split].speaker.nunique()
            out[split] = {
                "speakers": int(spk),
                "probes": int((self.probes.split == split).sum()),
                "target_trials": int(t.target.sum()),
                "nontarget_trials": int((~t.target).sum()),
            }
        return out


def split_speakers(speakers: pd.DataFrame, fraction: float, rng: np.random.Generator) -> dict:
    """``speakers`` has columns speaker, gender. Returns speaker -> 'calib' | 'test'."""
    assignment = {}
    strata = speakers["gender"].fillna("undeclared")
    for _, group in speakers.groupby(strata, sort=True):
        ids = group["speaker"].sort_values().to_numpy()
        rng.shuffle(ids)
        n_calib = int(round(len(ids) * fraction))
        assignment.update({s: "calib" for s in ids[:n_calib]})
        assignment.update({s: "test" for s in ids[n_calib:]})
    return assignment


def build_protocol(manifest: pd.DataFrame, config: ProtocolConfig | None = None) -> Protocol:
    config = config or ProtocolConfig()
    rng = np.random.default_rng(config.seed)
    counts = manifest.groupby("speaker").size()
    keep = counts[counts >= config.n_enroll + config.min_probes].index
    m = manifest[manifest.speaker.isin(keep)]

    speakers = m.groupby("speaker", as_index=False)["gender"].first()
    split_of = split_speakers(speakers, config.calib_fraction, rng)

    enroll_rows, probe_rows = [], []
    for speaker in sorted(keep):
        utts = m.loc[m.speaker == speaker, "utt_id"].sort_values().to_numpy()
        rng.shuffle(utts)
        split = split_of[speaker]
        enroll_rows += [(split, speaker, u) for u in utts[: config.n_enroll]]
        probe_utts = utts[config.n_enroll : config.n_enroll + config.max_probes]
        probe_rows += [(split, speaker, u) for u in probe_utts]

    enroll = pd.DataFrame(enroll_rows, columns=["split", "speaker", "utt_id"])
    probes = pd.DataFrame(probe_rows, columns=["split", "speaker", "utt_id"])

    trial_parts = []
    for split in ("calib", "test"):
        models = enroll.loc[enroll.split == split, "speaker"].unique()
        p = probes[probes.split == split]
        grid = pd.DataFrame(
            {
                "split": split,
                "model": np.repeat(models, len(p)),
                "probe_utt": np.tile(p.utt_id.to_numpy(), len(models)),
                "probe_speaker": np.tile(p.speaker.to_numpy(), len(models)),
            }
        )
        grid["target"] = grid.model == grid.probe_speaker
        trial_parts.append(grid)
    trials = pd.concat(trial_parts, ignore_index=True)
    return Protocol(config=config, enroll=enroll, probes=probes, trials=trials)
