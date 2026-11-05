"""Evaluation pipeline on a tiny synthetic Common Voice-shaped dataset."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from evaluation.channels import add_noise, g711, mulaw_8bit, narrowband
from evaluation.embed import EmbeddingCache
from evaluation.manifest import load_commonvoice
from evaluation.protocol import ProtocolConfig, build_protocol
from evaluation.scoring import bootstrap, eer_rate, score_trials, summarise

from .conftest import FakeEmbedder, synth_voice

SR = 16000


@pytest.fixture
def cv_root(tmp_path):
    """Fake locale folder: 12 speakers (8 male, 4 female) x 8 clips, plus one with 2 clips."""
    root = tmp_path / "el"
    (root / "clips").mkdir(parents=True)
    rows, durs = [], []
    speakers = [
        (f"spk{i:02d}", "male_masculine" if i < 8 else "female_feminine", 4 + 1 * i)
        for i in range(12)
    ] + [("tiny", "male_masculine", 0)]
    for spk, gender, n in speakers:
        pitch = 150 + 37 * len(rows) % 900 if spk == "tiny" else 120 + 60 * int(spk[3:])
        for k in range(n if spk != "tiny" else 2):
            name = f"{spk}_{k}.mp3"  # WAV content; libsndfile detects the format from the header
            sf.write(root / "clips" / name, synth_voice(pitch, 3.0, seed=k), SR, format="WAV")
            rows.append(
                {
                    "client_id": spk,
                    "path": name,
                    "sentence_id": f"s{k}",
                    "gender": gender,
                    "age": "thirties",
                }
            )
            durs.append({"clip": name, "duration[ms]": 3000})
    pd.DataFrame(rows).to_csv(root / "validated.tsv", sep="\t", index=False)
    pd.DataFrame(durs).to_csv(root / "clip_durations.tsv", sep="\t", index=False)
    return root


def test_manifest_normalises_gender_and_duration(cv_root):
    m = load_commonvoice(cv_root)
    assert set(m.gender.dropna()) == {"male", "female"}
    assert (m.duration_s == 3.0).all()
    assert m.utt_id.is_unique


def test_protocol_is_speaker_disjoint_and_deterministic(cv_root):
    m = load_commonvoice(cv_root)
    p = build_protocol(m, ProtocolConfig(max_probes=3))
    assert "tiny" not in set(p.enroll.speaker)  # too few clips

    calib = set(p.enroll[p.enroll.split == "calib"].speaker)
    test = set(p.enroll[p.enroll.split == "test"].speaker)
    assert calib and test and not calib & test

    # stratified: both genders appear in both splits
    gender = m.groupby("speaker").gender.first()
    assert {gender[s] for s in calib} == {gender[s] for s in test} == {"male", "female"}

    # enrollment and probe clips never overlap, probes capped
    assert not set(p.enroll.utt_id) & set(p.probes.utt_id)
    assert p.probes.groupby("speaker").size().max() <= 3

    # trials never cross splits
    split_of = dict(zip(p.enroll.speaker, p.enroll.split, strict=False))
    assert (p.trials.model.map(split_of) == p.trials.split).all()
    assert (p.trials.probe_speaker.map(split_of) == p.trials.split).all()

    again = build_protocol(m, ProtocolConfig(max_probes=3))
    pd.testing.assert_frame_equal(p.trials, again.trials)


def test_end_to_end_separates_synthetic_speakers(cv_root, tmp_path):
    m = load_commonvoice(cv_root)
    p = build_protocol(m)
    cache = EmbeddingCache(tmp_path / "emb", FakeEmbedder())
    path_of = dict(zip(m.utt_id, m.path, strict=True))
    emb = cache.get("clean", [(u, path_of[u]) for u in m.utt_id])
    scored = score_trials(p, emb, emb)
    result = summarise(scored, n_boot=50)
    assert result["test_eer"]["value"] < 0.05
    assert result["test_score_means"]["target"] > result["test_score_means"]["nontarget"]

    # second call is served from the cache file
    cache2 = EmbeddingCache(tmp_path / "emb", FakeEmbedder())
    assert cache2._load("clean").keys() == emb.keys()


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    rows = []
    for m in range(30):
        rows += [{"model": m, "target": True, "score": rng.normal(0.6, 0.1)} for _ in range(5)]
        rows += [{"model": m, "target": False, "score": rng.normal(0.1, 0.1)} for _ in range(50)]
    iv = bootstrap(pd.DataFrame(rows), eer_rate, n_boot=200)
    assert iv.low <= iv.value <= iv.high
    assert iv.high - iv.low > 0


def _band_power(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return spec[(f >= lo) & (f < hi)].sum()


def test_narrowband_removes_energy_above_4khz():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    y = narrowband(x)
    assert y.shape == x.shape
    assert _band_power(y, 4500, 8000) < 1e-3 * _band_power(x, 4500, 8000)
    assert _band_power(y, 500, 3000) > 0.5 * _band_power(x, 500, 3000)


def test_mulaw_is_close_but_lossy():
    x = synth_voice(200, 1.0)
    y = mulaw_8bit(x)
    snr = 10 * np.log10(np.sum(x**2) / np.sum((x - y) ** 2))
    assert 25 < snr < 50
    assert g711(x).shape == x.shape


def test_add_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    x = synth_voice(200, 2.0)
    noise = rng.standard_normal(SR).astype(np.float32)
    y = add_noise(x, noise, snr_db=10, rng=rng)
    snr = 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2))
    assert snr == pytest.approx(10, abs=0.1)
