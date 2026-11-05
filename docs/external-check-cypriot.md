# External check: Cypriot Greek spontaneous speech (pre-registration)

Status: **written and committed before any score on this dataset was
computed.** Implementation: `evaluation/external_check.py`.

## Purpose and its limit

Can a service calibrated on Greek Common Voice (read speech, standard Greek,
about 4 s clips) be deployed unchanged on **spontaneous Cypriot Greek** (about
27 s answers, another variety), without a **gross** increase in false accepts?

This is a check for gross failure only. The dataset has **10 speakers**, so 45
speaker pairs determine any false-accept rate. The threshold study (E1) showed
that 10 speakers cannot resolve FAR at the 1% level. This check therefore
cannot confirm or refute the threshold method, and is not reported as doing
so.

## Data

- Common Voice Spontaneous Speech 5.0, Cypriot Greek (`el-CY`): 1,221 clips,
  10 speakers. Only `ss-corpus-el-CY.tsv` rows with `votes ≥ 1` are used:
  the 1,217 validated clips listed in the dataset's README.
- Protocol: `evaluation/protocol.py` with every speaker in one split, seed
  `20261105`, 3 enrollment clips per speaker, up to 20 probes. Every voiceprint
  is scored against every probe.

## Frozen system (nothing is re-fitted on this dataset)

- Model: ECAPA-VoxCeleb.
- Cohort: the 158 Greek Common Voice cohort speakers of the threshold study.
- Thresholds: M3 (AS-norm, empirical quantile, α = 1%) fitted on the 118
  Greek Common Voice development speakers: wideband 2.2985, narrowband 1.6818.
- Scoring and threshold selection go through `VerificationService.decide`, and
  the channel comes from the service's own detector.
- For reference only: M1 (raw cosine) thresholds from the same speakers.

## Conditions

- `clean`: probes as recorded.
- `narrowband`: probes passed through `simulate_narrowband`.
- Enrollment is always as recorded.

## Outcome rule (fixed in advance)

For each condition, compute FAR on all non-target trials, with a 95%
speaker-level bootstrap interval (B = 1000, resampling the 10 model speakers).

| Outcome | Rule |
|---|---|
| **Gross failure** | lower bound of the FAR interval > 5% (5× the target) |
| **No gross failure detected** | upper bound ≤ 5% |
| **Inconclusive** | otherwise |

FRR, the M1 reference and the channel-detection counts are reported
descriptively. No threshold, cohort or protocol parameter will be changed
after scores are seen. A bootstrap over 10 clusters is itself unreliable, and
its intervals should be read as indicative.

## Results

Source: `results/external_check_cypriot.json`. 10 speakers, 200 target and
1,800 non-target trials. The system was frozen on Greek Common Voice.

| Condition | Method | FAR (target 1%) | FRR |
|---|---|---|---|
| clean | **M3** | **7.1%** [4.6–9.6] | 1.5% [0.0–4.5] |
| clean | M1 (reference) | 4.2% [2.2–6.3] | 1.5% [0.0–4.5] |
| narrowband | **M3** | **6.6%** [2.2–12.5] | 2.5% [0.5–6.0] |
| narrowband | M1 (reference) | 3.8% [0.7–8.6] | 6.5% [2.0–12.5] |

**Pre-registered outcome: inconclusive in both conditions.** On clean audio
the lower bound (4.6%) falls just short of the 5% gross-failure bound. On
narrowband the interval spans 2–12%.

Descriptive observations. These are not conclusions: the bootstrap over 10
speakers is itself unreliable.

- **The frozen calibration does not appear to hold here.** Point estimates
  are about 7× the 1% target, and on clean audio the interval stays well above
  1%. This is consistent with the study's first conclusion: recalibrate for
  each deployment domain. Dialect, speaking style (spontaneous vs. read) and
  clip length (about 27 s vs. 4 s) all changed at once, and the check cannot
  separate them.
- **AS-norm with an out-of-domain cohort did not help** (7.1% vs. 4.2% for raw
  scores; the intervals overlap). One plausible reason is that the cohort is
  short read Greek sentences, so the probe-side statistics are measured
  against the wrong reference. Hypothesis for future work: the cohort must
  match the deployment domain.
- **Genuine users are rejected less often** (FRR 1.5% vs. 4.3% on Common
  Voice), likely because long clips give more stable embeddings. The whole
  score distribution shifted, which is what breaks a fixed threshold.
- **Channel detection** classified 200/200 clean probes as wideband and
  200/200 simulated-telephone probes as narrowband, on data it had never seen.
