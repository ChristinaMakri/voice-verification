# Threshold study: design (pre-registration)

Status: **approved 2026-11-05**, before any of the experiments below were run.
Implementation: `evaluation/threshold_study.py`. Changes made after results are seen go in [Deviations](#deviations),
with the reason.

## Question

> Which way of setting the acceptance threshold keeps its false-accept promise
> on speakers it has never seen, and what does that cost in false rejects?

We are looking for a **procedure**, not a number. A threshold value is
specific to a model, a channel and a population. A procedure is what transfers
to another deployment.

## What is already known (disclosure)

Before this design was written, we ran three experiments on the full protocol
(`results/clean__*.json`). That means we have **already seen test-split
numbers** for one method (the empirical quantile, fitted on all 118
calibration speakers). The relevant finding: a target of FAR = 1% gave 1.5%
[1.3–1.8] on test speakers, and the service default of 0.25 gave 18.5% FAR on
clean audio. Nothing was tuned on those numbers. From here on the test split
is a lockbox (see [Final evaluation](#final-evaluation)).

## Data

- Greek Common Voice Scripted Speech 27.0, `validated.tsv`.
- Protocol from `evaluation/protocol.py`, unchanged: seed `20261105`, speakers
  with at least 5 clips, 3 enrollment clips, up to 5 probes, speaker-disjoint
  50/50 split stratified by declared gender.
  - **Development pool:** the 118 *calib* speakers. All method development
    happens here.
  - **Lockbox:** the 117 *test* speakers. Used once, at the end.
- **Cohort** for score normalisation: the 158 validated speakers with 1–4
  clips (336 clips). They are excluded from the protocol by its clip minimum,
  so they are disjoint from both the development pool and the lockbox. Each
  cohort speaker is represented by the centroid of their clip embeddings.
- Model: `speechbrain/spkrec-ecapa-voxceleb`, used as is.

## Conditions

Enrollment is always `clean`. Probes are `clean` or `narrowband`, which have
clearly different score distributions (target mean 0.64 vs. 0.39). `g711` is
left out because it behaved like `narrowband`, and every extra condition adds
comparisons.

## Methods

Each method takes the scored trials of a *fit* set of speakers and returns
one threshold. The design is 2 × 2 (score type × rule), plus one reference.

| ID | Scores | Rule |
|---|---|---|
| M1 | raw cosine | **Empirical:** the lowest threshold with FAR_fit ≤ α (`threshold_at_far`). |
| M2 | raw cosine | **Conservative:** bootstrap the fit speakers (B = 200), take the M1 threshold of each resample, and use the 95th percentile of those thresholds. |
| M3 | AS-norm | Empirical, on normalised scores |
| M4 | AS-norm | Conservative, on normalised scores |
| M0 | raw cosine | *Reference, not deployable:* M1 fitted on the whole development pool, eval speakers included. It shows how much the eval estimate itself fluctuates. |

**AS-norm** (adaptive symmetric normalisation), with K = 50:

```
s_norm = ½ · [ (s − μ_e) / σ_e  +  (s − μ_p) / σ_p ]
```

μ_e and σ_e are the mean and standard deviation of the K highest cosine scores
between the enrollment centroid and the cohort. μ_p and σ_p are the same for
the probe embedding. The cohort is always embedded in the `clean` condition:
a deployment has one cohort, not one per call.

## Resampling (development pool only)

The same splits are used for every method, so the comparisons are paired.

- R = 200 repetitions, seed `7`.
- In each repetition, 38 speakers are drawn as **eval** speakers. The fit set
  is a random subset of size n of the remaining 80.
- n ∈ {10, 20, 40, 60, 80}.
- Trials are the subset of the protocol grid whose model and probe speakers
  both belong to the set (fit or eval).

## Metrics (per repetition, on eval speakers)

- FAR_eval and FRR_eval at the fitted threshold.

Summarised over the 200 repetitions:

- **Violation rate:** the share of repetitions with FAR_eval > α. Primary
  safety metric. It also contains the noise of a 38-speaker eval set, which M0
  quantifies.
- **Bias:** median(FAR_eval) / α.
- **Cost:** median FRR_eval.
- **Spread:** the 10th–90th percentile of FAR_eval and of the threshold.

α = 1% is primary. α = 0.1% is reported **descriptively only**: with about 6k
impostor trials per eval set it rests on a few dozen errors, so we draw no
conclusion from it.

## Experiments

- **E1: calibration size.** All methods × all n × both conditions, with fit
  and eval in the same condition. Question: how many speakers does a
  calibration need before the violation rate and spread stop shrinking?
- **E2: method choice.** E1 at n = 80. Question: which rule meets the safety
  criterion at the lowest cost?
- **E3: cross-channel transfer.** n = 80. Fit on `clean`, eval on
  `narrowband`, and the reverse, for M1–M4. Question: does a threshold set on
  one channel keep its FAR on another, and does AS-norm change that?

## Decision rules (fixed in advance)

1. A method **meets the safety criterion** in a condition when its violation
   rate at n = 80 is at most 10 percentage points above M0's. M0 is the floor
   that eval-set noise alone produces.
2. Among the methods that meet it in **both** conditions, recommend the one
   with the lowest median FRR, averaged over the two conditions.
3. If none meets it in both, report the trade-off and recommend nothing.
4. E3 can only add a caveat to the recommendation ("recalibrate per
   channel"). It cannot change the choice from rule 2.
5. Calibration size: report the smallest n whose violation rate is within
   5 points of the n = 80 value for the recommended method. Recommend that
   size as a minimum.

## Final evaluation

Run once, after E1–E3 and the recommendation are written down.

- Fit the recommended method, and M1 for comparison, on all 118 development
  speakers. Evaluate on the 117 lockbox speakers, in both conditions and in
  both transfer directions.
- Report FAR and FRR with 95% speaker-bootstrap intervals (B = 1000).
- The recommendation is **confirmed** if the lockbox FAR point estimate is
  ≤ α in both matched conditions. Otherwise it is reported as **not
  confirmed**, and nothing is re-tuned.

## Out of scope

Noise, spoofing, fairness by gender or age, model comparison, and
same-channel phone enrollment. See the README limitations.

## Deviations

1. **Code fix, no design change.** The first `develop` run crashed in
   `summarise` before any summary was produced: pandas dropped the grouping
   column `alpha`. The fix reads alpha from the group key. Raw repetitions are
   now saved before summarising, and a unit test covers `summarise`. No result
   had been seen before the re-run.
2. **Weakness of the safety criterion, recorded, decision unchanged.** Rule 1
   ("at most 10 points above M0") turned out to test whether a method is
   *unbiased*, not whether it is *conservative*. M1 is M0 with fewer speakers,
   so it passes automatically. All four methods passed, and the choice was made
   on FRR alone. We keep the pre-registered decision. A deployment that needs a
   guarantee rather than an average should use an absolute criterion instead
   (e.g. violation rate ≤ 20%).

## Development results (calib speakers only)

Source: `results/threshold_study/{e1_e2,e3,decision}.json`. 200 repetitions,
38 eval speakers each, α = 1%.

**Decision (pre-registered rules): M3** (AS-norm + empirical quantile).
Minimum calibration size: 60 speakers. Recalibrate per channel: yes.

| n = 80 | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| violation rate, clean | 42% | 42% | 22.5% | 41.5% | 22% |
| violation rate, narrowband | 44.5% | 45% | 19% | 40.5% | 19% |
| median FAR, clean | 0.96% | 0.94% | 0.79% | 0.95% | 0.79% |
| median FRR, clean | 7.8% | 7.7% | 8.0% | **5.6%** | 6.4% |
| median FRR, narrowband | 31.4% | 32.4% | 37.3% | **30.5%** | 34.7% |

Findings:

- **Eval noise is large.** Even M0, with a median FAR of 0.96%, exceeds 1% in
  42–45% of the 38-speaker eval sets.
- **AS-norm improves discrimination.** At the same median FAR, FRR falls from
  7.7% to 5.6% on clean audio and from 32.4% to 30.5% on narrowband, and it is
  lower at every n.
- **The conservative rule** halves the violation rate. It costs +0.3 FRR points
  on clean audio but +4.9 on narrowband, where many genuine scores sit near the
  threshold.
- **Calibration size.** The 90th-percentile FAR of M1 on clean audio is 3.3%
  at n = 10, 1.9% at n = 20, 1.5% at n = 40, and 1.35% at n = 80. Gains are
  small beyond about 40 speakers. Violation rates carry about ±3.5 points of
  repetition noise, so rule 5's choice of 60 over 40 is not strongly supported.
- **Cross-channel transfer is asymmetric.**
  - A clean-calibrated threshold used on narrowband gives FAR 0.15% and FRR 56%:
    safe but unusable.
  - A narrowband-calibrated threshold used on clean gives **FAR 3.3%** (M1),
    above target in 100% of repetitions. AS-norm reduces this to 1.9% (M3) or
    1.5% (M4) but does not fix it. A single threshold across channels is
    unsafe.
- **α = 0.1%** (descriptive only): narrowband FRR is 64–73%.

*Exploratory, not a conclusion:* on clean audio M4 has both a lower violation
rate (22% vs. 42%) and a lower FRR (6.4% vs. 7.7%) than M1. This was noticed
after looking at the results, so it needs new data to confirm and does not
replace the pre-registered choice.

Before the lockbox was opened: because M3 is unbiased rather than
conservative, a lockbox FAR slightly above 1% in one condition is quite likely
even if the method is sound (compare M0). The confirmation rule stands as
written.

## Final (lockbox) results

Run once. Thresholds were fitted on all 118 development speakers and
evaluated on the 117 lockbox speakers (`results/threshold_study/final.json`).
Values in brackets are 95% speaker-bootstrap intervals.

| Method | Fitted on → used on | FAR (target 1%) | FRR |
|---|---|---|---|
| **M3** | clean → clean | **0.97%** [0.86–1.09] | 4.3% [2.0–7.5] |
| **M3** | narrowband → narrowband | **0.90%** [0.64–1.21] | 39.8% [34.4–46.4] |
| M1 | clean → clean | 1.54% [1.31–1.80] | 4.5% [2.2–7.8] |
| M1 | narrowband → narrowband | 1.52% [1.04–2.13] | 36.5% [30.8–43.0] |
| M3 | narrowband → clean | 2.04% [1.87–2.22] | 3.2% [1.2–5.9] |
| M1 | narrowband → clean | **4.83%** [4.25–5.44] | 2.4% [0.6–5.1] |
| M3 | clean → narrowband | 0.38% [0.26–0.52] | 49.3% [43.1–56.2] |
| M1 | clean → narrowband | 0.36% [0.20–0.62] | 58.0% [52.3–64.3] |

**The recommendation is confirmed:** M3's lockbox FAR is ≤ 1% in both matched
conditions.

Reading the results:

- **M1 does not hold its promise on new speakers.** It overshoots to about
  1.5% in both channels, and the interval excludes 1% on clean audio. The
  lockbox speakers are slightly more confusable than the development speakers.
  Resampling *inside* the development pool could not reveal that shift. AS-norm
  absorbed it: normalising each score by how the voiceprint and probe behave
  against a cohort makes the threshold less dependent on which population it
  was fitted on.
- On clean audio, M3 reaches a lower FAR than M1 (0.97% vs. 1.54%) at the same
  FRR (about 4.4%), so it is strictly better. On narrowband the two sit at
  different FARs, so their FRRs are not directly comparable.
- **Cross-channel use is unsafe, more so than in development.** A
  narrowband-calibrated threshold used on clean audio gives 4.8% FAR with M1
  and 2.0% with M3.
- One lockbox, about 500 target trials: absolute values carry the intervals
  shown, and the narrowband FAR interval includes 1%.

## Conclusions

1. **Calibrate on the deployment's own channel and population.** A threshold
   from another dataset can be badly wrong: the service default of 0.25 gave
   18.5% FAR here.
2. **Use score normalisation (AS-norm) before thresholding.** In this study it
   lowered FRR at equal FAR, and it was the only method whose FAR held on
   unseen speakers.
3. **Use one threshold per channel, and detect the channel.** Applying a
   threshold across channels silently multiplies false accepts (here up to
   about 5×).
4. **Calibrate with at least about 40–60 speakers.** With 10, one calibration
   in ten ends up with FAR over 3× the target.
5. If the target must be a guarantee rather than an average, add a
   conservative margin (the M2/M4 rule), and expect it to cost more on
   telephone audio.

Scope: one model (ECAPA-VoxCeleb), one language and dataset (Greek Common
Voice, read speech), simulated telephone channels, α = 1%. See the README
limitations.
