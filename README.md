# voice-verification

A small, self-hosted **1:1 speaker verification** service.

A user enrolls a few recordings of their voice. Later, when they claim an
identity (a customer id, an account number, or whatever your system uses), you
send a new recording and the service answers one question: *does this voice
match the voiceprint stored for that identity?*

```
            claimed id + recording
caller ───► your app / voice bot ─────────────► POST /v1/voiceprints/{id}/verify
                                                      │
                     detect channel (wideband / narrowband)   ECAPA-TDNN embedding (192-d)
                                                      │
              cosine vs. centroid of the enrolled embeddings → AS-norm against a cohort
                                                      │
                        compare with the threshold calibrated for that channel
                                                      │
                  { verified, score, threshold, channel, calibrated, spoof_check, ... }
```

> **Verification, not identification.** The service never searches the whole
> database for the closest voice (1:N). 1:N gets less accurate the more people
> you enroll. 1:1 verification against a claimed identity is how voice
> biometrics is used in practice.

## Status

Proof of concept. It does a useful job and is tested, but read
[Limitations](#limitations-and-safe-use) before you rely on it.

## Quick start

```bash
cp .env.example .env
# fill in VV_ENCRYPTION_KEY and VV_API_KEY:
python -c "import base64,os;print(base64.b64encode(os.urandom(32)).decode())"

docker compose up --build
```

The first start downloads the SpeechBrain model (~80 MB) into the `models`
volume.

**The service refuses to verify until it is calibrated** (`503 not
calibrated`). See [Calibrating the threshold](#calibrating-the-threshold).
For local experiments only, `VV_ALLOW_UNCALIBRATED=true` falls back to a raw
cosine threshold, which is unsafe.

```bash
# enroll: at least 8 s of speech in total, across one or more files
curl -H "X-API-Key: $VV_API_KEY" \
     -F files=@enroll1.wav -F files=@enroll2.wav \
     http://localhost:8000/v1/voiceprints/user-123

# verify
curl -H "X-API-Key: $VV_API_KEY" -F file=@probe.wav \
     http://localhost:8000/v1/voiceprints/user-123/verify
# {"user_id":"user-123","verified":true,"score":3.12,"threshold":2.30,
#  "score_type":"asnorm","raw_score":0.61,"channel":"wideband","calibrated":true,
#  "speech_seconds":4.2,"spoof_check":"none","spoof_bonafide":true}
```

Interactive API docs: <http://localhost:8000/docs>

## API

All `/v1` endpoints need the `X-API-Key` header.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/voiceprints/{user_id}` | Enroll. Multipart field `files`, one or more recordings. |
| `POST` | `/v1/voiceprints/{user_id}/verify` | Verify. Multipart field `file`. |
| `GET` | `/v1/voiceprints/{user_id}` | Is the user enrolled, and with how many samples. |
| `DELETE` | `/v1/voiceprints/{user_id}` | Erase every voiceprint of the user. |
| `GET` | `/health` | Liveness and calibration status (`calibrated`, `channels`), no auth. |

Errors: `401` bad or missing API key · `404` not enrolled · `409` per-user
sample limit reached · `422` bad user id, unreadable or too-long audio, or not
enough speech · `503` not calibrated, or no threshold calibrated for the
detected channel.

`/verify` returns the decision along with what it was based on: `score` (the
value compared with the threshold), `score_type` (`asnorm` or `raw`),
`raw_score` (cosine, for diagnostics), `channel` (detected), `threshold`
(the one used for that channel) and `calibrated`.

Accepted audio: anything libsndfile reads (WAV including μ-law/A-law, FLAC,
OGG). Audio is downmixed to mono and resampled to 16 kHz.

## How it works

1. **Embedding.** [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
   (SpeechBrain, trained on VoxCeleb) maps a recording to a 192-dim vector that
   captures *who* is speaking rather than *what* they say. Verification is
   text-independent, so the user does not have to repeat a passphrase.
2. **Enrollment.** Each recording becomes one embedding. The first enrollment
   needs `VV_MIN_ENROLL_SPEECH_S` (8 s) of speech in total. Recordings from
   different sessions and devices make a more robust voiceprint.
3. **Storage.** Only embeddings are stored, never audio. Each embedding is
   encrypted with AES-256-GCM, and the user id is bound as associated data, so
   a row copied onto another user fails to decrypt.
4. **Channel.** The probe is classified as `wideband` (microphone) or
   `narrowband` (telephone). A file recorded at ≤ 8 kHz is narrowband. Otherwise
   it is narrowband when less than −35 dB of its energy lies above 4 kHz.
5. **Score.** The probe embedding is compared by cosine similarity with the
   normalised mean of the user's embeddings. The score is then **AS-normalised**:
   each side is compared with a cohort of stranger voiceprints, and the score is
   expressed as how far it lies above what that voiceprint and that probe
   usually score.
6. **Decision.** `verified = score >= threshold[channel]`, with thresholds from
   the calibration file.

## Design decisions

Each choice, why it was made, and where the reason comes from. *Study* means
the [threshold study](docs/threshold-study.md). *Physics* means a property of
the signal, fixed before any data was measured. *Convention* means standard
practice we did not re-test.

| Decision | Why | Basis |
|---|---|---|
| **1:1 verification, never 1:N search** | 1:N error grows with the number of enrolled people; verifying a claimed identity is the use voice biometrics supports | Convention |
| **ECAPA-TDNN (SpeechBrain, VoxCeleb), used as is** | Strong open pretrained model, about 0.1 s per clip on CPU, no training data needed. Its 3% EER on Greek read speech shows it carries across languages. | Convention; checked in our baseline |
| **Text-independent** | No passphrase to remember or overhear; works on whatever the caller says | Convention |
| **Centroid of enrollment embeddings, cosine score** | The model is trained with an angular-margin loss, so the angle between embeddings is the natural comparison. Averaging several recordings smooths out a single bad one. | Convention |
| **≥ 8 s of enrollment speech, ≥ 2 s per probe** | Short audio gives noisy embeddings. These are conservative defaults. | Convention; *not studied here* |
| **Store only embeddings, AES-256-GCM, user id as associated data** | Voiceprints are biometric data. Binding the id means a row copied onto another user fails to decrypt instead of verifying the wrong person. | Security |
| **Pseudonymous user ids enforced** | The voiceprint table alone should not identify anyone | GDPR |
| **AS-norm before thresholding, K = 50** | Lowered FRR at equal FAR, and was the only method whose FAR held on unseen speakers (0.97% vs. 1.54%). K = 50 is a common choice and was not tuned. | Study (pre-registered, lockbox-confirmed) |
| **Cohort: separate speakers, encrypted, bound to the calibration by hash** | Cohort speakers must be neither users nor calibration speakers, or information leaks. A different cohort shifts every score, so the service refuses a cohort the calibration was not made with. | Study design; engineering |
| **Empirical quantile at the target FAR** | Chosen by the pre-registered rules: unbiased and cheapest in FRR. If you need a guarantee rather than an average, add a margin (the study's conservative rule). | Study |
| **One threshold per channel** | A narrowband-calibrated threshold used on wideband audio gave 2–5× the target FAR, silently | Study (E3) |
| **Channel detection: source rate ≤ 8 kHz, or < −35 dB above 4 kHz** | A phone line passes about 300–3400 Hz. Microphone speech keeps some energy above 4 kHz. The cut-off was fixed from that, then measured: 100% of simulated phone clips and 99.9% of G.711 clips were detected correctly. 5% of "clean" Common Voice clips were flagged, and turned out to be genuinely band-limited (99.9% of their energy below 3.4 kHz). | Physics; measured afterwards |
| **Warn below 40 calibration speakers** | With 10 speakers, 1 calibration in 10 exceeded 3× the target FAR. Gains were small beyond about 40. | Study (E1) |
| **Refuse to verify when uncalibrated (503)** | The old default of 0.25 accepted 18.5% of impostors and nothing showed it. A security control should fail loudly, not silently. | Study finding; engineering |
| **Tag voiceprints, cohort and calibration with the model id** | Embeddings from different models are not comparable. A mismatch is refused, not silently scored. | Engineering |
| **Report `spoof_check: "none"`** | No anti-spoofing detector ships yet. Saying so explicitly keeps callers from assuming one. | Engineering |

## Calibrating the threshold

**Do not use the default `VV_THRESHOLD=0.25` in production.** On Greek Common
Voice it accepted 18.5% of impostors. A threshold does not transfer between
datasets, channels or populations. The
[threshold study](docs/threshold-study.md) (pre-registered, with a held-out
lockbox) found:

| Finding | Practical rule |
|---|---|
| A threshold from another dataset can be far off (0.25 gave 18.5% FAR) | Calibrate on your own channel and users |
| AS-norm score normalisation lowered FRR at equal FAR, and was the only method whose FAR held on unseen speakers (0.97% vs. 1.54% for raw scores, target 1%) | Normalise scores before thresholding |
| A narrowband-calibrated threshold used on wideband audio gave 2–5× the target FAR | One threshold per channel; detect the channel |
| With 10 calibration speakers, 1 in 10 calibrations exceeded 3× the target FAR | Use at least about 40–60 speakers |

The service implements that method. Calibration takes three steps, and needs
recordings from **two separate groups of speakers**, neither of whom are users
of the service:

```bash
pip install '.[ecapa]'
export VV_ENCRYPTION_KEY=...          # the service key; cohort is encrypted with it

# 1. cohort: ≥ 100 "stranger" speakers, one folder each
vv-build-cohort cohort_data/ --out cohort.npz

# 2. calibration: ≥ 40 other speakers, one folder each, ideally from your channel.
#    The first 3 files per speaker (sorted) form the voiceprint; the rest are probes.
vv-calibrate calib_data/ --cohort cohort.npz --far 0.01 --out calibration.json
#    No phone recordings? --simulate-narrowband derives the narrowband threshold
#    from wideband ones (marked "simulated" in the file; real calls are better).

# 3. point the service at both files
VV_COHORT_PATH=cohort.npz VV_CALIBRATION_PATH=calibration.json
```

`vv-calibrate` prints the per-channel threshold, the FAR and FRR it gives on
the calibration speakers, and the EER. It warns when a channel has fewer than
40 speakers. At startup the service checks that the calibration, the cohort
and the model belong together, and refuses to start otherwise. Pick the
operating point for your risk: a lower `--far` means more genuine users fall
back to another check.

Need public test data in a specific language? [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets)
is CC0 and labels recordings by speaker (`client_id`). Downsample it to 8 kHz
to approximate a phone line.

## Evaluation

`evaluation/` measures how well the pipeline works on public data, and where
it breaks: clean vs. telephone-band audio, short probes, noise, and spoofing.

```bash
uv pip install -e '.[ecapa,eval]'
python -m evaluation.run \
    --cv-root data/commonvoice-el/cv-corpus-27.0-2026-09-11/el \
    --enroll-condition clean --probe-condition g711
```

**Protocol.** Speakers with at least 5 recordings are split 50/50 into a
*calibration* and a *test* set (speaker-disjoint, stratified by declared
gender, fixed seed). Each speaker contributes 3 enrollment recordings and up to
5 probes. Every voiceprint is scored against every probe in its split.
Thresholds are chosen on calibration speakers only and reported on test
speakers. Every rate comes with a 95% interval from a speaker-level bootstrap.

**Conditions** (`evaluation/channels.py`) are simulations. `narrowband` is a
300–3400 Hz telephone band at 8 kHz. `g711` adds μ-law coding on top of it.
Each one changes one factor while the speakers and trials stay the same.

### Data & licensing

Datasets are **not** in this repository and must not be added to it. The
Common Voice terms forbid re-hosting or re-sharing the data and forbid
attempting to determine speakers' identities. To reproduce the results,
download the dataset yourself from the
[Mozilla Data Collective](https://mozilladatacollective.com) (Common Voice
Scripted Speech 27.0 – Greek) into `data/`, which is git-ignored.

Only aggregate metrics are published (`results/`). Trial lists, speaker ids,
per-speaker results and embeddings are never published. Embeddings are
derived voiceprints of real people and stay in the local cache under
`data/work/`.

### What this evaluation can and cannot tell you

The Greek Common Voice subset gives 235 usable speakers. That is enough for
some questions and too few for others. We report only the first kind:

| Reliable | Not concluded (reported as a limitation only) |
|---|---|
| **Relative** effects of a condition (channel, duration, noise), because the same speakers and trials are compared | **Absolute** error rates for production. About 500 target trials per split leave intervals that are several points wide. |
| False-accept rate (≈59k impostor trials per split) | **Gender or age differences.** Only 22 speakers declare female, 19 of whom are usable, and about half declare nothing. The overall numbers mostly reflect male voices. |
| Whether a calibrated threshold transfers to unseen speakers | **Real phone networks.** Codecs are simulated. Packet loss, echo cancellation, AGC and handsets are not. |
| | **Conversational speech.** Clips are short read sentences (median ≈4 s), so results are likely optimistic for real calls. |

**External check** ([docs/external-check-cypriot.md](docs/external-check-cypriot.md)):
the system was frozen on Greek Common Voice and applied unchanged to 10 Cypriot
Greek speakers answering questions spontaneously (about 27 s clips). The
pre-registered outcome is *inconclusive*: 10 speakers are too few. FAR came out
at about 7% against a 1% target, which indicates that thresholds and the
AS-norm cohort must be re-built for each deployment domain. Channel detection
was 400/400 correct.

Known protocol caveats:

- Three enrollment clips give 7–20 s of audio, silence included, so a few
  speakers fall below the service's 8 s speech gate. The evaluation does not
  apply that gate.
- Common Voice audio is already MP3-compressed, so "clean" means lightly coded,
  not studio quality.
- A `client_id` is an account and may occasionally cover more than one person.
- About 5% of "clean" probe clips (53 of 1015, from 24 speakers) are
  genuinely band-limited, with 99.9% of their energy below 3.4 kHz, likely
  from headsets in call mode. The study's `clean` condition therefore
  contains a little telephone-like audio. The service's channel detector
  routes such clips to the narrowband threshold.

## Configuration

Every setting is an env var with the `VV_` prefix. See
[`src/voice_verification/config.py`](src/voice_verification/config.py) and
[`.env.example`](.env.example). The ones that decide security:

| Variable | Meaning |
|---|---|
| `VV_ENCRYPTION_KEY` | AES-256 key for voiceprints and cohort (`vv-keygen`). Keep it in a secret manager. |
| `VV_API_KEY` | Shared secret for the `X-API-Key` header |
| `VV_CALIBRATION_PATH` | `calibration.json` from `vv-calibrate`. Required for `/verify`. |
| `VV_COHORT_PATH` | `cohort.npz` from `vv-build-cohort`. Required when the calibration uses AS-norm. |
| `VV_ALLOW_UNCALIBRATED` | Development only: verify with the raw `VV_THRESHOLD`, which is unsafe |

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest                    # unit + API tests use synthetic audio and a fake embedder
docker compose up -d db   # optional: enables the Postgres integration test
VV_TEST_DATABASE_URL=postgresql://vv:vv@localhost:5432/voiceprints pytest
```

The test suite needs neither PyTorch nor real voices. All test audio is
synthetic.

## Limitations and safe use

- **Do not use the voice as the only factor.** Modern voice cloning can fool a
  speaker-verification model with a few seconds of someone's audio. Treat
  `verified` as one risk signal, combined with something the user knows or has
  (PIN, OTP), and step up when the stakes are high.
- **No anti-spoofing yet.** `spoofing.py` defines the interface, and responses
  report `spoof_check: "none"` until a presentation-attack detector (e.g. an
  AASIST model trained on ASVspoof) is plugged in.
- **Channel mismatch.** ECAPA-VoxCeleb was trained on wideband web video.
  Telephone codecs, noise and speakerphones lower accuracy. Calibrate on your
  own data, and consider fine-tuning on telephone speech.
- **Crude speech detection.** The energy gate rejects silence but not noise.
  Use a neural VAD (e.g. Silero) for real deployments.
- **No rate limiting or lockout.** Put the service behind a gateway that
  throttles repeated verify attempts per user id.
- **Model changes invalidate voiceprints.** Embeddings are tagged with the
  model id. A new model needs re-enrollment.

### Privacy (GDPR and similar laws)

A voiceprint used to identify someone is **biometric data**, a special
category under GDPR Article 9. Before you process real users:

- have a lawful basis, usually explicit opt-in consent, and offer a
  non-biometric alternative to anyone who declines;
- carry out a Data Protection Impact Assessment;
- use **pseudonymous user ids**, never national id numbers, phone numbers or
  names. The API enforces `[A-Za-z0-9_.-]{1,128}`;
- keep the encryption key out of the database (secret manager / KMS) and
  honour erasure requests via `DELETE`.

## License

MIT
