import pytest
from fastapi.testclient import TestClient

from voice_verification.api import create_app

from .conftest import synth_voice, to_wav

AUTH = {"X-API-Key": "test-key"}
ALICE, BOB = 220.0, 700.0  # synthetic "speakers"


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def wav_files(pitch, n=2, seconds=4.0):
    return [
        ("files", (f"{i}.wav", to_wav(synth_voice(pitch, seconds, seed=i)), "audio/wav"))
        for i in range(n)
    ]


def verify(client, user, pitch, seconds=3.0, seed=99):
    wav = to_wav(synth_voice(pitch, seconds, seed=seed))
    return client.post(
        f"/v1/voiceprints/{user}/verify",
        headers=AUTH,
        files={"file": ("probe.wav", wav, "audio/wav")},
    )


def test_health_needs_no_auth(client):
    assert client.get("/health").json()["status"] == "ok"


def test_requires_api_key(client):
    assert client.get("/v1/voiceprints/alice").status_code == 401
    assert client.get("/v1/voiceprints/alice", headers={"X-API-Key": "wrong"}).status_code == 401


def test_enroll_then_verify_genuine_and_impostor(client):
    r = client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE))
    assert r.status_code == 201, r.text
    assert r.json()["samples_total"] == 2

    genuine = verify(client, "alice", ALICE).json()
    assert genuine["verified"] is True
    assert genuine["spoof_check"] == "none"

    impostor = verify(client, "alice", BOB).json()
    assert impostor["verified"] is False
    assert impostor["score"] < genuine["score"]


def test_status_and_delete(client):
    client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE))
    assert client.get("/v1/voiceprints/alice", headers=AUTH).json() == {
        "user_id": "alice",
        "enrolled": True,
        "samples": 2,
    }
    assert client.delete("/v1/voiceprints/alice", headers=AUTH).json()["deleted"] == 2
    assert verify(client, "alice", ALICE).status_code == 404


def test_verify_unknown_user_is_404(client):
    assert verify(client, "nobody", ALICE).status_code == 404


def test_first_enrollment_needs_enough_speech(client):
    r = client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE, n=1, seconds=3))
    assert r.status_code == 422
    assert "first enrollment" in r.json()["detail"]


def test_short_probe_rejected(client):
    client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE))
    assert verify(client, "alice", ALICE, seconds=1.0).status_code == 422


def test_enrollment_limit(client):
    client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE, n=2))
    r = client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE, n=2))
    assert r.status_code == 409


def test_invalid_user_id(client):
    assert verify(client, "bad id!", ALICE).status_code == 422


def test_upload_size_limit(client, service):
    service.settings.max_upload_bytes = 1000
    r = client.post("/v1/voiceprints/alice", headers=AUTH, files=wav_files(ALICE))
    assert r.status_code == 422
    assert "exceeds" in r.json()["detail"]
