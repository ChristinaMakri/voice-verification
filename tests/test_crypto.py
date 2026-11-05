import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from voice_verification.crypto import VoiceprintCipher, generate_key


def test_roundtrip():
    cipher = VoiceprintCipher(generate_key())
    emb = np.random.default_rng(0).standard_normal(192).astype(np.float32)
    np.testing.assert_array_equal(cipher.decrypt("u1", cipher.encrypt("u1", emb)), emb)


def test_ciphertext_is_bound_to_user_id():
    cipher = VoiceprintCipher(generate_key())
    blob = cipher.encrypt("alice", np.ones(4, dtype=np.float32))
    with pytest.raises(InvalidTag):
        cipher.decrypt("mallory", blob)


def test_wrong_key_fails():
    blob = VoiceprintCipher(generate_key()).encrypt("u1", np.ones(4, dtype=np.float32))
    with pytest.raises(InvalidTag):
        VoiceprintCipher(generate_key()).decrypt("u1", blob)


@pytest.mark.parametrize("key", ["not base64!!", "c2hvcnQ="])
def test_bad_keys_rejected(key):
    with pytest.raises(ValueError):
        VoiceprintCipher(key)
