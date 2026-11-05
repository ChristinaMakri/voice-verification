"""Encryption of voiceprints at rest (AES-256-GCM).

The user id is bound as associated data, so a ciphertext copied onto another
user's row fails to decrypt instead of silently verifying the wrong person.
"""

import base64
import binascii
import os

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE_BYTES = 12


class VoiceprintCipher:
    def __init__(self, key_b64: str):
        try:
            key = base64.b64decode(key_b64, validate=True)
        except binascii.Error as exc:
            raise ValueError("encryption key is not valid base64") from exc
        if len(key) != 32:
            raise ValueError("encryption key must decode to exactly 32 bytes")
        self._aead = AESGCM(key)

    def encrypt(self, user_id: str, embedding: np.ndarray) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        plaintext = np.asarray(embedding, dtype="<f4").tobytes()
        return nonce + self._aead.encrypt(nonce, plaintext, user_id.encode())

    def decrypt(self, user_id: str, blob: bytes) -> np.ndarray:
        nonce, ciphertext = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
        plaintext = self._aead.decrypt(nonce, ciphertext, user_id.encode())
        return np.frombuffer(plaintext, dtype="<f4").astype(np.float32)


def generate_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


def main() -> None:
    print(generate_key())
