"""
Mission link security layer for Project C.R.A.T.E.R.

Design rationale (for the jury / Q&A):
  - AEAD (Authenticated Encryption with Associated Data), not plain
    encryption: a tampered or corrupted ciphertext must be REJECTED, not
    silently decrypted into garbage that then (maybe) fails CRC a layer up.
    ChaCha20-Poly1305 gives us confidentiality and integrity as one
    property in one call, which is the real-world standard for securing a
    telemetry link, not two loosely-stacked mechanisms.
  - This sits OUTSIDE the telemetry/FEC layer, not inside it: every
    already-built wire packet (FULL, DELTA, or PARITY -- CRC8 already
    applied) is treated as an opaque blob and wrapped. FEC's XOR parity
    math still operates on *plaintext* packet bytes before this layer ever
    sees them, so encryption adds zero complexity to the resilience logic
    and the two concerns stay cleanly separated: CRC8 = corruption
    detection, FEC = loss recovery, this layer = confidentiality +
    tamper-rejection. Each one independently explainable.
  - Key handling: a real mission provisions a pre-shared mission key through
    a secure ground-based key-exchange process before launch -- there is no
    "login" on a one-way space link. For this demo, the key is derived
    locally (HKDF over a fixed passphrase) so the demo is reproducible
    without a secrets file; swapping that one function for a real
    provisioned key is the only change a flight version would need.
"""
from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

NONCE_LEN = 12  # ChaCha20-Poly1305 standard nonce size
TAG_LEN = 16    # Poly1305 authentication tag, appended by the library itself


class SecurityError(ValueError):
    """Raised when a packet fails authentication (corrupted or tampered)."""


def derive_mission_key(passphrase: str = "CRATER-ATEC26-MISSION-KEY-DEMO",
                        salt: bytes = b"crater-lunar-rover-v1") -> bytes:
    """Derives a 256-bit key via HKDF-SHA256. In a flight system this would
    instead be a key provisioned through a secure pre-launch exchange, not
    a passphrase baked into source -- flagged here deliberately rather than
    hidden, since judges should be able to ask about this directly."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"crater-telemetry-link")
    return hkdf.derive(passphrase.encode("utf-8"))


class MissionCrypto:
    """Wraps every wire packet with ChaCha20-Poly1305 AEAD. Stateless aside
    from the key -- each packet carries its own random nonce, so packet
    reordering/loss on the lossy link (the whole point of this project)
    never breaks decryption of the packets that do arrive."""

    def __init__(self, key: bytes | None = None):
        self.key = key if key is not None else derive_mission_key()
        self._aead = ChaCha20Poly1305(self.key)
        self.encrypt_count = 0
        self.decrypt_ok_count = 0
        self.decrypt_fail_count = 0

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(NONCE_LEN)
        ciphertext = self._aead.encrypt(nonce, plaintext, None)
        self.encrypt_count += 1
        return nonce + ciphertext

    def decrypt(self, wire_bytes: bytes) -> bytes:
        if len(wire_bytes) < NONCE_LEN + TAG_LEN:
            self.decrypt_fail_count += 1
            raise SecurityError("packet too short to contain nonce + auth tag")
        nonce, ciphertext = wire_bytes[:NONCE_LEN], wire_bytes[NONCE_LEN:]
        try:
            plaintext = self._aead.decrypt(nonce, ciphertext, None)
        except Exception as e:  # cryptography raises InvalidTag on failure
            self.decrypt_fail_count += 1
            raise SecurityError(f"authentication failed: {e}") from e
        self.decrypt_ok_count += 1
        return plaintext

    def stats(self) -> dict:
        return {
            "encrypt_count": self.encrypt_count,
            "decrypt_ok_count": self.decrypt_ok_count,
            "decrypt_fail_count": self.decrypt_fail_count,
        }
