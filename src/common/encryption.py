"""
Encryption utilities for simulating host <-> enclave boundary security.

We use AES-GCM for authenticated symmetric encryption because:
- It provides confidentiality + integrity.
- It is fast and appropriate for repeated message encryption (token streaming).
- It mirrors how real TEEs often use symmetric keys for data-in-use protection.

This module intentionally does NOT log or persist keys or plaintext.
"""

from __future__ import annotations

import os
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# -------------------------------------------------------------------------
# Key Utilities
# -------------------------------------------------------------------------

def generate_key() -> bytes:
    """
    Generates a fresh 256-bit AES key for worker or host use.
    AES-GCM supports 128/192/256 bit keys; we use 256 for simulation.
    """
    return AESGCM.generate_key(bit_length=256)


# -------------------------------------------------------------------------
# AES-GCM Encryption / Decryption
# -------------------------------------------------------------------------

def encrypt_blob(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypts a plaintext blob using AES-GCM.
    
    The blob returned is: nonce (12 bytes) || ciphertext
    
    Inputs:
        plaintext: raw bytes to encrypt
        key: AES 256-bit key

    Output:
        encrypted bytes (nonce + ciphertext + tag)
    """
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("encrypt_blob() expects plaintext as bytes")

    aesgcm = AESGCM(key)

    # AES-GCM standard nonce size is 96 bits (12 bytes)
    nonce = os.urandom(12)

    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    # Return nonce + ciphertext (ciphertext includes auth tag)
    return nonce + ciphertext


def decrypt_blob(encrypted_blob: bytes, key: bytes) -> bytes:
    """
    Decrypts a blob encrypted with AES-GCM.
    
    Input:
        encrypted_blob: nonce || ciphertext
        key: AES key used for encryption

    Output:
        plaintext bytes
    """
    if len(encrypted_blob) < 12:
        raise ValueError("Encrypted blob too short: missing nonce")

    nonce = encrypted_blob[:12]
    ciphertext = encrypted_blob[12:]

    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
        return plaintext
    except Exception as e:
        # Never leak plaintext; only raise general error
        raise ValueError("Decryption failed") from e


# -------------------------------------------------------------------------
# Token Streaming Helpers (Optional)
# -------------------------------------------------------------------------

def encrypt_token(token: str, key: bytes) -> bytes:
    """
    Convenience helper to encrypt a single token (string).
    """
    return encrypt_blob(token.encode("utf-8"), key)


def decrypt_token(encrypted_token: bytes, key: bytes) -> str:
    """
    Convenience helper to decrypt a single token into string form.
    """
    return decrypt_blob(encrypted_token, key).decode("utf-8")

