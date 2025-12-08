# src/common/encryption.py

"""
Encryption utilities for simulating host <-> enclave boundary security.

We use:
- AES-GCM for authenticated symmetric encryption (fast, confidentiality + integrity).
- Hybrid RSA + AES for orchestrator -> worker (enclave) messages:
    * Orchestrator generates an ephemeral AES key.
    * Encrypts the plaintext with AES-GCM using that key.
    * Encrypts the AES key with the worker's RSA public key (OAEP).
    * Packs [4-byte key_len][rsa_encrypted_key][aes_cipher_blob].

The worker:
    * Uses its RSA private key to recover the AES key.
    * Uses AES-GCM to get the plaintext back.

For this simulation, the worker's RSA keypair is stored on disk so that
both orchestrator and worker processes share the same keypair.

This module does NOT log or persist plaintext anywhere.
"""

from __future__ import annotations

import os
import struct
from typing import Tuple, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding


# -------------------------------------------------------------------------
# AES key generation
# -------------------------------------------------------------------------


def generate_key() -> bytes:
    """
    Generates a fresh 256-bit AES key for worker or host use.
    AES-GCM supports 128/192/256 bit keys; we use 256 for simulation.
    """
    return AESGCM.generate_key(bit_length=256)


# -------------------------------------------------------------------------
# Worker RSA keypair (simulated enclave identity, shared via disk)
# -------------------------------------------------------------------------

RSA_KEY_SIZE = 2048
RSA_PUBLIC_EXPONENT = 65537

# Module-local cache for already-loaded PEM blobs
_WORKER_PRIVATE_PEM: Optional[bytes] = None
_WORKER_PUBLIC_PEM: Optional[bytes] = None


def _rsa_key_paths() -> Tuple[str, str]:
    """
    Returns (private_path, public_path) for worker RSA keypair.

    By default, stores keys under a 'keys/' directory next to the repo root,
    but you can override with SECURELLM_RSA_KEY_DIR if you want.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(base_dir, "..", "..", "keys"))
    key_dir = os.getenv("SECURELLM_RSA_KEY_DIR", default_dir)
    os.makedirs(key_dir, exist_ok=True)

    priv_path = os.path.join(key_dir, "worker_rsa_private.pem")
    pub_path = os.path.join(key_dir, "worker_rsa_public.pem")
    return priv_path, pub_path


def _generate_rsa_keypair() -> Tuple[bytes, bytes]:
    """
    Generate an RSA keypair for a worker (simulated enclave).

    Returns
    -------
    private_pem : bytes
        PEM-encoded PKCS8 private key (unencrypted).
    public_pem : bytes
        PEM-encoded SubjectPublicKeyInfo public key.
    """
    private_key = rsa.generate_private_key(
        public_exponent=RSA_PUBLIC_EXPONENT,
        key_size=RSA_KEY_SIZE,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _ensure_worker_rsa_keys() -> None:
    """
    Ensure the worker RSA keypair exists on disk and is loaded into memory.

    The first process to call this will generate the keypair and write it to disk.
    Subsequent calls (in any process) will just read the existing files.
    """
    global _WORKER_PRIVATE_PEM, _WORKER_PUBLIC_PEM

    if _WORKER_PRIVATE_PEM is not None and _WORKER_PUBLIC_PEM is not None:
        return

    priv_path, pub_path = _rsa_key_paths()

    if os.path.exists(priv_path) and os.path.exists(pub_path):
        # Load existing keys from disk
        with open(priv_path, "rb") as f:
            _WORKER_PRIVATE_PEM = f.read()
        with open(pub_path, "rb") as f:
            _WORKER_PUBLIC_PEM = f.read()
        return

    # No keys on disk yet: generate and persist them
    private_pem, public_pem = _generate_rsa_keypair()

    with open(priv_path, "wb") as f:
        f.write(private_pem)
    with open(pub_path, "wb") as f:
        f.write(public_pem)

    _WORKER_PRIVATE_PEM = private_pem
    _WORKER_PUBLIC_PEM = public_pem


def get_worker_private_key() -> bytes:
    """
    Returns the PEM-encoded RSA private key for the worker.

    In a real TEE, this key would never be visible to the host.
    Here it's used by the worker process to simulate enclave decryption.
    """
    _ensure_worker_rsa_keys()
    assert _WORKER_PRIVATE_PEM is not None
    return _WORKER_PRIVATE_PEM


def get_worker_public_key() -> bytes:
    """
    Returns the PEM-encoded RSA public key for the worker.

    In a real PCC-style system, the host learns this from the attestation report.
    Here the orchestrator uses it to encrypt payloads into the enclave.
    """
    _ensure_worker_rsa_keys()
    assert _WORKER_PUBLIC_PEM is not None
    return _WORKER_PUBLIC_PEM


# -------------------------------------------------------------------------
# Hybrid RSA + AES helpers
# -------------------------------------------------------------------------


def hybrid_encrypt_for_worker(plaintext: bytes, worker_public_pem: bytes) -> bytes:
    """
    Hybrid encryption used by the orchestrator to send data *into* the enclave.

    Steps:
      1. Generate random 256-bit AES key.
      2. Encrypt the plaintext with AES-GCM via `encrypt_blob`.
      3. Encrypt the AES key with the worker's RSA public key (OAEP).
      4. Pack as: [4 bytes big-endian rsa_key_len][rsa_key_bytes][aes_cipher_blob].

    Returns
    -------
    hybrid_blob : bytes
        Opaque binary blob that the worker can decrypt.
    """
    # Load worker's public key
    public_key = serialization.load_pem_public_key(worker_public_pem)

    # 1) Ephemeral AES key
    aes_key = os.urandom(32)  # 256-bit

    # 2) AES-GCM encrypt the plaintext
    aes_cipher_blob = encrypt_blob(plaintext, aes_key)

    # 3) RSA encrypt the AES key itself
    rsa_encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # 4) Pack [key_len][rsa_key][aes_blob]
    header = struct.pack("!I", len(rsa_encrypted_key))
    hybrid_blob = header + rsa_encrypted_key + aes_cipher_blob
    return hybrid_blob


def hybrid_decrypt_at_worker(hybrid_blob: bytes, worker_private_pem: bytes) -> bytes:
    """
    Hybrid decryption used *inside* the worker (enclave).

    Input: blob produced by `hybrid_encrypt_for_worker`.

    Steps:
      1. Parse [4-byte key_len][rsa_key_bytes][aes_cipher_blob].
      2. Use worker RSA private key to recover AES key.
      3. Use AES-GCM (decrypt_blob) to recover plaintext.

    Returns
    -------
    plaintext : bytes
    """
    # Load worker's private key
    private_key = serialization.load_pem_private_key(
        worker_private_pem,
        password=None,
    )

    # 1) Parse header
    if len(hybrid_blob) < 4:
        raise ValueError("Hybrid blob too short (missing key length header)")

    (key_len,) = struct.unpack("!I", hybrid_blob[:4])
    offset = 4

    if len(hybrid_blob) < 4 + key_len:
        raise ValueError("Hybrid blob too short (missing RSA-encrypted key)")

    rsa_encrypted_key = hybrid_blob[offset : offset + key_len]
    aes_cipher_blob = hybrid_blob[offset + key_len :]

    # 2) RSA decrypt AES key
    aes_key = private_key.decrypt(
        rsa_encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # 3) AES-GCM decrypt using the recovered AES key
    plaintext = decrypt_blob(aes_cipher_blob, aes_key)
    return plaintext


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
# Token Streaming Helpers
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