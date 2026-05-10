"""
Symmetric encryption helpers for storing third-party credentials at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) keyed by a derivative of SECRET_KEY,
so rotating SECRET_KEY invalidates all stored ciphertexts (intentional).
"""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _key() -> bytes:
    digest = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_str(plaintext: str) -> str:
    return Fernet(_key()).encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_str(token: str) -> str | None:
    try:
        return Fernet(_key()).decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        return None
