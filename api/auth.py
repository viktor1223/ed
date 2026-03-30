"""JWT authentication helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode

SECRET_KEY = os.environ.get("ED_SECRET_KEY", "dev-secret-change-in-production")
TOKEN_EXPIRY = 86400 * 7  # 7 days


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return hmac.compare_digest(dk.hex(), dk_hex)


def create_token(teacher_id: int) -> str:
    payload = {
        "sub": teacher_id,
        "exp": int(time.time()) + TOKEN_EXPIRY,
    }
    payload_bytes = urlsafe_b64encode(json.dumps(payload).encode())
    sig = hmac.new(SECRET_KEY.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return payload_bytes.decode() + "." + sig


def decode_token(token: str) -> int | None:
    """Returns teacher_id or None if invalid/expired."""
    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(
            SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        # urlsafe_b64decode needs padding
        padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(urlsafe_b64decode(padded))
        if payload.get("exp", 0) < time.time():
            return None
        return payload["sub"]
    except Exception:
        return None
