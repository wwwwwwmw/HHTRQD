from __future__ import annotations

import base64
import hashlib
import hmac
import os

try:
    import bcrypt  # type: ignore

    _HAS_BCRYPT = True
except Exception:
    bcrypt = None
    _HAS_BCRYPT = False


_PBKDF2_ITERATIONS = 210_000


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("ascii"))


def _hash_password_pbkdf2(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
        dklen=32,
    )
    return f"pbkdf2${_PBKDF2_ITERATIONS}${_b64e(salt)}${_b64e(dk)}"


def _verify_password_pbkdf2(password: str, encoded: str) -> bool:
    try:
        _, iter_text, salt_b64, dk_b64 = encoded.split("$", 3)
        iters = int(iter_text)
        salt = _b64d(salt_b64)
        expected = _b64d(dk_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iters,
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def hash_password(password: str) -> str:
    if _HAS_BCRYPT:
        pw_bytes = password.encode("utf-8")
        hashed = bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=12))
        return hashed.decode("utf-8")
    return _hash_password_pbkdf2(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False

    # Preferred explicit format
    if password_hash.startswith("pbkdf2$"):
        return _verify_password_pbkdf2(password, password_hash)

    # Backward-compat: hashes seeded via bcrypt are stored as raw bcrypt strings.
    if _HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception:
            return False

    # bcrypt not installed + hash looks like bcrypt => can't verify
    return False
