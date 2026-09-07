"""Official AG-ReID.v2 identity parsing and experiment audit metadata."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping


# AG-ReID.v2's official loader concatenates the P, T and A components.  P by
# itself is only unique inside the current video and collapses distinct tracks.
IDENTITY_SCHEME = "agreid_v2_p+t+a_v1"
FILENAME_PATTERN = (
    r"P(?P<person>\d+)T(?P<timestamp>\d+)A(?P<altitude>\d+)C\d+F"
)
IDENTITY_PATTERN = re.compile(
    r"P(?P<person>\d+)T(?P<timestamp>\d+)A(?P<altitude>\d+)"
)
OFFICIAL_PROTOCOL_IDENTITY_COUNTS = {
    "exp1_aerial_to_cctv.txt": 534,
    "exp2_aerial_to_wearable.txt": 519,
    "exp4_cctv_to_aerial.txt": 534,
    "exp5_wearable_to_aerial.txt": 519,
}


def parse_identity_id(value: str, pattern: str | re.Pattern[str] | None = None) -> int:
    """Return the official integer PID formed by concatenating P + T + A."""

    regex = IDENTITY_PATTERN if pattern is None else (
        pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
    )
    text = os.path.basename(os.fspath(value))
    match = regex.search(text)
    if not match:
        raise ValueError(f"Cannot parse AG-ReID.v2 P+T+A identity from {value}")

    groups = match.groupdict()
    required = ("person", "timestamp", "altitude")
    if any(not groups.get(name) for name in required):
        raise ValueError(
            "AG-ReID.v2 identity patterns must capture named groups "
            "'person', 'timestamp', and 'altitude'; P alone is not a valid PID"
        )
    return int("".join(groups[name] for name in required))


def validate_identity_count(actual: int, expected: int | None, context: str) -> None:
    """Fail fast when an official dataset split is parsed with wrong semantics."""

    if expected is not None and actual != expected:
        raise RuntimeError(
            f"{context}: parsed {actual} identities, expected {expected}. "
            "AG-ReID.v2 identities must use the composite P+T+A key."
        )


def checkpoint_identity_scheme(payload) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    scheme = payload.get("identity_scheme")
    if scheme:
        return str(scheme)
    config = payload.get("config")
    if isinstance(config, Mapping) and config.get("identity_scheme"):
        return str(config["identity_scheme"])
    return None


def require_current_identity_scheme(payload, path: str) -> None:
    """Reject supervised checkpoints trained before the P+T+A parser fix."""

    scheme = checkpoint_identity_scheme(payload)
    if scheme != IDENTITY_SCHEME:
        found = scheme or "missing (legacy P-only checkpoint)"
        raise RuntimeError(
            f"Refusing checkpoint {path}: identity scheme is {found!r}, expected "
            f"{IDENTITY_SCHEME!r}. Rerun this supervised stage after the parser fix."
        )
