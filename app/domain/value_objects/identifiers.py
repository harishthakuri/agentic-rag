"""Entity identifiers.

UUIDv7 (RFC 9562) starts with a millisecond timestamp, so new IDs are
roughly sorted by creation time. Compared with random UUIDv4 this keeps
B-tree index inserts local (less page splitting) while still letting the
application assign IDs before the row is written.
"""

import secrets
import time
from uuid import UUID

_RAND_A_BITS = 12
_RAND_B_BITS = 62


def new_id() -> UUID:
    unix_ms = time.time_ns() // 1_000_000
    rand = secrets.randbits(_RAND_A_BITS + _RAND_B_BITS)
    rand_a = rand >> _RAND_B_BITS
    rand_b = rand & ((1 << _RAND_B_BITS) - 1)

    value = (unix_ms & ((1 << 48) - 1)) << 80  # 48-bit timestamp
    value |= 0x7 << 76  # version 7
    value |= rand_a << 64
    value |= 0b10 << 62  # RFC 9562 variant
    value |= rand_b
    return UUID(int=value)
