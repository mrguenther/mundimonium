from __future__ import annotations

from itertools import count


class HashByIndex:
  """Mixin giving each instance a unique, monotonically assigned hash.

  Identity and hashing are based on a per-instance counter rather than object
  identity or field values, so instances hash and compare consistently even if
  `__eq__`/`__hash__` would otherwise be affected by mutable state.
  """

  _MAX_INDEX = 0x7fffffffffffffff  # 2**(64-1)-1 == max signed 64-bit integer
  _hash_index = 0

  @classmethod
  def _next_hash(cls) -> int:
    """Advances and returns the next hash value in the shared sequence."""
    cls._hash_index = (
        (cls._hash_index + 1) & cls._MAX_INDEX)
    return hash((cls._hash_index,))

  @classmethod
  def hash_index(cls) -> int:
    """Returns the most recently assigned counter value."""
    return cls._hash_index

  @classmethod
  def skip_first(cls, hash_count: int) -> None:
    """Advances the counter so the next `hash_count` values are skipped."""
    cls._hash_index = hash_count & cls._MAX_INDEX

  def __new__(cls, *args, **kwargs):
    """Assigns this instance the next unique hash before `__init__` runs."""
    instance = super().__new__(cls)
    instance._hash = HashByIndex._next_hash()
    return instance

  def __eq__(self, other: HashByIndex) -> bool:
    """Identity equality: an instance equals only itself (unless overridden)."""
    return self is other

  def __hash__(self) -> int:
    """Returns this instance's assigned unique hash."""
    return self._hash
