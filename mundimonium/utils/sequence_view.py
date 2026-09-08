from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar('_T')


class SequenceView(Sequence[_T]):
  """A thin, read-only view over an existing sequence.

  Lets a class expose a `list`-like attribute without handing out a
  reference callers could mutate, and without copying the underlying
  container on every access. Backed by `collections.abc.Sequence`:
  implementing just `__getitem__` and `__len__` gives iteration, `in`,
  `reversed()`, `.index()`, and `.count()` for free. The standard library
  has a ready-made proxy for mappings (`types.MappingProxyType`) but not
  for sequences, so this fills that gap.
  """

  def __init__(self, items: Sequence[_T]):
    """Constructs a view over `items`.

    Args:
      items: The sequence to wrap. Not copied -- if `items` is itself
        mutable, later changes to it are visible through this view.
    """
    self._items = items

  def __getitem__(self, index):
    return self._items[index]

  def __len__(self) -> int:
    return len(self._items)

  def __repr__(self) -> str:
    return f"{type(self).__name__}({self._items!r})"
