import io

import pytest

from mundimonium.rendering.protocol import read_frame, write_frame


def test_write_then_read_round_trips_header_and_body():
  buffer = io.BytesIO()
  write_frame(buffer, {"type": "get_mesh", "id": "1"}, b"hello")
  buffer.seek(0)
  header, body = read_frame(buffer)
  assert header == {"type": "get_mesh", "id": "1"}
  assert body == b"hello"


def test_zero_length_body_round_trips():
  buffer = io.BytesIO()
  write_frame(buffer, {"type": "error", "message": "oops"})
  buffer.seek(0)
  header, body = read_frame(buffer)
  assert header == {"type": "error", "message": "oops"}
  assert body == b""


class _FragmentedStream:
  """A stream that never returns more than a few bytes per `read()` call,
  the way a real pipe legitimately can -- for exercising `read_frame`'s
  reassembly logic."""

  def __init__(self, data: bytes, max_chunk: int = 3):
    self._data = data
    self._pos = 0
    self._max_chunk = max_chunk

  def read(self, count: int) -> bytes:
    chunk = self._data[self._pos:self._pos + min(count, self._max_chunk)]
    self._pos += len(chunk)
    return chunk


def test_read_frame_handles_a_fragmented_stream():
  buffer = io.BytesIO()
  write_frame(buffer, {"type": "mesh"}, b"0123456789")

  header, body = read_frame(_FragmentedStream(buffer.getvalue()))
  assert header == {"type": "mesh"}
  assert body == b"0123456789"


def test_read_frame_raises_on_a_stream_truncated_mid_header():
  # Claims a 5-byte header but the stream only has 3.
  buffer = io.BytesIO(b"\x05\x00\x00\x00abc")
  with pytest.raises(EOFError):
    read_frame(buffer)


def test_read_frame_raises_on_a_stream_truncated_mid_body():
  buffer = io.BytesIO()
  write_frame(buffer, {"type": "mesh"}, b"0123456789")
  truncated = io.BytesIO(buffer.getvalue()[:-3])
  with pytest.raises(EOFError):
    read_frame(truncated)
