from __future__ import annotations

from typing import BinaryIO
import json
import struct

_LENGTH_STRUCT = struct.Struct('<I')


def _read_exact(stream: BinaryIO, count: int) -> bytes:
  """Reads exactly `count` bytes from `stream`, looping as needed.

  A single `stream.read(count)` call is not guaranteed to return all
  `count` bytes even when more are coming (e.g. over a pipe), so this
  loops until either exactly `count` bytes are collected or the stream is
  exhausted.

  Args:
    stream: The binary stream to read from.
    count: The exact number of bytes required.

  Returns:
    Exactly `count` bytes.

  Raises:
    EOFError: If the stream ends before `count` bytes are available.
  """
  chunks: list[bytes] = []
  remaining = count
  while remaining > 0:
    chunk = stream.read(remaining)
    if not chunk:
      raise EOFError(
          f"Stream closed after {count - remaining} of {count} expected "
          "bytes.")
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def read_frame(stream: BinaryIO) -> tuple[dict, bytes]:
  """Reads one framed message from `stream`.

  Wire format: a 4-byte little-endian header length, that many bytes of
  UTF-8 JSON, a 4-byte little-endian body length, then that many raw
  bytes.

  Args:
    stream: The binary stream to read from (e.g. `sys.stdin.buffer`).

  Returns:
    A `(header, body)` pair: the parsed JSON header and the raw body
    bytes (`body` is `b''`, not `None`, when the frame's body length is
    zero).

  Raises:
    EOFError: If the stream ends mid-frame.
  """
  (header_length,) = _LENGTH_STRUCT.unpack(_read_exact(stream, 4))
  header = json.loads(_read_exact(stream, header_length).decode('utf-8'))
  (body_length,) = _LENGTH_STRUCT.unpack(_read_exact(stream, 4))
  body = _read_exact(stream, body_length)
  return header, body


def write_frame(stream: BinaryIO, header: dict, body: bytes = b'') -> None:
  """Writes one framed message to `stream` and flushes it.

  The explicit flush matters: `stream` is typically a pipe to another
  process, not a TTY, so writes are fully buffered by default -- without
  flushing, a reader on the other end can hang waiting for bytes that
  were "written" but never actually left this process.

  Args:
    stream: The binary stream to write to (e.g. `sys.stdout.buffer`).
    header: The JSON-serializable header to write.
    body: The raw body bytes (default empty).
  """
  header_bytes = json.dumps(header).encode('utf-8')
  stream.write(_LENGTH_STRUCT.pack(len(header_bytes)))
  stream.write(header_bytes)
  stream.write(_LENGTH_STRUCT.pack(len(body)))
  stream.write(body)
  stream.flush()
