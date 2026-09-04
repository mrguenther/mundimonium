from __future__ import annotations

from mundimonium.coordinates.tessellation import Tessellation

import numpy as np


def tessellation_to_buffers(
    tessellation: Tessellation,
) -> tuple[bytes, bytes, int, int]:
  """Exports `tessellation`'s current mesh as renderer-ready buffers.

  Assumes `TessellationVertex.projection_coordinates` are already
  Euclidean 3D positions -- true for `SphericalTessellation`, but *not*
  for `HyperbolicTessellation` (whose `projection_coordinates` are
  Minkowski `(X, Y, Z)`, per that class's own docstrings). Do not reuse
  this function for a hyperbolic tessellation without first converting to
  Euclidean coordinates.

  Args:
    tessellation: The tessellation to export.

  Returns:
    A `(positions_bytes, indices_bytes, vertex_count, face_count)` tuple.

    `positions_bytes` is `vertex_count * 3` little-endian `float32`
    values (`x0, y0, z0, x1, ...`), one triple per vertex in
    `tessellation.vertices`'s order.

    `indices_bytes` is `face_count * 3` little-endian `uint32` values
    (`b0, s0, d0, b1, ...`), one triangle per face in
    `tessellation.faces`'s order, each index into the `positions_bytes`
    vertex order above.
  """
  vertices = tessellation.vertices
  index_of = {vertex: i for i, vertex in enumerate(vertices)}
  positions = np.array(
      [vertex.projection_coordinates for vertex in vertices],
      dtype=np.float32)
  indices = np.array(
      [
          [index_of[face.vertex_b], index_of[face.vertex_s],
           index_of[face.vertex_d]]
          for face in tessellation.faces
      ],
      dtype=np.uint32)
  return (
      positions.tobytes(), indices.tobytes(),
      tessellation.num_vertices, tessellation.num_faces)
