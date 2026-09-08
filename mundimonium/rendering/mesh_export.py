from __future__ import annotations

from mundimonium.coordinates.generic_tessellation import faces_around_vertex
from mundimonium.coordinates.isometric import IsometricDirection
from mundimonium.coordinates.tessellation import Tessellation

import numpy as np


def tessellation_to_buffers(
    tessellation: Tessellation,
) -> tuple[bytes, bytes, int, int, list[list[int]], list[list[int]]]:
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
    A `(positions_bytes, indices_bytes, vertex_count, face_count,
    adjacency, vertex_faces)` tuple.

    `positions_bytes` is `vertex_count * 3` little-endian `float32`
    values (`x0, y0, z0, x1, ...`), one triple per vertex in
    `tessellation.vertices`'s order.

    `indices_bytes` is `face_count * 3` little-endian `uint32` values
    (`b0, s0, d0, b1, ...`), one triangle per face in
    `tessellation.faces`'s order, each index into the `positions_bytes`
    vertex order above.

    `adjacency` has one `[neighbor_b, neighbor_s, neighbor_d]` entry per
    face (same order as `indices_bytes`'s own triangles): `neighbor_b`
    is the *triangle* index (not vertex index, and into this same list --
    i.e. matching `indices_bytes`'s triangle order) of the face across
    the edge opposite that triangle's own `b0`/`b1`/... vertex (the edge
    between its `s`/`d` vertices), and correspondingly for `neighbor_s`/
    `neighbor_d`; `-1` for a mesh boundary edge with no such neighbor.

    `vertex_faces` has one entry per vertex (same order as
    `positions_bytes`), the triangle indices of every face touching that
    vertex, in cyclic order around it -- or in `TessellationVertex.
    adjacent_faces`'s own (unordered) order if the vertex lies on an open
    mesh boundary, where no such cycle closes (see `faces_around_vertex`).
  """
  vertices = tessellation.vertices
  faces = tessellation.faces
  vertex_index_of = {vertex: i for i, vertex in enumerate(vertices)}
  face_index_of = {face: i for i, face in enumerate(faces)}

  positions = np.array(
      [vertex.projection_coordinates for vertex in vertices],
      dtype=np.float32)
  indices = np.array(
      [
          [vertex_index_of[face.vertex_b], vertex_index_of[face.vertex_s],
           vertex_index_of[face.vertex_d]]
          for face in faces
      ],
      dtype=np.uint32)

  adjacency = [
      [
          face_index_of[neighbor] if neighbor is not None else -1
          for neighbor in (face.face_on_edge(direction)
                           for direction in IsometricDirection)
      ]
      for face in faces
  ]

  vertex_faces = []
  for vertex in vertices:
    adjacent = vertex.adjacent_faces()
    fan = faces_around_vertex(adjacent[0], vertex) if adjacent else []
    ordered = fan if fan is not None else adjacent
    vertex_faces.append([face_index_of[face] for face in ordered])

  return (
      positions.tobytes(), indices.tobytes(),
      tessellation.num_vertices, tessellation.num_faces,
      adjacency, vertex_faces)
