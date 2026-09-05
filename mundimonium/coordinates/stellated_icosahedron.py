"""Icosahedron/stellated-icosahedron `GenericTessellation` geometry.

Shared, non-test-specific construction helpers -- used by `conftest.py`'s
pytest fixtures and by `mundimonium/rendering/generic_demo.py`'s production
demo content alike.
"""

from __future__ import annotations

from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.tessellation import TessellationVertex

import math
import numpy as np

_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_BASE_RAW = [
    [-1.0, _PHI, 0.0], [1.0, _PHI, 0.0], [-1.0, -_PHI, 0.0], [1.0, -_PHI, 0.0],
    [0.0, -1.0, _PHI], [0.0, 1.0, _PHI], [0.0, -1.0, -_PHI], [0.0, 1.0, -_PHI],
    [_PHI, 0.0, -1.0], [_PHI, 0.0, 1.0], [-_PHI, 0.0, -1.0], [-_PHI, 0.0, 1.0],
]
_BASE_FACES = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]


def build_icosahedron(face_type=None, vertex_type=None):
  """A `GenericTessellation` populated as a regular icosahedron.

  Returns (tessellation, vertices, faces).
  """
  tess = GenericTessellation(face_type=face_type, vertex_type=vertex_type)
  vertex_cls = vertex_type or TessellationVertex
  verts = [vertex_cls(list(v)) for v in _BASE_RAW]
  faces = [
      tess.add_face([verts[a], verts[b], verts[c]])
      for a, b, c in _BASE_FACES
  ]
  return tess, verts, faces


def _subdivide_triangle_mesh(
    positions: list[np.ndarray],
    face_index_triples: list[tuple[int, int, int]],
    frequency: int,
    tess: GenericTessellation,
    vertex_cls: type[TessellationVertex],
) -> tuple[list[TessellationVertex], list]:
  """Replaces each flat triangle in `face_index_triples` with `frequency**2`
  smaller true faces (real `TessellationFace`s via `add_face`, not
  `NestingIsoGrid` sectors from `.subdivide()`), welding shared corners and
  edges into single vertex objects across adjacent input triangles so the
  result is one connected mesh rather than many disconnected triangles.

  Each input triangle is subdivided via a standard barycentric grid: for
  `i + j + k == frequency`, a lattice point at `(i * A + j * B + k * C) /
  frequency`, where `A`, `B`, `C` are `positions[index_a/b/c]`. Since every
  input triangle here is equilateral, this produces `frequency**2` smaller
  equilateral triangles per input face, each with the same winding order as
  its parent (so outward-facing normals are preserved).

  Corner and edge lattice points are deduplicated by the global vertex
  indices they fall between (not by 3D position), so two input triangles
  sharing an edge -- regardless of which of their own `(index_a, index_b,
  index_c)` slots that edge occupies, or which direction each names it --
  resolve to the exact same subdivided vertex objects along that edge.

  Args:
    positions:          3D position for each global vertex index referenced
                         by `face_index_triples`.
    face_index_triples:  `(index_a, index_b, index_c)` triples into
                         `positions`, one per input triangle, in the winding
                         order `add_face` expects.
    frequency:          Number of subdivisions per edge; each input triangle
                         becomes `frequency**2` new faces.
    tess:               The `GenericTessellation` to register new faces on.
    vertex_cls:         Constructor for new vertices (e.g. `TessellationVertex`
                         or a subclass).

  Returns:
    `(vertices, faces)`: every newly created vertex and face, in creation
    order (not including any vertex/face from the un-subdivided input mesh,
    which is never itself registered on `tess`).
  """
  corner_vertices: dict[int, TessellationVertex] = {}
  edge_vertices: dict[tuple[int, int, int], TessellationVertex] = {}
  vertices: list[TessellationVertex] = []
  faces = []

  def corner_vertex(index: int) -> TessellationVertex:
    if index not in corner_vertices:
      vertex = vertex_cls(list(positions[index]))
      corner_vertices[index] = vertex
      vertices.append(vertex)
    return corner_vertices[index]

  def edge_vertex(
      index_from: int, index_to: int, steps_from: int) -> TessellationVertex:
    if index_from < index_to:
      low, high, steps_from_low = index_from, index_to, steps_from
    else:
      low, high, steps_from_low = index_to, index_from, frequency - steps_from
    key = (low, high, steps_from_low)
    if key not in edge_vertices:
      fraction = steps_from_low / frequency
      position = positions[low] * (1 - fraction) + positions[high] * fraction
      vertex = vertex_cls(list(position))
      edge_vertices[key] = vertex
      vertices.append(vertex)
    return edge_vertices[key]

  for index_a, index_b, index_c in face_index_triples:
    position_a = positions[index_a]
    position_b = positions[index_b]
    position_c = positions[index_c]
    side_length = (
        float(np.linalg.norm(position_b - position_a)) / frequency)

    grid: dict[tuple[int, int], TessellationVertex] = {}
    for i in range(frequency + 1):
      for j in range(frequency + 1 - i):
        k = frequency - i - j
        if i == frequency:
          grid[(i, j)] = corner_vertex(index_a)
        elif j == frequency:
          grid[(i, j)] = corner_vertex(index_b)
        elif k == frequency:
          grid[(i, j)] = corner_vertex(index_c)
        elif k == 0:
          grid[(i, j)] = edge_vertex(index_a, index_b, j)
        elif j == 0:
          grid[(i, j)] = edge_vertex(index_a, index_c, k)
        elif i == 0:
          grid[(i, j)] = edge_vertex(index_b, index_c, k)
        else:
          position = (
              i * position_a + j * position_b + k * position_c) / frequency
          vertex = vertex_cls(list(position))
          vertices.append(vertex)
          grid[(i, j)] = vertex

    for i in range(frequency):
      for j in range(frequency - i):
        corner_0 = grid[(i, j)]
        corner_1 = grid[(i + 1, j)]
        corner_2 = grid[(i, j + 1)]
        faces.append(tess.add_face(
            [corner_0, corner_1, corner_2], side_length=side_length))
        if i + j < frequency - 1:
          corner_3 = grid[(i + 1, j + 1)]
          faces.append(tess.add_face(
              [corner_1, corner_3, corner_2], side_length=side_length))

  return vertices, faces


def build_subdivided_icosahedron(frequency, face_type=None, vertex_type=None):
  """Like `build_icosahedron`, but each of the 20 faces is replaced by
  `frequency**2` smaller true faces (real `TessellationFace`s, not
  `NestingIsoGrid` sectors), welded together at shared vertices/edges.

  A finer, more realistic-scale mesh than the 20-face base icosahedron for
  exercising `flatten_region`: `RELAXATION_RADIUS` on the base mesh
  inevitably spans the whole, highly-curved solid, but on a subdivided mesh
  it only spans a small, much-less-curved neighborhood -- closer to how
  `GenericTessellation` is meant to be used in practice.

  Returns (tessellation, vertices, faces). `vertices`/`faces` are the newly
  created subdivision-triangle objects -- the 12 original icosahedron
  corners and 20 original faces are never themselves registered on `tess`.
  """
  tess = GenericTessellation(face_type=face_type, vertex_type=vertex_type)
  vertex_cls = vertex_type or TessellationVertex
  positions = [np.array(v) for v in _BASE_RAW]
  vertices, faces = _subdivide_triangle_mesh(
      positions, _BASE_FACES, frequency, tess, vertex_cls)
  return tess, vertices, faces


def build_subdivided_stellated_icosahedron(
    frequency, face_type=None, vertex_type=None):
  """Like `build_stellated_icosahedron`, but each of the 60 tetrahedron-cap
  faces is replaced by `frequency**2` smaller true faces, for the same
  reason `build_subdivided_icosahedron` gives the plain icosahedron one.

  Returns (tessellation, vertices, faces), with the same caveat as
  `build_subdivided_icosahedron`: these are the newly created subdivision
  vertices/faces, not the original 32-vertex/60-face stellated mesh.
  """
  tess = GenericTessellation(face_type=face_type, vertex_type=vertex_type)
  vertex_cls = vertex_type or TessellationVertex
  base_positions = [np.array(v) for v in _BASE_RAW]

  index_a0, index_b0 = _BASE_FACES[0][0], _BASE_FACES[0][1]
  edge_length = float(np.linalg.norm(
      base_positions[index_a0] - base_positions[index_b0]))
  circumradius = edge_length / math.sqrt(3.0)
  apex_height = math.sqrt(edge_length ** 2 - circumradius ** 2)

  apex_positions = []
  face_index_triples = []
  num_base_vertices = len(base_positions)
  for face_index, (index_a, index_b, index_c) in enumerate(_BASE_FACES):
    va = base_positions[index_a]
    vb = base_positions[index_b]
    vc = base_positions[index_c]
    centroid = (va + vb + vc) / 3.0
    normal = np.cross(vb - va, vc - va)
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, centroid) < 0:  # keep the apex pointing outward
      normal = -normal
    apex_positions.append(centroid + normal * apex_height)

    apex_index = num_base_vertices + face_index
    face_index_triples.append((apex_index, index_a, index_b))
    face_index_triples.append((apex_index, index_b, index_c))
    face_index_triples.append((apex_index, index_c, index_a))

  positions = base_positions + apex_positions
  vertices, faces = _subdivide_triangle_mesh(
      positions, face_index_triples, frequency, tess, vertex_cls)
  return tess, vertices, faces


def build_stellated_icosahedron(face_type=None, vertex_type=None):
  """A `GenericTessellation` shaped like a stellated icosahedron.

  Each of the 20 icosahedron faces is replaced by a regular tetrahedron
  "cap": a new apex vertex, raised above the face's centroid along its
  outward normal, is connected to each pair of the face's own vertices,
  giving 3 new equilateral faces (all congruent to the original icosahedron
  edge length, so the mesh stays uniform, as `GenericTessellation`'s Heat
  Method solver assumes) in place of the 1 they replace.

  This gives a closed but non-convex mesh with 32 vertices and 60 faces --
  unlike the icosahedron, most vertices now have irregular valence (the
  original 12 vertices go from valence 5 to valence 10; the 20 new apex
  vertices have valence 3), and it's a good stress case for anything that
  assumed convexity or valence-6-ish regularity.

  Returns (tessellation, vertices, faces). `vertices` is the original 12
  icosahedron vertices followed by the 20 new apex vertices (one per
  original face, in `_BASE_FACES` order). `faces` is the 60 new faces, 3 per
  original face (in `_BASE_FACES` order), each triple ordered
  `(apex, va, vb), (apex, vb, vc), (apex, vc, va)`.
  """
  tess = GenericTessellation(face_type=face_type, vertex_type=vertex_type)
  vertex_cls = vertex_type or TessellationVertex
  base_verts = [vertex_cls(list(v)) for v in _BASE_RAW]

  a0, b0 = _BASE_FACES[0][0], _BASE_FACES[0][1]
  edge_length = float(np.linalg.norm(
      np.array(_BASE_RAW[a0]) - np.array(_BASE_RAW[b0])))
  circumradius = edge_length / math.sqrt(3.0)
  apex_height = math.sqrt(edge_length ** 2 - circumradius ** 2)

  apex_verts = []
  faces = []
  for a, b, c in _BASE_FACES:
    va = np.array(_BASE_RAW[a])
    vb = np.array(_BASE_RAW[b])
    vc = np.array(_BASE_RAW[c])
    centroid = (va + vb + vc) / 3.0
    normal = np.cross(vb - va, vc - va)
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, centroid) < 0:  # keep the apex pointing outward
      normal = -normal
    apex_vertex = vertex_cls(list(centroid + normal * apex_height))
    apex_verts.append(apex_vertex)

    va_v, vb_v, vc_v = base_verts[a], base_verts[b], base_verts[c]
    faces.append(tess.add_face(
        [apex_vertex, va_v, vb_v], side_length=edge_length))
    faces.append(tess.add_face(
        [apex_vertex, vb_v, vc_v], side_length=edge_length))
    faces.append(tess.add_face(
        [apex_vertex, vc_v, va_v], side_length=edge_length))

  return tess, base_verts + apex_verts, faces
