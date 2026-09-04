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
