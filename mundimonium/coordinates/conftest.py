"""Shared pytest fixtures for mundimonium/coordinates/*_test.py.

Actual tests live in each module's own `<module>_test.py`; this file only
holds setup that's identical across several of them.
"""

import math

import pytest

from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.tessellation import TessellationVertex

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


def build_icosahedron(face_type=None):
  """A `GenericTessellation` populated as a regular icosahedron.

  Returns (tessellation, vertices, faces).
  """
  tess = GenericTessellation(face_type=face_type)
  verts = [TessellationVertex(list(v)) for v in _BASE_RAW]
  faces = [
      tess.add_face([verts[a], verts[b], verts[c]])
      for a, b, c in _BASE_FACES
  ]
  return tess, verts, faces


@pytest.fixture(scope="module")
def icosahedron():
  return build_icosahedron()
