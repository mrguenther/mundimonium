"""Shared pytest fixtures for mundimonium/coordinates/*_test.py.

Actual tests live in each module's own `<module>_test.py`; this file only
holds setup that's identical across several of them.
"""

import math

import numpy as np
import pytest

from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.isometric import IsometricDirection
from mundimonium.coordinates.tessellation import TessellationFace, TessellationVertex

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


def build_stellated_icosahedron():
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
  tess = GenericTessellation()
  base_verts = [TessellationVertex(list(v)) for v in _BASE_RAW]

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
    apex_vertex = TessellationVertex(list(centroid + normal * apex_height))
    apex_verts.append(apex_vertex)

    va_v, vb_v, vc_v = base_verts[a], base_verts[b], base_verts[c]
    faces.append(tess.add_face(
        [apex_vertex, va_v, vb_v], side_length=edge_length))
    faces.append(tess.add_face(
        [apex_vertex, vb_v, vc_v], side_length=edge_length))
    faces.append(tess.add_face(
        [apex_vertex, vc_v, va_v], side_length=edge_length))

  return tess, base_verts + apex_verts, faces


# ---------------------------------------------------------------------------
# Extensible collection of GenericTessellation mesh-shape fixtures
# ---------------------------------------------------------------------------
#
# `GenericTessellation` is meant to work for arbitrary equilateral-triangle
# mesh topologies, not just nicely-symmetric ones like the icosahedron. Tests
# that assert mesh-shape-agnostic properties (e.g. "geodesic_distance is
# symmetric", "shortest_path_by_segment never straddles two grids") should be
# parametrized over every registered `GenericTessellationFixture` subclass
# (via the `generic_mesh` fixture below) instead of hardcoding the
# icosahedron, so that adding a new, weirder mesh shape automatically
# exercises every such test with no changes required there.

class GenericTessellationFixture:
  """Base class for a registered `GenericTessellation` mesh-shape fixture.

  Subclassing automatically registers the subclass (see `__init_subclass__`),
  so `GenericTessellationFixture.all()` always lists every fixture defined
  anywhere, in definition order. Each subclass must implement `build()`,
  `any_face()`, `two_adjacent_faces()`, and `two_far_apart_faces()` itself
  (deliberately not derived generically -- the best choice of faces for
  testing purposes can depend on the specific topology, and a generic
  derivation, e.g. via BFS, is itself nontrivial test-fixture logic that
  could carry its own bugs).
  """

  _fixtures: list[type["GenericTessellationFixture"]] = []

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    GenericTessellationFixture._fixtures.append(cls)

  @classmethod
  def all(cls) -> list[type["GenericTessellationFixture"]]:
    """Every registered fixture subclass, in definition order."""
    return list(GenericTessellationFixture._fixtures)

  @classmethod
  def build(
      cls,
  ) -> tuple[GenericTessellation, list[TessellationVertex],
             list[TessellationFace]]:
    """Constructs (and should cache) this fixture's tessellation."""
    raise NotImplementedError()

  @classmethod
  def any_face(cls) -> TessellationFace:
    """An arbitrary face of this mesh."""
    raise NotImplementedError()

  @classmethod
  def two_adjacent_faces(cls) -> tuple[TessellationFace, TessellationFace]:
    """Two faces of this mesh that share an edge."""
    raise NotImplementedError()

  @classmethod
  def two_far_apart_faces(cls) -> tuple[TessellationFace, TessellationFace]:
    """Two faces of this mesh separated by several hops."""
    raise NotImplementedError()


class IcosahedronFixture(GenericTessellationFixture):
  """Regular icosahedron: 20 congruent equilateral faces, closed and convex.
  """

  _cached: tuple | None = None

  @classmethod
  def build(cls):
    if cls._cached is None:
      cls._cached = build_icosahedron()
    return cls._cached

  @classmethod
  def any_face(cls) -> TessellationFace:
    _, _, faces = cls.build()
    return faces[0]

  @classmethod
  def two_adjacent_faces(cls) -> tuple[TessellationFace, TessellationFace]:
    face = cls.any_face()
    return face, face.face_on_edge(IsometricDirection.B)

  @classmethod
  def two_far_apart_faces(cls) -> tuple[TessellationFace, TessellationFace]:
    _, _, faces = cls.build()
    return faces[0], faces[15]


class StellatedIcosahedronFixture(GenericTessellationFixture):
  """Stellated icosahedron: 60 equilateral faces, closed and non-convex.

  Each icosahedron face is capped with a regular tetrahedron; see
  `build_stellated_icosahedron` for details.
  """

  _cached: tuple | None = None

  @classmethod
  def build(cls):
    if cls._cached is None:
      cls._cached = build_stellated_icosahedron()
    return cls._cached

  @classmethod
  def any_face(cls) -> TessellationFace:
    _, _, faces = cls.build()
    return faces[0]

  @classmethod
  def two_adjacent_faces(cls) -> tuple[TessellationFace, TessellationFace]:
    face = cls.any_face()
    return face, face.face_on_edge(IsometricDirection.B)

  @classmethod
  def two_far_apart_faces(cls) -> tuple[TessellationFace, TessellationFace]:
    # Sub-face index 0 (original face 0's cap) and 3*15 (original face
    # 15's cap), matching `IcosahedronFixture.two_far_apart_faces`'s
    # indexing scheme.
    _, _, faces = cls.build()
    return faces[0], faces[3 * 15]


@pytest.fixture(params=GenericTessellationFixture.all(),
                 ids=lambda f: f.__name__)
def generic_mesh(request) -> type[GenericTessellationFixture]:
  """Parametrized over every registered `GenericTessellationFixture`.

  Add a new mesh shape by defining a new `GenericTessellationFixture`
  subclass anywhere it'll be imported (e.g. in this file); every test that
  takes this fixture will automatically run against it too.
  """
  return request.param
