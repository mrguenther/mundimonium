"""Shared pytest fixtures for mundimonium/coordinates/*_test.py.

Actual tests live in each module's own `<module>_test.py`; this file only
holds setup that's identical across several of them.
"""

import pytest

from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.isometric import IsometricDirection
from mundimonium.coordinates.stellated_icosahedron import (
    build_icosahedron, build_stellated_icosahedron,
)
from mundimonium.coordinates.tessellation import TessellationFace, TessellationVertex


@pytest.fixture(scope="module")
def icosahedron():
  return build_icosahedron()


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
