import pytest

from mundimonium.coordinates.conftest import build_icosahedron
from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace, LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.coordinates.tessellation import TessellationVertex


@pytest.fixture(scope="module")
def sphere():
  return SphericalTessellation(radius=1.0, frequency=1, face_type=LodMeshFace)


@pytest.fixture(scope="module")
def lod_icosahedron():
  return build_icosahedron(face_type=LodMeshFace)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_faces_are_both_lod_mesh_face_and_sector(sphere):
  face = sphere._faces[0]
  assert isinstance(face, LodMeshFace)
  assert isinstance(face, LodMeshSector)


def test_top_level_face_child_type_is_plain_sector_not_face(sphere):
  face = sphere._faces[0]
  assert face.child_type is LodMeshSector


def test_subdivided_children_are_plain_sectors(sphere):
  face = sphere._faces[1]
  face.subdivide(resolution=2)
  for child in face.children:
    assert type(child) is LodMeshSector
    assert not isinstance(child, LodMeshFace)
  assert child.root is face
  assert child.parent is face


# ---------------------------------------------------------------------------
# to_mesh_coordinates
# ---------------------------------------------------------------------------

def test_face_to_mesh_coordinates_matches_tessellation_directly(sphere):
  face = sphere._faces[2]
  point = face.centroid_local_coords
  assert face.to_mesh_coordinates(point) == pytest.approx(
      sphere.coords_at_point(point))


def test_sector_to_mesh_coordinates_matches_projection_to_root(sphere):
  face = sphere._faces[3]
  face.subdivide(resolution=3)
  child = face.children[2]
  point = IsometricPoint.center(child)
  expected = sphere.coords_at_point(point.project_onto_root_grid())
  assert child.to_mesh_coordinates(point) == pytest.approx(expected)


def test_point_to_mesh_coordinates_agrees_with_grid_method(sphere):
  face = sphere._faces[4]
  face.subdivide(resolution=2)
  child = face.children[0]
  point = IsometricPoint.center(child)
  assert point.to_mesh_coordinates() == pytest.approx(
      child.to_mesh_coordinates(point))


# ---------------------------------------------------------------------------
# nearby_grid_distance
# ---------------------------------------------------------------------------

def test_nearby_grid_distance_within_the_same_lod_tree(sphere):
  face = sphere._faces[5]
  face.subdivide(resolution=2)
  c1, c2 = face.children[0], face.children[1]
  p1, p2 = IsometricPoint.center(c1), IsometricPoint.center(c2)
  dist = LodMeshSector.nearby_grid_distance(p1, p2)
  assert dist is not None
  assert dist > 0


def test_nearby_grid_distance_across_adjacent_faces(sphere):
  face = sphere._faces[6]
  neighbor = face.face_on_edge(IsometricDirection.B)
  face.subdivide(resolution=2)
  neighbor.subdivide(resolution=2)
  p1 = IsometricPoint.center(face.children[0])
  p2 = IsometricPoint.center(neighbor.children[0])
  dist = LodMeshSector.nearby_grid_distance(p1, p2)
  assert dist is not None
  assert dist > 0


def test_nearby_grid_distance_across_unrelated_faces_is_none(sphere):
  face_a = sphere._faces[7]
  face_b = sphere._faces[15]  # not adjacent to face_a on this icosahedron
  face_a.subdivide(resolution=2)
  face_b.subdivide(resolution=2)
  p1 = IsometricPoint.center(face_a.children[0])
  p2 = IsometricPoint.center(face_b.children[0])
  assert LodMeshSector.nearby_grid_distance(p1, p2) is None


def test_nearby_grid_distance_rejects_non_lod_mesh_sector_grids():
  class NotASector:
    grid = None

  assert LodMeshSector.nearby_grid_distance(NotASector(), NotASector()) is None


# ---------------------------------------------------------------------------
# geodesic_distance
# ---------------------------------------------------------------------------

def test_geodesic_distance_delegates_to_root_tessellation(sphere):
  face = sphere._faces[8]
  neighbor = face.face_on_edge(IsometricDirection.S)
  face.subdivide(resolution=2)
  neighbor.subdivide(resolution=2)
  p1 = IsometricPoint.center(face.children[0])
  p2 = IsometricPoint.center(neighbor.children[0])

  expected = sphere.geodesic_distance(
      p1.project_onto_root_grid(), p2.project_onto_root_grid())
  assert LodMeshSector.geodesic_distance(p1, p2) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LodMeshSector.canonicalize_point / LodMeshFace.canonicalize_point
# ---------------------------------------------------------------------------

def test_sector_canonicalize_point_is_noop_when_in_bounds(sphere):
  face = sphere._faces[0]
  face.subdivide(resolution=2)
  child = face.children[0]
  p = IsometricPoint(child, 0.1 * child.altitude, 0.1 * child.altitude)
  result = LodMeshSector.canonicalize_point(p)
  assert result is p
  assert p.grid is child


def test_face_canonicalize_point_single_edge_crossing_matches_projection(sphere):
  face = sphere._faces[9]
  neighbor = face.face_on_edge(IsometricDirection.S)
  alt = face.altitude
  p = IsometricPoint(face, 0.3 * alt, -0.01 * alt)
  expected = IsometricPoint(face, p.b, p.s).project_onto_adjacent_grid(neighbor)

  LodMeshFace.canonicalize_point(p)

  assert p.grid is expected.grid
  assert p.b == pytest.approx(expected.b)
  assert p.s == pytest.approx(expected.s)


def test_sector_canonicalize_point_moves_to_a_sibling_within_the_same_face(
    sphere):
  face = sphere._faces[10]
  face.subdivide(resolution=4)
  # (i_b=1, i_s=0) borders an interior sibling edge, not the face's own
  # outer edge, so a small negative excursion should land on that sibling.
  child = face.child_at(1, 0, False)
  alt = child.altitude
  p = IsometricPoint(child, -0.01 * alt, 0.3 * alt)
  expected_root = child.to_root_isometric(p.b, p.s, p.d)

  LodMeshSector.canonicalize_point(p)

  assert p.grid is not child
  assert p.grid.parent is face
  assert p.grid.to_root_isometric(p.b, p.s, p.d) == pytest.approx(
      expected_root, abs=1e-9)
  assert -1e-9 <= p.b <= p.grid.altitude + 1e-9
  assert -1e-9 <= p.s <= p.grid.altitude + 1e-9
  assert -1e-9 <= p.d <= p.grid.altitude + 1e-9


def test_sector_canonicalize_point_escapes_to_a_different_top_level_face():
  # Regression test covering both bugs fixed this session together: the
  # `LodMeshSector.canonicalize_point` classmethod-staleness bug (crossing a
  # mesh edge into a sibling `LodMeshFace`) and the auto-subdivision path
  # (the target face hasn't been subdivided yet, so it must use its own
  # `default_resolution`).
  # Uses a dedicated sphere (rather than the shared `sphere` fixture) so no
  # other test's subdivisions can interfere with `neighbor`'s state.
  sphere = SphericalTessellation(radius=1.0, frequency=1, face_type=LodMeshFace)
  face = sphere._faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
  face.subdivide(resolution=2)
  child = face.children[0]
  assert neighbor.children is None
  alt = child.altitude

  # Just past face's own B edge (i_b=0 for this child, so the small negative
  # excursion carries straight through to the face level).
  p = IsometricPoint(child, -0.01 * alt, 0.3 * alt)

  # Ground truth: the exact face-to-face projection across the shared edge,
  # computed independently of `LodMeshSector.canonicalize_point`.
  face_b, face_s, _ = child.to_root_isometric(p.b, p.s, p.d)
  expected = IsometricPoint(face, face_b, face_s).project_onto_adjacent_grid(
      neighbor)

  LodMeshSector.canonicalize_point(p)

  assert p.grid.root is neighbor
  assert neighbor.children is not None  # auto-subdivided
  assert neighbor.resolution == 2  # LodMeshFace's own default_resolution
  assert p.grid.depth == 1
  landed_b, landed_s, _ = p.grid.to_root_isometric(p.b, p.s, p.d)
  assert (landed_b, landed_s) == pytest.approx(
      (expected.b, expected.s), abs=1e-9)
  assert -1e-9 <= p.b <= p.grid.altitude + 1e-9
  assert -1e-9 <= p.s <= p.grid.altitude + 1e-9
  assert -1e-9 <= p.d <= p.grid.altitude + 1e-9


def test_sector_canonicalize_point_raises_at_mesh_boundary():
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0.5, 1, 0])
  tess = GenericTessellation(face_type=LodMeshFace)
  face = tess.add_face([v1, v2, v3])
  face.subdivide(resolution=2)
  child = face.children[0]
  alt = child.altitude
  p = IsometricPoint(child, -10 * alt, -10 * alt)
  with pytest.raises(EndOfMeshSurfaceException):
    LodMeshSector.canonicalize_point(p)


def test_sector_canonicalize_point_lands_in_a_valid_sector_far_out_of_bounds(
    lod_icosahedron):
  # Exercises the full integration end-to-end: a point far outside a nested
  # LOD sector must walk up to the enclosing `LodMeshFace`, escape across
  # the mesh via `GenericTessellation.geodesically_canonicalize_point`, and
  # then descend back into a sector at the same original depth -- the path
  # that previously had zero coverage for `GenericTessellation` faces.
  _, _, faces = lod_icosahedron
  face = faces[0]
  face.subdivide(resolution=2)
  child = face.children[0]
  alt = child.altitude
  p = IsometricPoint(child, -4.3 * alt, -2.1 * alt)

  p.canonicalize()

  assert isinstance(p.grid, LodMeshSector)
  assert p.grid.depth == child.depth
  assert -1e-6 <= p.b <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.s <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.d <= p.grid.altitude + 1e-6
