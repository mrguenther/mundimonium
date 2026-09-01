import pytest

from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace, LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation


@pytest.fixture(scope="module")
def sphere():
  return SphericalTessellation(radius=1.0, frequency=1, face_type=LodMeshFace)


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
