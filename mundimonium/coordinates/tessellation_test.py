import math

import pytest

from mundimonium.coordinates.exceptions import (
    EndOfMeshSurfaceException, NotAdjacentException,
)
from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex,
)


# `icosahedron` fixture (a small closed mesh shared across several test
# files) comes from conftest.py.


# ---------------------------------------------------------------------------
# Tessellation subclassing / __init_subclass__ validation
# ---------------------------------------------------------------------------

def test_registered_impls_include_generic_tessellation():
  assert GenericTessellation in Tessellation._impls


def test_subclass_may_not_shadow_base_init_kwargs():
  with pytest.raises(AttributeError):
    class BadTessellation(Tessellation):
      def __init__(self, *, vertex_type: type = None, **kwargs):
        super().__init__(**kwargs)

      def coords_at_point(self, point):
        raise NotImplementedError()

      def new_point_at_coords(self, *coords):
        raise NotImplementedError()

      def get_face_at_coords(self, *coords):
        raise NotImplementedError()

      def geodesic_distance(self, p1, p2):
        raise NotImplementedError()

      def shortest_path(self, p1, p2):
        raise NotImplementedError()

      def shortest_path_by_segment(self, p1, p2):
        raise NotImplementedError()

      def geodesically_canonicalize_point(self, point):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Vertex/face registration
# ---------------------------------------------------------------------------

def test_icosahedron_has_12_vertices_and_20_faces(icosahedron):
  tess, verts, faces = icosahedron
  assert len(tess._vertices) == 12
  assert len(tess._faces) == 20


def test_add_face_rejects_wrong_vertex_count():
  tess = GenericTessellation()
  v1, v2 = TessellationVertex([0, 0, 0]), TessellationVertex([1, 0, 0])
  with pytest.raises(ValueError):
    tess.add_face([v1, v2])


def test_add_face_side_length_and_area_are_mutually_exclusive():
  tess = GenericTessellation()
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0, 1, 0])
  with pytest.raises(ValueError):
    tess.add_face([v1, v2, v3], side_length=1.0, area=1.0)


def test_add_face_default_side_length():
  tess = GenericTessellation(face_side_length=2.0)
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0, 1, 0])
  face = tess.add_face([v1, v2, v3])
  assert face.side_length == pytest.approx(2.0)


def test_add_face_explicit_side_length():
  tess = GenericTessellation()
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0, 1, 0])
  face = tess.add_face([v1, v2, v3], side_length=3.0)
  assert face.side_length == pytest.approx(3.0)


def test_add_face_from_area():
  tess = GenericTessellation()
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0, 1, 0])
  side_length = 2.0
  area = (math.sqrt(3) / 4) * side_length ** 2
  face = tess.add_face([v1, v2, v3], area=area)
  assert face.side_length == pytest.approx(side_length)


def test_add_face_from_linear_distortion_factor():
  tess = GenericTessellation(face_side_length=2.0)
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0, 1, 0])
  face = tess.add_face([v1, v2, v3], linear_distortion_factor=1.5)
  assert face.side_length == pytest.approx(3.0)


def test_vertices_and_faces_are_read_only_views_over_the_private_lists(
    icosahedron):
  tess, _, _ = icosahedron
  assert list(tess.vertices) == tess._vertices
  assert list(tess.faces) == tess._faces
  assert tess.vertices[0] is tess._vertices[0]
  assert tess.num_vertices == len(tess._vertices)
  assert tess.num_faces == len(tess._faces)
  with pytest.raises(TypeError):
    tess.vertices[0] = TessellationVertex([0, 0, 0])


def test_registering_the_same_face_twice_is_a_noop(icosahedron):
  tess, verts, faces = icosahedron
  before = len(tess._faces)
  tess.register_face(faces[0])
  assert len(tess._faces) == before


def test_register_face_rejects_wrong_type():
  tess = GenericTessellation()
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0, 1, 0])
  other_tess = GenericTessellation()
  face = other_tess.add_face([v1, v2, v3])

  class FussyFace(TessellationFace):
    pass

  tess2 = GenericTessellation(face_type=FussyFace)
  with pytest.raises(TypeError):
    tess2.register_face(face)


# ---------------------------------------------------------------------------
# TessellationVertex
# ---------------------------------------------------------------------------

def test_vertex_coordinates_and_projection(icosahedron):
  _, verts, _ = icosahedron
  v = verts[0]
  assert v.projection_coordinates == (v.x, v.y, v.z)


def test_setting_a_coordinate_updates_adjacent_face_centroids(icosahedron):
  tess, verts, faces = icosahedron
  face = verts[0].adjacent_faces()[0]
  before = face.centroid_projection_coords
  verts[0].x = verts[0].x + 5.0
  after = face.centroid_projection_coords
  assert after != before
  verts[0].x = verts[0].x - 5.0  # restore, since verts are module-scoped


def test_vertex_is_adjacent_to_its_faces_and_neighbors(icosahedron):
  _, verts, faces = icosahedron
  face = faces[0]
  assert face.vertex_b.is_adjacent_to(face)
  assert face.vertex_b.is_adjacent_to_face(face)
  assert face.vertex_b.is_adjacent_to(face.vertex_s)
  assert not face.vertex_b.is_adjacent_to(face.vertex_b)  # not adjacent to self


def test_vertex_is_adjacent_to_rejects_unknown_type(icosahedron):
  _, verts, _ = icosahedron
  with pytest.raises(TypeError):
    verts[0].is_adjacent_to(42)


# ---------------------------------------------------------------------------
# TessellationFace: adjacency and directions
# ---------------------------------------------------------------------------

def test_every_face_has_three_distinct_neighbors(icosahedron):
  _, _, faces = icosahedron
  for face in faces:
    neighbors = [face.face_on_edge(d) for d in IsometricDirection]
    assert all(n is not None for n in neighbors)
    assert len(set(id(n) for n in neighbors)) == 3
    assert face not in neighbors


def test_faces_are_mutually_adjacent(icosahedron):
  _, _, faces = icosahedron
  face = faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
  assert face.is_adjacent_to_face(neighbor)
  assert neighbor.is_adjacent_to_face(face)
  assert face.is_adjacent_to(neighbor)


def test_direction_away_from_face_and_face_on_edge_are_inverse(icosahedron):
  _, _, faces = icosahedron
  face = faces[0]
  for direction in IsometricDirection:
    neighbor = face.face_on_edge(direction)
    assert face.direction_away_from_face(neighbor) == direction


def test_direction_away_from_face_rejects_non_adjacent(icosahedron):
  _, _, faces = icosahedron
  # Faces 0 and 10 aren't adjacent in this construction.
  with pytest.raises(NotAdjacentException):
    faces[0].direction_away_from_face(faces[10])


def test_direction_toward_vertex_and_vertex_at_are_inverse(icosahedron):
  _, _, faces = icosahedron
  face = faces[0]
  for direction in IsometricDirection:
    vertex = face.vertex_at(direction)
    assert face.direction_toward_vertex(vertex) == direction


def test_direction_toward_vertex_rejects_non_bounding_vertex(icosahedron):
  _, verts, faces = icosahedron
  face = faces[0]
  outside_vertex = next(
      v for v in verts
      if v not in (face.vertex_b, face.vertex_s, face.vertex_d))
  with pytest.raises(NotAdjacentException):
    face.direction_toward_vertex(outside_vertex)


def test_face_is_adjacent_to_rejects_unknown_type(icosahedron):
  _, _, faces = icosahedron
  with pytest.raises(TypeError):
    faces[0].is_adjacent_to(42)


def test_centroid_local_coords_is_the_face_center(icosahedron):
  _, _, faces = icosahedron
  face = faces[0]
  center = face.centroid_local_coords
  assert isinstance(center, IsometricPoint)
  assert center.b == pytest.approx(face.apothem)
  assert center.s == pytest.approx(face.apothem)


# ---------------------------------------------------------------------------
# TessellationFace.canonicalize_point
# ---------------------------------------------------------------------------

def test_canonicalize_point_is_noop_when_already_in_bounds(icosahedron):
  _, _, faces = icosahedron
  face = faces[0]
  alt = face.altitude
  p = IsometricPoint(face, 0.3 * alt, 0.3 * alt)
  result = TessellationFace.canonicalize_point(p)
  assert result is p
  assert result.grid is face


def test_canonicalize_point_rejects_non_face_grid():
  class NotAFace:
    pass

  p = IsometricPoint(NotAFace(), 0.1, 0.1)
  with pytest.raises(TypeError):
    TessellationFace.canonicalize_point(p)


@pytest.mark.parametrize("edge", list(IsometricDirection))
def test_canonicalize_point_single_edge_crossing_matches_projection(
    icosahedron, edge):
  # Regression test: covering all three edges (not just B) is what catches
  # bugs like `canonicalize_point` picking the wrong edge for the S/D cases.
  _, _, faces = icosahedron
  face = faces[0]
  alt = face.altitude
  new_grid = face.face_on_edge(edge)

  # A point just past the given edge, but still within bounds along the
  # other two -- the "cheap" single-hop case.
  if edge == IsometricDirection.B:
    b, s = -0.01 * alt, 0.3 * alt
  elif edge == IsometricDirection.S:
    b, s = 0.3 * alt, -0.01 * alt
  else:  # D: push b + s just past alt, so d = alt - b - s is negative.
    b, s = 0.6 * alt, 0.41 * alt

  original = IsometricPoint(face, b, s)
  expected = IsometricPoint(face, b, s).project_onto_adjacent_grid(new_grid)

  p = IsometricPoint(face, b, s)
  TessellationFace.canonicalize_point(p)
  assert p.grid is expected.grid
  assert p.b == pytest.approx(expected.b)
  assert p.s == pytest.approx(expected.s)


def test_canonicalize_point_far_out_of_bounds_lands_in_a_valid_face(
    icosahedron):
  _, _, faces = icosahedron
  face = faces[0]
  alt = face.altitude
  # Far outside every edge at once -- forces the "expensive" geodesic path.
  p = IsometricPoint(face, -3.2 * alt, -1.7 * alt)
  TessellationFace.canonicalize_point(p)
  assert -1e-6 <= p.b <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.s <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.d <= p.grid.altitude + 1e-6


def test_canonicalize_point_at_mesh_boundary_raises():
  # A single triangle has no neighbors, so any out-of-bounds point should
  # signal that it fell off the mesh surface.
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0.5, 1, 0])
  tess = GenericTessellation()
  face = tess.add_face([v1, v2, v3])
  p = IsometricPoint(face, -0.5, 0.3)
  with pytest.raises(EndOfMeshSurfaceException):
    TessellationFace.canonicalize_point(p)


# ---------------------------------------------------------------------------
# Tessellation.distance()
# ---------------------------------------------------------------------------

def test_distance_delegates_to_face_type(icosahedron):
  tess, _, faces = icosahedron
  face = faces[0]
  p1 = face.centroid_local_coords
  p2 = IsometricPoint(face, face.apothem + 0.05, face.apothem)
  assert tess.distance(p1, p2) == pytest.approx(p1.distance_from(p2))
