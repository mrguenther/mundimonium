import math
import random

import pytest

from mundimonium.coordinates.hyperbolic_tessellation import (
    HyperbolicTessellation, _hyperbolic_distance, _minkowski_of_klein,
)
from mundimonium.coordinates.isometric import IsometricPoint


@pytest.fixture(scope="module")
def tess():
  return HyperbolicTessellation(order=7, rings=3)


@pytest.fixture(scope="module")
def tess8():
  return HyperbolicTessellation(order=8, rings=2)


def _random_point(face, rng):
  weights = [rng.random() for _ in range(3)]
  total = sum(weights)
  wb, ws, _ = [w / total for w in weights]
  return IsometricPoint(face, wb * face.altitude, ws * face.altitude)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_order_below_seven_is_rejected():
  with pytest.raises(ValueError):
    HyperbolicTessellation(order=6)


def test_every_face_has_the_tilings_edge_length(tess):
  for face in tess.faces:
    assert face.side_length == pytest.approx(tess._edge_length)


def test_interior_vertices_have_valence_equal_to_order(tess8):
  # A vertex is fully "interior" (its fan of `order` faces is complete)
  # once every face around it has been built -- true of every vertex
  # discovered at least one ring before the mesh's current edge.
  interior = [v for v in tess8.vertices if len(v.adjacent_faces()) == 8]
  assert len(interior) > 0
  boundary_valences = {
      len(v.adjacent_faces()) for v in tess8.vertices
      if len(v.adjacent_faces()) != 8
  }
  # Partially-completed frontier vertices only ever have 2 or 3 faces
  # (from the one or two already-built faces touching the edge that
  # discovered them) before their own fan is completed.
  assert boundary_valences <= {2, 3}


def test_every_vertex_lies_on_the_hyperboloid(tess):
  for v in tess.vertices:
    x, y, z = v.projection_coordinates
    assert x * x + y * y - z * z == pytest.approx(-1.0, abs=1e-6)
    assert z <= -1.0


# ---------------------------------------------------------------------------
# Mesh coordinates (Klein disk)
# ---------------------------------------------------------------------------

def test_coords_at_point_round_trips_through_get_face_at_coords(tess):
  rng = random.Random(0)
  for _ in range(30):
    face = tess.faces[rng.randrange(len(tess.faces))]
    p = _random_point(face, rng)
    u, v = tess.coords_at_point(p)
    assert tess.get_face_at_coords(u, v) is face


def test_new_point_at_coords_round_trips_coords_at_point(tess):
  rng = random.Random(1)
  for _ in range(30):
    face = tess.faces[rng.randrange(len(tess.faces))]
    p = _random_point(face, rng)
    u, v = tess.coords_at_point(p)
    p2 = tess.new_point_at_coords(u, v)
    assert p2.grid is face
    assert p2.b == pytest.approx(p.b, abs=1e-6)
    assert p2.s == pytest.approx(p.s, abs=1e-6)


# ---------------------------------------------------------------------------
# geodesic_distance / shortest_path_by_segment
# ---------------------------------------------------------------------------

def test_geodesic_distance_same_face_matches_minkowski_distance_directly(tess):
  # Deliberately *not* compared against `p1.distance_from(p2)`
  # (`IsometricGrid.local_distance`'s flat local-frame formula): unlike
  # `GenericTessellation`, where that flat formula *is* the definition of
  # geodesic distance, here `geodesic_distance` uses the tessellation's
  # exact global embedding instead, which is not the same thing even for
  # two points on the same face -- a hyperbolic triangle's interior isn't
  # isometric to a flat one, only its 3 vertices are (by construction, to
  # the tiling's own edge length). This test instead cross-checks against
  # an independent, direct Minkowski-model computation.
  face = tess.faces[0]
  alt = face.altitude
  p1 = IsometricPoint(face, 0.2 * alt, 0.3 * alt)
  p2 = IsometricPoint(face, 0.5 * alt, 0.1 * alt)
  expected = _hyperbolic_distance(
      _minkowski_of_klein(*tess.coords_at_point(p1)),
      _minkowski_of_klein(*tess.coords_at_point(p2)))
  assert tess.geodesic_distance(p1, p2) == pytest.approx(expected)


def test_geodesic_distance_is_symmetric(tess):
  rng = random.Random(2)
  for _ in range(30):
    fa = tess.faces[rng.randrange(len(tess.faces))]
    fb = tess.faces[rng.randrange(len(tess.faces))]
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    d12 = tess.geodesic_distance(p1, p2)
    d21 = tess.geodesic_distance(p2, p1)
    assert d12 == pytest.approx(d21, rel=1e-6)


def test_shortest_path_by_segment_same_face_is_a_single_segment(tess):
  face = tess.faces[0]
  alt = face.altitude
  p1 = IsometricPoint(face, 0.1 * alt, 0.1 * alt)
  p2 = IsometricPoint(face, 0.3 * alt, 0.2 * alt)
  assert tess.shortest_path_by_segment(p1, p2) == [(p1, p2)]


def test_shortest_path_by_segment_endpoints_match_request(tess):
  rng = random.Random(3)
  for _ in range(20):
    fa = tess.faces[rng.randrange(len(tess.faces))]
    fb = tess.faces[rng.randrange(len(tess.faces))]
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    segs = tess.shortest_path_by_segment(p1, p2)
    assert segs[0][0] is p1
    assert segs[-1][1] is p2


def test_shortest_path_by_segment_boundary_crossings_agree_across_faces(tess):
  # Consecutive segments must "hand off" at the same physical (Klein)
  # location, each expressed in its own face's local coordinates.
  rng = random.Random(4)
  for _ in range(20):
    fa = tess.faces[rng.randrange(len(tess.faces))]
    fb = tess.faces[rng.randrange(len(tess.faces))]
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    segs = tess.shortest_path_by_segment(p1, p2)
    for (_, exit_point), (entry_point, _) in zip(segs, segs[1:]):
      exit_uv = tess.coords_at_point(exit_point)
      entry_uv = tess.coords_at_point(entry_point)
      assert exit_uv[0] == pytest.approx(entry_uv[0], abs=1e-6)
      assert exit_uv[1] == pytest.approx(entry_uv[1], abs=1e-6)


def test_geodesic_distance_matches_sum_of_segment_hyperbolic_lengths(tess):
  # Cross-checks the O(1) direct distance formula against the segment-by-
  # segment walk, computing each segment's length independently via the
  # Minkowski-model distance formula (not `IsometricPoint.distance_from`,
  # which uses the flat local-frame formula -- correct within one face,
  # but a genuinely independent check here).
  rng = random.Random(5)
  for _ in range(15):
    fa = tess.faces[rng.randrange(len(tess.faces))]
    fb = tess.faces[rng.randrange(len(tess.faces))]
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    direct = tess.geodesic_distance(p1, p2)
    segs = tess.shortest_path_by_segment(p1, p2)
    total = 0.0
    for a, b in segs:
      pa = _minkowski_of_klein(*tess.coords_at_point(a))
      pb = _minkowski_of_klein(*tess.coords_at_point(b))
      total += _hyperbolic_distance(pa, pb)
    assert total == pytest.approx(direct, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# geodesically_canonicalize_point
# ---------------------------------------------------------------------------

def test_geodesically_canonicalize_point_is_idempotent_when_in_bounds(tess):
  face = tess.faces[2]
  alt = face.altitude
  p = IsometricPoint(face, 0.3 * alt, 0.3 * alt)
  tess.geodesically_canonicalize_point(p)
  assert p.grid is face
  assert p.b == pytest.approx(0.3 * alt)
  assert p.s == pytest.approx(0.3 * alt)


def test_geodesically_canonicalize_point_lands_in_bounds():
  # A fresh, unshared tessellation (rather than the module-scoped `tess`
  # fixture): repeatedly canonicalizing large out-of-bounds displacements
  # grows the mesh outward each time, and this test's own moderate scale
  # range is only reliable near the origin -- picking a face from a mesh
  # some *other* test had already extended far out (deep into the regime
  # where Klein-disk floating-point precision genuinely breaks down, an
  # inherent limitation of any disk model at large hyperbolic radii, not a
  # bug) would make this test's pass/fail depend on unrelated test order.
  tess = HyperbolicTessellation(order=7, rings=2)
  rng = random.Random(6)
  for _ in range(30):
    face = tess.faces[rng.randrange(len(tess.faces))]
    alt = face.altitude
    scale = rng.uniform(0.05, 0.8)
    angle = rng.uniform(0, 2 * math.pi)
    wb = 1.0 / 3.0 + scale * math.cos(angle)
    ws = 1.0 / 3.0 + scale * math.sin(angle)
    p = IsometricPoint(face, wb * alt, ws * alt)
    tess.geodesically_canonicalize_point(p)
    wb2, ws2, wd2 = p.barycentric
    assert -1e-6 <= wb2 <= 1 + 1e-6
    assert -1e-6 <= ws2 <= 1 + 1e-6
    assert -1e-6 <= wd2 <= 1 + 1e-6


# ---------------------------------------------------------------------------
# extend / extend_to_include
# ---------------------------------------------------------------------------

def test_extend_grows_the_mesh_and_preserves_existing_faces_and_vertices():
  tess = HyperbolicTessellation(order=7, rings=2)
  faces_before = list(tess.faces)
  vertices_before = list(tess.vertices)
  tess.extend(rings=1)
  assert len(tess.faces) > len(faces_before)
  assert len(tess.vertices) > len(vertices_before)
  assert all(f in tess.faces for f in faces_before)
  assert all(v in tess.vertices for v in vertices_before)


def test_extend_to_include_covers_a_distant_point():
  tess = HyperbolicTessellation(order=7, rings=1)
  far_point = _minkowski_of_klein(0.9, 0.0)
  tess.extend_to_include(far_point)
  # A face lookup for the same Klein coordinates should now succeed
  # without needing to extend further mid-walk.
  faces_before = list(tess.faces)
  tess.get_face_at_coords(0.9, 0.0)
  assert tess.faces == faces_before


def test_get_face_at_coords_auto_extends_the_mesh_for_a_distant_target():
  tess = HyperbolicTessellation(order=7, rings=1)
  face_count_before = len(tess.faces)
  face = tess.get_face_at_coords(0.9, 0.0)
  assert face is not None
  assert len(tess.faces) > face_count_before
