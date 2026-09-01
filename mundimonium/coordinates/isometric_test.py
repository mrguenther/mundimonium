import math

import pytest

from mundimonium.coordinates.isometric import (
    IsometricDirection, IsometricGrid, IsometricPoint, IsometricVector,
    isometric_distance,
)
from mundimonium.coordinates.nesting_iso_grid import isometric_to_cartesian


class FakeGrid(IsometricGrid):
  """Minimal concrete `IsometricGrid` with no cross-grid adjacency."""

  @classmethod
  def nearby_grid_distance(cls, p1, p2):
    return None

  @classmethod
  def geodesic_distance(cls, p1, p2):
    return None


# ---------------------------------------------------------------------------
# isometric_distance
# ---------------------------------------------------------------------------

def test_isometric_distance_along_a_single_axis():
  assert isometric_distance(1.0, 0.0) == pytest.approx(1.0)
  assert isometric_distance(0.0, 1.0) == pytest.approx(1.0)


def test_isometric_distance_of_zero():
  assert isometric_distance(0.0, 0.0) == pytest.approx(0.0)


def test_isometric_distance_matches_law_of_cosines_at_60_degrees():
  a, b = 3.0, 5.0
  expected = math.sqrt(a**2 + b**2 - a * b)  # 2*cos(60deg) == 1
  assert isometric_distance(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# IsometricDirection
# ---------------------------------------------------------------------------

def test_direction_values_are_b_s_d_in_order():
  assert IsometricDirection.B.value == 0
  assert IsometricDirection.S.value == 1
  assert IsometricDirection.D.value == 2


def test_rotated_cw_cycles_b_s_d():
  assert IsometricDirection.B.rotated_cw_by_index(1) == IsometricDirection.S
  assert IsometricDirection.S.rotated_cw_by_index(1) == IsometricDirection.D
  assert IsometricDirection.D.rotated_cw_by_index(1) == IsometricDirection.B


def test_rotated_ccw_is_inverse_of_rotated_cw():
  for direction in IsometricDirection:
    for index in range(3):
      assert direction.rotated_cw_by_index(index).rotated_ccw_by_index(
          index) == direction


def test_rotated_by_3_is_identity():
  for direction in IsometricDirection:
    assert direction.rotated_cw_by_index(3) == direction


def test_repr_and_str_are_lowercase_letters():
  assert repr(IsometricDirection.B) == "b"
  assert repr(IsometricDirection.S) == "s"
  assert repr(IsometricDirection.D) == "d"
  assert str(IsometricDirection.D) == "d"


# ---------------------------------------------------------------------------
# IsometricGrid construction
# ---------------------------------------------------------------------------

def test_grid_requires_exactly_one_size_argument():
  with pytest.raises(ValueError):
    FakeGrid()
  with pytest.raises(ValueError):
    FakeGrid(side_length=1.0, altitude=1.0)


def test_grid_derives_consistent_sizes_from_side_length():
  grid = FakeGrid(side_length=2.0)
  assert grid.side_length == pytest.approx(2.0)
  assert grid.altitude == pytest.approx(2.0 * math.sqrt(3) / 2)
  assert grid.apothem == pytest.approx(grid.altitude / 3)


def test_grid_derives_consistent_sizes_from_altitude():
  grid = FakeGrid(altitude=3.0)
  assert grid.altitude == pytest.approx(3.0)
  # Round-tripping through side_length should reproduce the same altitude.
  reconstructed = FakeGrid(side_length=grid.side_length)
  assert reconstructed.altitude == pytest.approx(3.0)


def test_grid_derives_consistent_sizes_from_apothem():
  grid = FakeGrid(apothem=1.0)
  assert grid.apothem == pytest.approx(1.0)
  assert grid.altitude == pytest.approx(3.0 * grid.apothem)


# ---------------------------------------------------------------------------
# IsometricGrid distance dispatch / common-grid lookups
# ---------------------------------------------------------------------------

def test_local_distance_requires_same_grid():
  g1, g2 = FakeGrid(side_length=1.0), FakeGrid(side_length=1.0)
  p1 = IsometricPoint(g1, 0.1, 0.1)
  p2 = IsometricPoint(g2, 0.1, 0.1)
  assert IsometricGrid.local_distance(p1, p2) is None


def test_distance_uses_local_distance_on_shared_grid():
  grid = FakeGrid(side_length=1.0)
  p1 = IsometricPoint(grid, 0.1, 0.1)
  p2 = IsometricPoint(grid, 0.4, 0.1)
  b_comp, s_comp = p2.b - p1.b, p2.s - p1.s
  expected = isometric_distance(b_comp - 0.5 * s_comp, s_comp - 0.5 * b_comp)
  assert grid.distance(p1, p2) == pytest.approx(
      grid.local_distance(p1, p2))
  assert grid.distance(p1, p2) == pytest.approx(expected)


class OtherFakeGrid(IsometricGrid):
  @classmethod
  def nearby_grid_distance(cls, p1, p2):
    return None

  @classmethod
  def geodesic_distance(cls, p1, p2):
    return None


def test_common_grid_type_same_type():
  g1, g2 = FakeGrid(side_length=1.0), FakeGrid(side_length=1.0)
  p1, p2 = IsometricPoint(g1, 0, 0), IsometricPoint(g2, 0, 0)
  assert IsometricGrid.common_grid_type(p1, p2) is FakeGrid


def test_common_grid_type_falls_back_to_shared_base():
  g1, g2 = FakeGrid(side_length=1.0), OtherFakeGrid(side_length=1.0)
  p1, p2 = IsometricPoint(g1, 0, 0), IsometricPoint(g2, 0, 0)
  assert IsometricGrid.common_grid_type(p1, p2) is IsometricGrid


def test_common_grid_type_requires_at_least_one_point():
  with pytest.raises(ValueError):
    IsometricGrid.common_grid_type()


def test_common_grid_returns_shared_grid_or_none():
  grid = FakeGrid(side_length=1.0)
  other = FakeGrid(side_length=1.0)
  p1 = IsometricPoint(grid, 0.1, 0.1)
  p2 = IsometricPoint(grid, 0.2, 0.2)
  p3 = IsometricPoint(other, 0.1, 0.1)
  assert IsometricGrid.common_grid(p1, p2) is grid
  assert IsometricGrid.common_grid(p1, p3) is None
  assert IsometricGrid.common_grid() is None


def test_project_onto_root_grid_is_identity_for_a_root_grid():
  grid = FakeGrid(side_length=1.0)
  p = IsometricPoint(grid, 0.2, 0.3)
  projected = grid.project_onto_root_grid(p)
  assert projected.grid is grid
  assert projected.b == pytest.approx(p.b)
  assert projected.s == pytest.approx(p.s)
  assert projected is not p  # copy by default


def test_project_onto_root_grid_in_place_returns_same_object():
  grid = FakeGrid(side_length=1.0)
  p = IsometricPoint(grid, 0.2, 0.3)
  assert grid.project_onto_root_grid(p, in_place=True) is p


def test_canonicalize_point_default_is_identity():
  grid = FakeGrid(side_length=1.0)
  p = IsometricPoint(grid, 0.2, 0.3)
  assert FakeGrid.canonicalize_point(p) is p


def test_to_mesh_coordinates_default_raises():
  grid = FakeGrid(side_length=1.0)
  p = IsometricPoint(grid, 0.2, 0.3)
  with pytest.raises(NotImplementedError):
    grid.to_mesh_coordinates(p)


# ---------------------------------------------------------------------------
# IsometricPoint construction and (b, s, d) <-> barycentric
# ---------------------------------------------------------------------------

@pytest.fixture
def grid():
  return FakeGrid(side_length=1.0)


def test_center_is_the_centroid(grid):
  p = IsometricPoint.center(grid)
  assert p.b == pytest.approx(grid.apothem)
  assert p.s == pytest.approx(grid.apothem)
  assert p.d == pytest.approx(grid.apothem)


def test_d_is_derived_from_b_and_s(grid):
  p = IsometricPoint(grid, 0.1, 0.2)
  assert p.d == pytest.approx(grid.altitude - 0.1 - 0.2)


def test_at_coordinates_from_two_of_three(grid):
  p = IsometricPoint.at_coordinates(grid, b=0.1, s=0.2)
  assert p.b == pytest.approx(0.1)
  assert p.s == pytest.approx(0.2)

  p2 = IsometricPoint.at_coordinates(grid, s=0.2, d=0.3)
  assert p2.s == pytest.approx(0.2)
  assert p2.d == pytest.approx(0.3)
  assert p2.b == pytest.approx(grid.altitude - 0.2 - 0.3)


def test_at_coordinates_requires_exactly_two_args(grid):
  with pytest.raises(AssertionError):
    IsometricPoint.at_coordinates(grid, b=0.1)
  with pytest.raises(AssertionError):
    IsometricPoint.at_coordinates(grid, b=0.1, s=0.2, d=0.3)


def test_from_barycentric_and_back(grid):
  p = IsometricPoint.from_barycentric(grid, 0.2, 0.3)
  wb, ws, wd = p.barycentric
  assert wb == pytest.approx(0.2)
  assert ws == pytest.approx(0.3)
  assert wd == pytest.approx(0.5)
  assert wb + ws + wd == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# IsometricPoint (b, s, d) setters: each moves purely along its own axis,
# leaving the Cartesian position along the other two axes unchanged.
# ---------------------------------------------------------------------------

def test_setting_b_preserves_cartesian_x(grid):
  p = IsometricPoint(grid, 0.3, 0.2)
  x0, y0 = isometric_to_cartesian(p.b, p.s, p.d)
  p.b = 0.6
  x1, y1 = isometric_to_cartesian(p.b, p.s, p.d)
  assert x1 == pytest.approx(x0)
  assert y1 == pytest.approx(0.6)
  assert y1 != pytest.approx(y0)


def test_setting_s_preserves_b_minus_d(grid):
  p = IsometricPoint(grid, 0.3, 0.2)
  bd0 = p.b - p.d
  p.s = 0.5
  assert p.s == pytest.approx(0.5)
  assert (p.b - p.d) == pytest.approx(bd0)


def test_setting_d_preserves_s_minus_b(grid):
  p = IsometricPoint(grid, 0.3, 0.2)
  sb0 = p.s - p.b
  p.d = 0.5
  assert p.d == pytest.approx(0.5)
  assert (p.s - p.b) == pytest.approx(sb0)


def test_getitem_and_setitem_by_direction(grid):
  p = IsometricPoint(grid, 0.1, 0.2)
  assert p[IsometricDirection.B] == pytest.approx(p.b)
  assert p[IsometricDirection.S] == pytest.approx(p.s)
  assert p[IsometricDirection.D] == pytest.approx(p.d)

  p[IsometricDirection.S] = 0.4
  assert p.s == pytest.approx(0.4)


def test_move_to_repositions_from_two_coordinates(grid):
  p = IsometricPoint(grid, 0.0, 0.0)
  p.move_to(b=0.2, s=0.3)
  assert p.b == pytest.approx(0.2)
  assert p.s == pytest.approx(0.3)


def test_move_to_requires_exactly_two_args(grid):
  p = IsometricPoint(grid, 0.0, 0.0)
  with pytest.raises(AssertionError):
    p.move_to(b=0.1)


# ---------------------------------------------------------------------------
# Vector arithmetic and point <-> vector operators
# ---------------------------------------------------------------------------

def test_vector_between_points_and_addition_round_trip(grid):
  p1 = IsometricPoint(grid, 0.1, 0.1)
  p2 = IsometricPoint(grid, 0.5, 0.2)
  v = IsometricVector.between_points(p1, p2)
  result = p1 + v
  assert result.b == pytest.approx(p2.b)
  assert result.s == pytest.approx(p2.s)


def test_point_minus_point_is_vector(grid):
  p1 = IsometricPoint(grid, 0.1, 0.1)
  p2 = IsometricPoint(grid, 0.5, 0.2)
  v = p2 - p1
  assert isinstance(v, IsometricVector)
  assert (p1 + v).b == pytest.approx(p2.b)
  assert (p1 + v).s == pytest.approx(p2.s)


def test_point_minus_vector_is_point(grid):
  p = IsometricPoint(grid, 0.5, 0.4)
  v = IsometricVector.with_net_b_s(0.1, 0.1)
  result = p - v
  assert isinstance(result, IsometricPoint)


def test_sub_rejects_unknown_type(grid):
  p = IsometricPoint(grid, 0.1, 0.1)
  with pytest.raises(TypeError):
    p - 5


def test_add_rejects_non_vector(grid):
  p = IsometricPoint(grid, 0.1, 0.1)
  with pytest.raises(AssertionError):
    p + 5


def test_iadd_and_isub_mutate_in_place(grid):
  p = IsometricPoint(grid, 0.1, 0.1)
  v = IsometricVector.with_net_b_s(0.05, 0.05)
  before_b = p.b
  result = p.__iadd__(v)
  assert result is p
  assert p.b != pytest.approx(before_b)

  p2 = IsometricPoint(grid, 0.1, 0.1)
  result2 = p2.__isub__(v)
  assert result2 is p2


def test_distance_from_matches_local_distance(grid):
  p1 = IsometricPoint(grid, 0.1, 0.1)
  p2 = IsometricPoint(grid, 0.4, 0.1)
  assert p1.distance_from(p2) == pytest.approx(grid.local_distance(p1, p2))
  assert p1.distance_from(p2) == pytest.approx(p2.distance_from(p1))


def test_repr_and_str(grid):
  p = IsometricPoint(grid, 0.1, 0.2)
  assert str(p) == f"({p.b}, {p.s}, {p.d})"
  assert "in grid" in repr(p)


def test_points_are_hashable_and_value_equal(grid):
  p1 = IsometricPoint(grid, 0.1, 0.2)
  p2 = IsometricPoint(grid, 0.1, 0.2)
  p3 = IsometricPoint(grid, 0.9, 0.2)
  assert p1 == p2
  assert hash(p1) == hash(p2)
  assert p1 != p3


# ---------------------------------------------------------------------------
# IsometricVector: components, deltas, and their (de)coupling
# ---------------------------------------------------------------------------

def test_with_net_b_s_reconstructs_delta_b_and_delta_s():
  v = IsometricVector.with_net_b_s(2.0, 1.0)
  assert v.delta_b == pytest.approx(2.0)
  assert v.delta_s == pytest.approx(1.0)


def test_with_net_b_d_reconstructs_delta_b_and_delta_d():
  v = IsometricVector.with_net_b_d(2.0, 1.0)
  assert v.delta_b == pytest.approx(2.0)
  assert v.delta_d == pytest.approx(1.0)


def test_with_net_s_d_reconstructs_delta_s_and_delta_d():
  v = IsometricVector.with_net_s_d(2.0, 1.0)
  assert v.delta_s == pytest.approx(2.0)
  assert v.delta_d == pytest.approx(1.0)


def test_delta_b_s_d_sum_relationship():
  # b_component/s_component and delta_b/delta_s/delta_d always satisfy this,
  # since d_component is constrained to 0.
  v = IsometricVector(1.5, -0.5)
  assert (v.delta_b + v.delta_s + v.delta_d) == pytest.approx(0.0)


def test_d_component_is_always_zero_and_cannot_be_set():
  v = IsometricVector(1.0, 2.0)
  assert v.d_component == 0
  with pytest.raises(TypeError):
    v.d_component = 1.0


def test_delta_setters_are_self_consistent_and_preserve_the_zero_sum():
  # Only 2 of (delta_b, delta_s, delta_d) are independent (they always sum to
  # 0, since d_component is fixed at 0), so setting one necessarily perturbs
  # the other two -- but the one being set must land exactly on its target.
  v = IsometricVector.with_net_b_s(1.0, 1.0)
  v.delta_b = 3.0
  assert v.delta_b == pytest.approx(3.0)
  assert (v.delta_b + v.delta_s + v.delta_d) == pytest.approx(0.0)

  v2 = IsometricVector.with_net_b_s(1.0, 1.0)
  v2.delta_s = -2.0
  assert v2.delta_s == pytest.approx(-2.0)
  assert (v2.delta_b + v2.delta_s + v2.delta_d) == pytest.approx(0.0)

  v3 = IsometricVector.with_net_b_d(1.0, 1.0)
  v3.delta_d = 5.0
  assert v3.delta_d == pytest.approx(5.0)
  assert (v3.delta_b + v3.delta_s + v3.delta_d) == pytest.approx(0.0)


def test_unit_vector_has_length_one():
  v = IsometricVector(3.0, -2.0)
  u = v.unit_vector()
  assert u.length == pytest.approx(1.0)


def test_length_matches_isometric_distance_between_delta_b_and_delta_s():
  v = IsometricVector(3.0, 0.0)
  expected = isometric_distance(
      v.delta_b - 0.5 * v.delta_s, v.delta_s - 0.5 * v.delta_b)
  assert v.length == pytest.approx(expected)


def test_length_setter_rescales_and_is_self_consistent():
  v = IsometricVector(3.0, 0.0)
  original_length = v.length
  v.length = 2 * original_length
  assert v.length == pytest.approx(2 * original_length)


def test_vector_length_between_two_points_matches_distance_from(grid):
  # The fundamental point/vector relationship: subtracting two points gives
  # a vector whose length equals the geodesic distance between them.
  p1 = IsometricPoint(grid, 0.1, 0.6)
  p2 = IsometricPoint(grid, 0.7, 0.2)
  v = p2 - p1
  assert v.length == pytest.approx(p1.distance_from(p2))
  assert v.length == pytest.approx(p2.distance_from(p1))


def test_vector_arithmetic_operators():
  v1 = IsometricVector(1.0, 2.0)
  v2 = IsometricVector(0.5, 0.5)
  assert (v1 + v2).b_component == pytest.approx(1.5)
  assert (v1 - v2).s_component == pytest.approx(1.5)
  assert (v1 * 2).b_component == pytest.approx(2.0)
  assert (v1 / 2).s_component == pytest.approx(1.0)


def test_vector_getitem_setitem_by_direction():
  v = IsometricVector.with_net_b_s(1.0, 2.0)
  assert v[IsometricDirection.B] == pytest.approx(v.delta_b)
  v[IsometricDirection.D] = 4.0
  assert v.delta_d == pytest.approx(4.0)


def test_vector_repr_and_str():
  v = IsometricVector.with_net_b_s(1.0, 2.0)
  assert repr(v) == f"<{v.delta_b}, {v.delta_s}, {v.delta_d}>"
  assert str(v) == repr(v)
