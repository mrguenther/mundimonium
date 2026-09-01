import math

import pytest

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.nesting_iso_grid import (
    NestingIsoGrid, RenderItem, SectorItem, isometric_to_cartesian,
)


# ---------------------------------------------------------------------------
# isometric_to_cartesian
# ---------------------------------------------------------------------------

def test_isometric_to_cartesian_maps_b_to_y():
  assert isometric_to_cartesian(1.0, 0.0, 0.0) == pytest.approx((0.0, 1.0))


def test_isometric_to_cartesian_x_depends_on_d_minus_s():
  x, y = isometric_to_cartesian(0.0, 1.0, 2.0)
  assert x == pytest.approx((2.0 - 1.0) / math.sqrt(3))
  assert y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def test_sector_item_defaults():
  item = SectorItem(position=None, payload="x")
  assert item.min_scale == 0.0
  assert item.max_scale == float("inf")


def test_render_item_fields():
  item = RenderItem(x=1.0, y=2.0, payload="y")
  assert (item.x, item.y, item.payload) == (1.0, 2.0, "y")


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------

def test_root_grid_has_no_parent_and_is_its_own_root():
  grid = NestingIsoGrid(altitude=1.0)
  assert grid.parent is None
  assert grid.root is grid
  assert grid.indices == (0, 0, 0)
  assert not grid.inverted


def test_resolution_none_means_leaf_with_no_children():
  grid = NestingIsoGrid(altitude=1.0)
  assert grid.resolution is None
  assert grid.children is None


def test_construction_with_resolution_creates_n_squared_children():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  assert grid.resolution == 4
  assert len(grid.children) == 16


def test_child_type_defaults_to_own_class():
  assert NestingIsoGrid.child_type is NestingIsoGrid


# ---------------------------------------------------------------------------
# nearby_grid_distance / geodesic_distance
# ---------------------------------------------------------------------------

def test_nearby_grid_distance_within_the_same_tree():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  c1, c2 = grid.children[0], grid.children[1]
  p1 = IsometricPoint.center(c1)
  p2 = IsometricPoint.center(c2)
  dist = NestingIsoGrid.nearby_grid_distance(p1, p2)
  assert dist is not None
  assert dist >= 0


def test_nearby_grid_distance_across_unrelated_trees_is_none():
  grid_a = NestingIsoGrid(altitude=1.0)
  grid_b = NestingIsoGrid(altitude=1.0)
  p1 = IsometricPoint.center(grid_a)
  p2 = IsometricPoint.center(grid_b)
  assert NestingIsoGrid.nearby_grid_distance(p1, p2) is None


def test_geodesic_distance_is_not_applicable_at_this_level():
  grid = NestingIsoGrid(altitude=1.0)
  p = IsometricPoint.center(grid)
  assert NestingIsoGrid.geodesic_distance(p, p) is None


# ---------------------------------------------------------------------------
# Tree-position properties and indices
# ---------------------------------------------------------------------------

def test_i_d_matches_upward_and_inverted_formulas():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  for child in grid.children:
    N = grid.resolution
    expected = (N - 2 - child.i_b - child.i_s if child.inverted
                else N - 1 - child.i_b - child.i_s)
    assert child.i_d == expected


def test_indices_sum_to_resolution_minus_one_or_two():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  for child in grid.children:
    total = sum(child.indices)
    assert total == (grid.resolution - 2 if child.inverted
                      else grid.resolution - 1)


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

def test_add_item_appends_to_items():
  grid = NestingIsoGrid(altitude=1.0)
  item = SectorItem(position=IsometricPoint.center(grid), payload="a")
  grid.add_item(item)
  assert grid.items == [item]


def test_subdivide_a_leaf_creates_children():
  grid = NestingIsoGrid(altitude=1.0)
  assert grid.children is None
  grid.subdivide(resolution=3)
  assert len(grid.children) == 9


def test_subdividing_twice_raises():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  with pytest.raises(ValueError):
    grid.subdivide(resolution=3)


# ---------------------------------------------------------------------------
# Child lookup
# ---------------------------------------------------------------------------

def test_child_at_matches_the_children_list_by_indices():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  for child in grid.children:
    assert grid.child_at(child.i_b, child.i_s, child.inverted) is child


def test_child_at_out_of_range_raises_index_error():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  with pytest.raises(IndexError):
    grid.child_at(10, 10, False)


def test_child_at_no_inverted_child_on_the_outer_edge_raises():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  # (i_b=3, i_s=0) is the last row: no inverted child exists there.
  with pytest.raises(IndexError):
    grid.child_at(3, 0, True)


def test_child_containing_and_getitem_agree():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  p = IsometricPoint(grid, 0.05, 0.05)
  assert grid.child_containing(p) is grid[p]


def test_child_containing_lands_in_the_expected_upward_child():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  p = IsometricPoint(grid, 0.05, 0.05)
  child = grid.child_containing(p)
  assert child.i_b == 0
  assert child.i_s == 0
  assert not child.inverted


def test_child_containing_rejects_points_from_other_grids():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  other = NestingIsoGrid(altitude=1.0)
  p = IsometricPoint.center(other)
  with pytest.raises(KeyError):
    grid.child_containing(p)


def test_child_containing_requires_subdivision():
  grid = NestingIsoGrid(altitude=1.0)
  p = IsometricPoint.center(grid)
  with pytest.raises(ValueError):
    grid.child_containing(p)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def test_local_to_parent_and_parent_to_local_are_inverses():
  grid = NestingIsoGrid(resolution=3, altitude=1.0)
  child = grid.children[4]
  b, s, d = 0.02, 0.01, child.altitude - 0.02 - 0.01
  parent_coords = child.local_to_parent(b, s, d)
  back = child.parent_to_local(*parent_coords)
  assert back == pytest.approx((b, s, d))


def test_local_to_parent_raises_for_root():
  grid = NestingIsoGrid(altitude=1.0)
  with pytest.raises(ValueError):
    grid.local_to_parent(0.1, 0.1, 0.1)


def test_parent_to_local_raises_for_root():
  grid = NestingIsoGrid(altitude=1.0)
  with pytest.raises(ValueError):
    grid.parent_to_local(0.1, 0.1, 0.1)


def test_to_root_isometric_matches_recursive_composition_for_grandchild():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  child = grid.children[0]
  child.subdivide(resolution=3)
  grandchild = child.child_at(0, 0, False)

  b_l, s_l = 0.03, 0.03
  d_l = grandchild.altitude - b_l - s_l

  # O(1) cached transform.
  cached = grandchild.to_root_isometric(b_l, s_l, d_l)

  # Manual recursive composition, for comparison.
  b_p, s_p, d_p = grandchild.local_to_parent(b_l, s_l, d_l)
  b_r, s_r, d_r = child.local_to_parent(b_p, s_p, d_p)

  assert cached == pytest.approx((b_r, s_r, d_r))


def test_to_root_cartesian_matches_isometric_to_cartesian_of_to_root_isometric():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  child = grid.children[3]
  b, s, d = 0.05, 0.05, child.altitude - 0.1
  cartesian = child.to_root_cartesian(b, s, d)
  expected = isometric_to_cartesian(*child.to_root_isometric(b, s, d))
  assert cartesian == pytest.approx(expected)


def test_root_to_root_isometric_is_identity():
  grid = NestingIsoGrid(altitude=1.0)
  assert grid.to_root_isometric(0.1, 0.2, 0.3) == pytest.approx((0.1, 0.2, 0.3))


def test_project_onto_root_grid_matches_to_root_isometric():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  p = IsometricPoint(child, 0.1, 0.05)
  projected = child.project_onto_root_grid(p)
  assert projected.grid is grid
  expected_b, expected_s, _ = child.to_root_isometric(p.b, p.s, p.d)
  assert projected.b == pytest.approx(expected_b)
  assert projected.s == pytest.approx(expected_s)


def test_project_onto_root_grid_in_place_mutates_and_returns_same_point():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  p = IsometricPoint(child, 0.1, 0.05)
  result = child.project_onto_root_grid(p, in_place=True)
  assert result is p
  assert p.grid is grid


# ---------------------------------------------------------------------------
# 2D rendering
# ---------------------------------------------------------------------------

def test_render_2d_respects_scale_visibility_window():
  grid = NestingIsoGrid(altitude=1.0)
  p = IsometricPoint.center(grid)
  grid.add_item(SectorItem(position=p, payload="a", min_scale=1.0, max_scale=2.0))
  assert [i.payload for i in grid.render_2d(scale=0.5)] == []
  assert [i.payload for i in grid.render_2d(scale=1.5)] == ["a"]
  assert [i.payload for i in grid.render_2d(scale=2.0)] == []  # exclusive


def test_render_2d_recurses_into_children():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  p = IsometricPoint.center(child)
  child.add_item(SectorItem(position=p, payload="child-item"))
  payloads = [i.payload for i in grid.render_2d(scale=1.0)]
  assert payloads == ["child-item"]


def test_render_2d_viewport_culls_out_of_view_subtrees():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  p = IsometricPoint.center(child)
  child.add_item(SectorItem(position=p, payload="item"))
  x, y = child.to_root_cartesian(p.b, p.s, p.d)

  # Viewport that contains the item.
  contains = (x - 1, y - 1, x + 1, y + 1)
  assert [i.payload for i in grid.render_2d(scale=1.0, viewport=contains)] == [
      "item"]

  # Viewport far away from everything.
  excludes = (1000.0, 1000.0, 1001.0, 1001.0)
  assert [i.payload for i in grid.render_2d(scale=1.0, viewport=excludes)] == []


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def test_repr_reflects_orientation_and_state():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[1]  # first inverted child
  assert child.inverted
  assert "v " in repr(child)
  assert repr(grid).startswith("NestingIsoGrid(^")
  assert "res=2" in repr(grid)


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------

def test_depth_increases_with_nesting():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  child.subdivide(resolution=2)
  grandchild = child.children[0]
  assert grid.depth == 0
  assert child.depth == 1
  assert grandchild.depth == 2


# ---------------------------------------------------------------------------
# default_resolution
# ---------------------------------------------------------------------------

def test_default_resolution_defaults_to_2_at_the_root():
  grid = NestingIsoGrid(altitude=1.0)
  grid.subdivide()
  assert grid.resolution == 2


def test_default_resolution_explicit_at_the_root():
  grid = NestingIsoGrid(altitude=1.0, default_resolution=5)
  grid.subdivide()
  assert grid.resolution == 5


def test_default_resolution_inherited_by_children_from_root():
  grid = NestingIsoGrid(resolution=2, altitude=1.0, default_resolution=5)
  child = grid.children[0]
  child.subdivide()
  assert child.resolution == 5


def test_default_resolution_inherited_transitively_by_grandchildren():
  grid = NestingIsoGrid(resolution=2, altitude=1.0, default_resolution=5)
  child = grid.children[0]
  child.subdivide(resolution=2)  # explicit at this level
  grandchild = child.children[0]
  grandchild.subdivide()  # not explicit -> inherits from child, not the root
  assert grandchild.resolution == 5


# ---------------------------------------------------------------------------
# child_containing float overload
# ---------------------------------------------------------------------------

def test_child_containing_float_overload_matches_point_overload():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  p = IsometricPoint(grid, 0.05, 0.05)
  by_point = grid.child_containing(p)
  by_coords = grid.child_containing(p.b, p.s, p.d)
  assert by_coords is by_point


def test_child_containing_float_overload_requires_subdivision():
  grid = NestingIsoGrid(altitude=1.0)
  with pytest.raises(ValueError):
    grid.child_containing(0.1, 0.1, 0.1)


# ---------------------------------------------------------------------------
# canonicalize_point
# ---------------------------------------------------------------------------

def test_canonicalize_point_is_noop_when_already_in_bounds():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  p = IsometricPoint(child, 0.1 * child.altitude, 0.1 * child.altitude)
  result = NestingIsoGrid.canonicalize_point(p)
  assert result is p
  assert p.grid is child
  assert p.b == pytest.approx(0.1 * child.altitude)
  assert p.s == pytest.approx(0.1 * child.altitude)


def test_canonicalize_point_moves_to_a_sibling_within_the_same_tree():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  # (i_b=1, i_s=0) borders an interior sibling edge (not the tree's own outer
  # boundary), so a small negative excursion in b should land on that sibling
  # rather than escaping the whole tree.
  child = grid.child_at(1, 0, False)
  alt = child.altitude
  p = IsometricPoint(child, -0.01 * alt, 0.3 * alt)
  expected_root = child.to_root_isometric(p.b, p.s, p.d)

  NestingIsoGrid.canonicalize_point(p)

  assert p.grid is not child
  assert p.grid.parent is grid
  assert p.grid.to_root_isometric(p.b, p.s, p.d) == pytest.approx(
      expected_root, abs=1e-9)
  assert -1e-9 <= p.b <= p.grid.altitude + 1e-9
  assert -1e-9 <= p.s <= p.grid.altitude + 1e-9
  assert -1e-9 <= p.d <= p.grid.altitude + 1e-9


def test_canonicalize_point_auto_subdivides_unvisited_sibling_using_default_resolution():
  grid = NestingIsoGrid(resolution=2, altitude=1.0, default_resolution=3)
  child_a = grid.children[0]
  child_b = grid.children[1]
  child_a.subdivide(resolution=2)
  grandchild = child_a.children[0]
  assert child_b.children is None

  # The exact isometric coordinates of `child_b`'s centroid, re-expressed in
  # `grandchild`'s local frame. This lands out of `grandchild`'s own bounds,
  # since the point actually lies within `child_b`, a different top-level
  # child that hasn't been subdivided yet.
  root_coords = child_b.local_to_parent(
      child_b.apothem, child_b.apothem, child_b.apothem)
  child_a_coords = child_a.parent_to_local(*root_coords)
  grandchild_coords = grandchild.parent_to_local(*child_a_coords)

  p = IsometricPoint(grandchild, grandchild_coords[0], grandchild_coords[1])
  NestingIsoGrid.canonicalize_point(p)

  assert child_b.children is not None  # auto-subdivided
  assert child_b.resolution == 3  # used default_resolution, not root's
  assert p.grid.depth == 2
  assert p.grid.root is grid
  assert p.grid.to_root_isometric(p.b, p.s, p.d) == pytest.approx(
      root_coords, abs=1e-9)


def test_canonicalize_point_raises_past_the_root_boundary():
  grid = NestingIsoGrid(resolution=2, altitude=1.0)
  child = grid.children[0]
  alt = child.altitude
  p = IsometricPoint(child, -10 * alt, -10 * alt)
  with pytest.raises(EndOfMeshSurfaceException):
    NestingIsoGrid.canonicalize_point(p)
