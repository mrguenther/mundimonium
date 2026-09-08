from __future__ import annotations

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshSector
from mundimonium.coordinates.tessellation import Tessellation

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# How many children a sector is split into, whether by `select_frontier`'s
# own auto-subdivide path or by an explicit `subdivide_sector` request.
SUBDIVISION_RESOLUTION = 2

DEFAULT_LOD_THRESHOLD = 4.0
DEFAULT_MAX_DEPTH = 6


@dataclass(frozen=True)
class SectorAddress:
  """Identifies one `LodMeshSector` within a `Tessellation`.

  `face_index` selects a top-level `LodMeshFace` (`tessellation.faces[
  face_index]`); `path` is the chain of `(i_b, i_s, inverted)` steps from
  there down to a nested `LodMeshSector`, each resolved via `child_at`.
  """
  face_index: int
  path: tuple[tuple[int, int, bool], ...] = ()

  def child(self, i_b: int, i_s: int, inverted: bool) -> SectorAddress:
    """The address of the named child of the sector this address names."""
    return SectorAddress(self.face_index, self.path + ((i_b, i_s, inverted),))

  def to_json(self) -> dict:
    """A JSON-safe representation, matching the wire protocol's shape."""
    return {
        'face': self.face_index,
        'path': [list(step) for step in self.path],
    }

  @classmethod
  def from_json(cls, data: dict) -> SectorAddress:
    """Inverse of `to_json`."""
    return cls(data['face'], tuple(tuple(step) for step in data['path']))


def resolve_sector_address(
    tessellation: Tessellation, address: SectorAddress,
) -> LodMeshSector:
  """Resolves `address` to the `LodMeshSector` it names.

  Raises:
    IndexError: If `address.face_index` is out of range.
    ValueError: If a step in `address.path` names a sector that hasn't
      been subdivided yet.
  """
  sector = tessellation.faces[address.face_index]
  for i_b, i_s, inverted in address.path:
    sector = sector.child_at(i_b, i_s, inverted)
  return sector


def select_frontier(
    tessellation: Tessellation,
    camera_position: Sequence[float],
    threshold: float = DEFAULT_LOD_THRESHOLD,
    max_depth: int = DEFAULT_MAX_DEPTH,
    auto_subdivide: bool = False,
    face_indices: Sequence[int] | None = None,
) -> list[tuple[LodMeshSector, SectorAddress]]:
  """Selects which sectors to render for a camera at `camera_position`.

  Recurses into a sector's children only while it's both close enough
  (relative to `threshold`) and shallow enough (`max_depth`) to want finer
  detail.

  A sector that wants finer detail but has no children yet is only
  subdivided if `auto_subdivide` is `True` -- otherwise it's rendered as
  one coarse triangle, since building new detail isn't assumed cheap (it's
  expected to eventually trigger procedural generation).

  Sectors are never merged back together, so lowering the camera's needs
  after a previous `auto_subdivide=True` call simply stops descending
  into already-built children rather than undoing anything.

  Every top-level face is included somewhere in the result regardless of
  `threshold`/`max_depth` -- those two only control subdivision *depth*,
  never whether a face is considered at all. A caller that only wants a
  *local neighborhood* rendered (e.g. flat mode, where a face on the far
  side of the mesh would otherwise still show up, badly distorted) needs
  to restrict `face_indices` itself; this function has no opinion on
  proximity beyond a single face's own children.

  Args:
    tessellation: A `Tessellation` built with a LOD-tree-aware `face_type`
      (`LodMeshFace`, or `RelaxableLodMeshFace` for a `GenericTessellation`).
    camera_position: The camera's `(x, y, z)` world position.
    threshold: How large (relative to a sector's own altitude) the
      distance to the camera must be before that sector is coarse enough
      to render as-is.
    max_depth: Never recurse past this many subdivisions below a
      top-level face, regardless of distance.
    auto_subdivide: Whether sectors lacking children may be subdivided to
      satisfy `threshold`. Defaults to `False`.
    face_indices: If given, only these top-level face indices are
      considered at all (each still recursed into normally) -- every
      index into `tessellation.faces` otherwise, unchanged from this
      function's prior behavior.

  Returns:
    A list of `(sector, address)` pairs -- the sectors to render, each
    paired with the address needed to later request its subdivision.
  """
  camera_position = np.array(camera_position, dtype=np.float64)
  frontier: list[tuple[LodMeshSector, SectorAddress]] = []
  indices = (
      range(len(tessellation.faces)) if face_indices is None
      else face_indices)
  for face_index in indices:
    face = tessellation.faces[face_index]
    _select_frontier(
        tessellation, face, SectorAddress(face_index), camera_position,
        threshold, max_depth, auto_subdivide, depth=0, frontier=frontier)
  return frontier


def _select_frontier(
    tessellation: Tessellation,
    sector: LodMeshSector,
    address: SectorAddress,
    camera_position: np.ndarray,
    threshold: float,
    max_depth: int,
    auto_subdivide: bool,
    depth: int,
    frontier: list[tuple[LodMeshSector, SectorAddress]],
) -> None:
  """`select_frontier`'s recursion, appending chosen sectors to `frontier`."""
  centroid = sector_world_position(
      tessellation, sector, IsometricPoint.center(sector))
  distance = float(np.linalg.norm(camera_position - centroid))
  wants_finer = distance < threshold * sector.altitude and depth < max_depth

  if wants_finer and sector.children is None and auto_subdivide:
    sector.subdivide(resolution=SUBDIVISION_RESOLUTION)

  if not wants_finer or sector.children is None:
    frontier.append((sector, address))
    return

  for child in sector.children:
    _select_frontier(
        tessellation, child,
        address.child(child.i_b, child.i_s, child.inverted),
        camera_position, threshold, max_depth, auto_subdivide, depth + 1,
        frontier)


def lod_frontier_to_buffers(
    tessellation: Tessellation,
    frontier: list[tuple[LodMeshSector, SectorAddress]],
) -> tuple[bytes, bytes, int, int]:
  """Exports a `select_frontier` result as renderer-ready buffers.

  Each frontier sector contributes its own 3 corners as 3 fresh vertices
  and 1 triangle -- unlike `mesh_export.tessellation_to_buffers`, corners
  aren't deduplicated across adjacent sectors (they aren't backed by
  shared `TessellationVertex` identity), so triangles at sector boundaries
  don't share vertices.

  Positions are computed via `Tessellation.point_to_3d_position`, so
  (unlike `mesh_export.py`) this only works for a tessellation whose
  `projection_coordinates` are already Euclidean 3D.

  Args:
    tessellation: The `frontier`'s owning tessellation.
    frontier: A `select_frontier` result.

  Returns:
    A `(positions_bytes, indices_bytes, vertex_count, face_count)` tuple,
    with the same buffer layout as `mesh_export.tessellation_to_buffers`
    (`vertex_count * 3` little-endian `float32` values, `face_count * 3`
    little-endian `uint32` indices) -- one triangle per frontier sector,
    in `frontier`'s order.
  """
  positions: list[np.ndarray] = []
  indices: list[tuple[int, int, int]] = []
  for sector, _address in frontier:
    altitude = sector.altitude
    corners = (
        IsometricPoint(sector, altitude, 0),  # vertex B
        IsometricPoint(sector, 0, altitude),  # vertex S
        IsometricPoint(sector, 0, 0),         # vertex D
    )
    base_index = len(positions)
    positions.extend(
        sector_world_position(tessellation, sector, corner)
        for corner in corners)
    indices.append((base_index, base_index + 1, base_index + 2))

  positions_array = np.array(positions, dtype=np.float32)
  indices_array = np.array(indices, dtype=np.uint32)
  return (
      positions_array.tobytes(), indices_array.tobytes(),
      len(positions), len(frontier))


def sector_world_position(
    tessellation: Tessellation,
    sector: LodMeshSector,
    local_point: IsometricPoint,
) -> np.ndarray:
  """The 3D world position of a point in `sector`'s own local frame."""
  return tessellation.point_to_3d_position(
      sector.project_onto_root_grid(local_point))
