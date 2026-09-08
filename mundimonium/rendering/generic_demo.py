"""Demo `GenericTessellation` content for the rendering pipeline.

`GenericTessellation` has no procedural world-generation content of its own
(see README.md) -- this seeds a fixed, hardcoded stellated icosahedron
purely to prove the pipeline generalizes beyond `SphericalTessellation`,
mirroring `server.py`'s own `_DEMO_ITEMS` pattern for the spherical case.
"""

from __future__ import annotations

from mundimonium.coordinates.generic_tessellation import (
    GenericTessellation, RelaxableVertex,
)
from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.nesting_iso_grid import SectorItem
from mundimonium.coordinates.relaxable_lod_mesh import RelaxableLodMeshFace
from mundimonium.coordinates.stellated_icosahedron import (
    build_stellated_icosahedron,
)

# Hardcoded (face index, label, min_scale, max_scale) demo markers, spread
# across the 60-face stellated icosahedron -- see `server.py`'s own
# `_DEMO_ITEMS` for the spherical-demo equivalent this mirrors.
_DEMO_ITEMS = [
    (0, 'Anchorhold', 0.0, float('inf')),
    (15, 'Millbrook', 0.0, float('inf')),
    (30, 'Stonegate', 0.0, float('inf')),
    (45, 'Hiddenreach', 2.0, float('inf')),
]


def build_demo_tessellation() -> GenericTessellation:
  """A fresh, `_DEMO_ITEMS`-seeded stellated-icosahedron `GenericTessellation`.

  Built with `face_type=RelaxableLodMeshFace, vertex_type=RelaxableVertex`
  so it can participate in the same LOD-tree-aware rendering pipeline
  (`lod_mesh_export.py`, `flat_mesh_export.py`) and `flatten_region` support
  `SphericalTessellation` already has, even though this fixed-size demo mesh
  doesn't itself need on-demand subdivision.
  """
  tessellation, _vertices, faces = build_stellated_icosahedron(
      face_type=RelaxableLodMeshFace, vertex_type=RelaxableVertex)
  for face_index, label, min_scale, max_scale in _DEMO_ITEMS:
    face = faces[face_index]
    face.add_item(SectorItem(
        position=IsometricPoint.center(face),
        payload={'kind': 'city', 'label': label},
        min_scale=min_scale, max_scale=max_scale))
  return tessellation
