"""`LodMeshFace` + `RelaxableFace`, combined.

`lod_mesh.py` and `generic_tessellation.py` are siblings -- neither imports
the other, and neither should have to just to let a downstream consumer use
both mixins together. This module is the join point: it's the only thing
that needs to know about both, so it's the only thing that imports both.
"""

from __future__ import annotations

from mundimonium.coordinates.generic_tessellation import RelaxableFace
from mundimonium.coordinates.lod_mesh import LodMeshFace


class RelaxableLodMeshFace(LodMeshFace, RelaxableFace):
  """A `LodMeshFace` that's also a `RelaxableFace`.

  Used via `GenericTessellation(..., face_type=RelaxableLodMeshFace,
  vertex_type=RelaxableVertex)` for a `GenericTessellation` that needs both
  LOD-tree awareness and `flatten_region` support. `RelaxableVertex` (from
  `generic_tessellation.py`) needs no LOD-aware counterpart of its own --
  `LodMeshFace`'s LOD-tree state lives entirely on faces/sectors, never on
  vertices.
  """
