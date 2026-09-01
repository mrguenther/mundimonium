from mundimonium.coordinates import (
    SphericalTessellation, Tessellation
)
from mundimonium.coordinates.lod_mesh import LodMeshFace

from typing import Any

class World:
  """
  TODO
  """
  def __init__(self,
               geometry_type: type[Tessellation] = SphericalTessellation,
               geometry_params: dict[str, Any] | None = None):
    if geometry_params is None:
      geometry_params = dict()
    elif "face_type" in geometry_params:
      raise ValueError("'face_type' may not be overridden from 'LodMeshFace'.")

    self._mesh = geometry_type(face_type=LodMeshFace, **geometry_params)

  @property
  def mesh(self):
    return self._mesh


def main():
  world = World(SphericalTessellation, {"frequency": 8})


if __name__ == "__main__":
  main()

