from mundimonium.coordinates import (
    SphericalTessellation, Tessellation
)
from mundimonium.coordinates.lod_mesh import LodMeshFace

class World:
  """
  TODO
  """
  def __init__(self,
               geometry_type: type[Tessellation] = SphericalTessellation,
               geometry_params: dict = {}):
    if "face_type" in geometry_params:
      raise ValueError("'face_type' may not be overridden from 'LodMeshFace'.")
    self._tessellation = geometry_type(face_type=LodMeshFace, **geometry_params)


def main():
  world = World(SphericalTessellation, {"frequency": 2})


if __name__ == "__main__":
  main()

