class NotAdjacentException(Exception):
  """
  Raised when an operation expects adjacent graph nodes but encounters
  non-adjacent nodes.
  """
  pass

class EndOfMeshSurfaceException(Exception):
  """
  Raised when an operation attempts to traverse the mesh surface to a new face
  and instead finds an unexpected end in the mesh surface.
  """
  pass
