import numpy as np

from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering.mesh_export import tessellation_to_buffers


def test_buffer_lengths_and_counts_agree():
  tess = SphericalTessellation(radius=2.0, frequency=1)
  positions_bytes, indices_bytes, vertex_count, face_count = (
      tessellation_to_buffers(tess))

  assert vertex_count == tess.num_vertices == 12
  assert face_count == tess.num_faces == 20
  assert len(positions_bytes) == vertex_count * 3 * 4  # float32
  assert len(indices_bytes) == face_count * 3 * 4  # uint32


def test_positions_match_projection_coordinates_in_order():
  tess = SphericalTessellation(radius=2.0, frequency=1)
  positions_bytes, _, _, _ = tessellation_to_buffers(tess)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  for i, vertex in enumerate(tess.vertices):
    np.testing.assert_allclose(
        positions[i], vertex.projection_coordinates, rtol=1e-6)


def test_indices_reference_the_correct_face_corners():
  tess = SphericalTessellation(radius=2.0, frequency=1)
  _, indices_bytes, _, _ = tessellation_to_buffers(tess)
  indices = np.frombuffer(indices_bytes, dtype=np.uint32).reshape(-1, 3)

  vertex_list = list(tess.vertices)
  for i, face in enumerate(tess.faces):
    b, s, d = (int(x) for x in indices[i])
    assert vertex_list[b] is face.vertex_b
    assert vertex_list[s] is face.vertex_s
    assert vertex_list[d] is face.vertex_d


def test_all_positions_lie_on_the_sphere():
  tess = SphericalTessellation(radius=3.0, frequency=2)
  positions_bytes, _, _, _ = tessellation_to_buffers(tess)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  radii = np.linalg.norm(positions, axis=1)
  np.testing.assert_allclose(radii, 3.0, rtol=1e-4)
