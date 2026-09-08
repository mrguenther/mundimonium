import numpy as np

from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.coordinates.tessellation import TessellationVertex
from mundimonium.rendering.mesh_export import tessellation_to_buffers


def test_buffer_lengths_and_counts_agree():
  tess = SphericalTessellation(radius=2.0, frequency=1)
  positions_bytes, indices_bytes, vertex_count, face_count, _, _ = (
      tessellation_to_buffers(tess))

  assert vertex_count == tess.num_vertices == 12
  assert face_count == tess.num_faces == 20
  assert len(positions_bytes) == vertex_count * 3 * 4  # float32
  assert len(indices_bytes) == face_count * 3 * 4  # uint32


def test_positions_match_projection_coordinates_in_order():
  tess = SphericalTessellation(radius=2.0, frequency=1)
  positions_bytes, _, _, _, _, _ = tessellation_to_buffers(tess)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  for i, vertex in enumerate(tess.vertices):
    np.testing.assert_allclose(
        positions[i], vertex.projection_coordinates, rtol=1e-6)


def test_indices_reference_the_correct_face_corners():
  tess = SphericalTessellation(radius=2.0, frequency=1)
  _, indices_bytes, _, _, _, _ = tessellation_to_buffers(tess)
  indices = np.frombuffer(indices_bytes, dtype=np.uint32).reshape(-1, 3)

  vertex_list = list(tess.vertices)
  for i, face in enumerate(tess.faces):
    b, s, d = (int(x) for x in indices[i])
    assert vertex_list[b] is face.vertex_b
    assert vertex_list[s] is face.vertex_s
    assert vertex_list[d] is face.vertex_d


def test_all_positions_lie_on_the_sphere():
  tess = SphericalTessellation(radius=3.0, frequency=2)
  positions_bytes, _, _, _, _, _ = tessellation_to_buffers(tess)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  radii = np.linalg.norm(positions, axis=1)
  np.testing.assert_allclose(radii, 3.0, rtol=1e-4)


def test_adjacency_is_symmetric_and_shares_two_vertices_per_edge():
  tess = SphericalTessellation(radius=1.0, frequency=1)
  _, indices_bytes, _, face_count, adjacency, _ = tessellation_to_buffers(
      tess)
  indices = np.frombuffer(indices_bytes, dtype=np.uint32).reshape(-1, 3)

  for face_index in range(face_count):
    for slot, neighbor_index in enumerate(adjacency[face_index]):
      assert neighbor_index != -1  # the sphere mesh has no boundary
      own_vertices = set(int(v) for v in indices[face_index])
      own_vertices.discard(int(indices[face_index][slot]))
      neighbor_vertices = set(int(v) for v in indices[neighbor_index])
      assert own_vertices <= neighbor_vertices  # the two shared vertices
      assert face_index in adjacency[neighbor_index]  # symmetric


def test_vertex_faces_lists_every_face_touching_that_vertex():
  tess = SphericalTessellation(radius=1.0, frequency=1)
  _, indices_bytes, vertex_count, _, _, vertex_faces = tessellation_to_buffers(
      tess)
  indices = np.frombuffer(indices_bytes, dtype=np.uint32).reshape(-1, 3)

  for vertex_index in range(vertex_count):
    expected = {
        face_index for face_index, corners in enumerate(indices)
        if vertex_index in corners
    }
    assert set(vertex_faces[vertex_index]) == expected


def test_adjacency_is_minus_one_for_a_boundary_edge():
  # Two faces sharing exactly one edge (v2-v3), otherwise open -- an
  # unclosed, boundary-having mesh, unlike the closed sphere above.
  v1 = TessellationVertex([0.0, 0.0, 0.0])
  v2 = TessellationVertex([1.0, 0.0, 0.0])
  v3 = TessellationVertex([0.5, 1.0, 0.0])
  v4 = TessellationVertex([1.5, 1.0, 0.0])
  tess = GenericTessellation()
  tess.add_face([v1, v2, v3])
  tess.add_face([v2, v4, v3])

  _, _, _, _, adjacency, _ = tessellation_to_buffers(tess)

  assert adjacency[0] == [1, -1, -1]  # shares only the v2-v3 (b-opposite) edge
  assert adjacency[1] == [-1, 0, -1]  # shares only the v2-v3 (s-opposite) edge


def test_vertex_faces_are_in_cyclic_order():
  tess = SphericalTessellation(radius=1.0, frequency=1)
  _, indices_bytes, _, _, adjacency, vertex_faces = tessellation_to_buffers(
      tess)
  indices = np.frombuffer(indices_bytes, dtype=np.uint32).reshape(-1, 3)

  for fan in vertex_faces:
    for i in range(len(fan)):
      current_face, next_face = fan[i], fan[(i + 1) % len(fan)]
      # Consecutive faces in the fan must actually be edge-adjacent to
      # each other -- not merely all touching the same vertex.
      assert next_face in adjacency[current_face]
