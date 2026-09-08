import * as THREE from 'three';

/**
 * Builds a `THREE.BufferGeometry` from the `{ positions, indices }` typed
 * arrays the Python bridge returns. Normals are computed client-side --
 * Python sends positions/indices only, rather than adding a normals array
 * to the wire protocol before it's needed.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} meshData
 * @returns {THREE.BufferGeometry}
 */
export function buildGeometry({ positions, indices }) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * `buildGeometry`'s output, wrapped in a lit material plus matching scene
 * lights.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} meshData
 * @returns {{ mesh: THREE.Mesh, lights: THREE.Light[] }}
 */
export function buildShadedMesh(meshData) {
  const geometry = buildGeometry(meshData);
  // `DoubleSide`: harmless for a properly-wound closed solid (back faces
  // stay hidden behind front ones from any exterior viewpoint regardless),
  // but load-bearing for GenericTessellation.flatten_region's flat-mode
  // output specifically -- its per-face blended affine transforms can
  // reflect rather than just rotate/scale (a documented, accepted
  // trade-off, not a bug), flipping some triangles' winding. A single-
  // sided material would cull exactly those triangles and shade them
  // inconsistently under a fixed light; DoubleSide renders them correctly
  // either way by flipping the effective normal for back-facing pixels.
  const material = new THREE.MeshStandardMaterial({
    color: 0x4c8bf5, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);

  const directional = new THREE.DirectionalLight(0xffffff, 2.0);
  directional.position.set(3, 5, 4);
  const ambient = new THREE.AmbientLight(0xffffff, 0.4);

  return { mesh, lights: [directional, ambient] };
}

/**
 * A `THREE.LineSegments` overlay drawing every triangle edge in `geometry`
 * -- `THREE.WireframeGeometry`, not `THREE.EdgesGeometry`: the latter only
 * draws edges between faces whose normals differ by more than a threshold
 * angle, which would render as nearly empty on flat mode's mesh (every
 * triangle there is coplanar, sharing the same computed `+Z` normal, so
 * adjacent triangles' shared edges wouldn't qualify). Primarily a debug
 * aid: flat mode has no shading variation at all to reveal whether
 * `flatten_region` is actually doing anything beyond a static orthographic
 * snapshot, but the mesh topology traced out by these edges will.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {number} [color]
 * @returns {THREE.LineSegments}
 */
export function buildWireframe(geometry, color = 0x000000) {
  const wireframeGeometry = new THREE.WireframeGeometry(geometry);
  const material = new THREE.LineBasicMaterial({ color });
  return new THREE.LineSegments(wireframeGeometry, material);
}
