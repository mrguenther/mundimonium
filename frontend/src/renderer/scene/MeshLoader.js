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
  const material = new THREE.MeshStandardMaterial({ color: 0x4c8bf5 });
  const mesh = new THREE.Mesh(geometry, material);

  const directional = new THREE.DirectionalLight(0xffffff, 2.0);
  directional.position.set(3, 5, 4);
  const ambient = new THREE.AmbientLight(0xffffff, 0.4);

  return { mesh, lights: [directional, ambient] };
}
