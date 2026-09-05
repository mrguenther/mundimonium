import * as THREE from 'three';

// DEBUG ONLY, not part of the shipped feature set -- renders each visible
// triangle's top-level face index as a floating text label at that face's
// own centroid, so otherwise visually-identical faces can be told apart
// while diagnosing SphericalTessellation's occasional flat-mode panning
// jumps. To remove: delete this file, its one import in index.js, the
// `debugLabelGroup` variable and its `sceneManager.addToScene` call, and
// the `buildFaceIndexLabels`/`debugLabelGroup` lines inside
// `updateMeshGeometry` (the `sectors` parameter can then also go, along
// with the argument each call site passes for it).

const LABEL_TEXTURE_SIZE = 128;
const LABEL_WORLD_SIZE = 0.12;

// One label texture per face index, built once and reused -- there are
// at most a few hundred faces, so this stays small.
const materialCache = new Map();

/** @returns {THREE.SpriteMaterial} */
function labelMaterial(faceIndex) {
  if (materialCache.has(faceIndex)) {
    return materialCache.get(faceIndex);
  }
  const canvas = document.createElement('canvas');
  canvas.width = LABEL_TEXTURE_SIZE;
  canvas.height = LABEL_TEXTURE_SIZE;
  const context = canvas.getContext('2d');
  context.font = 'bold 64px sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  const center = LABEL_TEXTURE_SIZE / 2;
  const text = String(faceIndex);
  context.strokeStyle = '#000000';
  context.lineWidth = 6;
  context.strokeText(text, center, center);
  context.fillStyle = '#ffffff';
  context.fillText(text, center, center);

  const material = new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas),
    depthTest: false, // stay legible even when behind the mesh
  });
  materialCache.set(faceIndex, material);
  return material;
}

/**
 * @param {THREE.BufferGeometry} geometry - Geometry whose triangles
 *   `sectors` addresses one-to-one (a `get_lod_mesh`/`get_flat_mesh`
 *   response's own geometry, before any further mutation).
 * @param {{ face: number, path: number[][] }[] | undefined} sectors
 * @returns {THREE.Sprite[]} One label per distinct top-level face
 *   appearing in `sectors`, positioned at the centroid of every triangle
 *   vertex belonging to that face. Empty if `sectors` is absent (e.g. a
 *   plain `get_mesh` response, which carries no per-triangle addresses).
 */
export function buildFaceIndexLabels(geometry, sectors) {
  if (!sectors || sectors.length === 0) {
    return [];
  }
  const positions = geometry.attributes.position.array;
  const indices = geometry.index.array;

  const sumsByFace = new Map(); // faceIndex -> [sumX, sumY, sumZ, count]
  for (let triangle = 0; triangle < sectors.length; triangle++) {
    const faceIndex = sectors[triangle].face;
    const sums = sumsByFace.get(faceIndex) || [0, 0, 0, 0];
    for (let corner = 0; corner < 3; corner++) {
      const vertexIndex = indices[triangle * 3 + corner];
      sums[0] += positions[vertexIndex * 3];
      sums[1] += positions[vertexIndex * 3 + 1];
      sums[2] += positions[vertexIndex * 3 + 2];
      sums[3] += 1;
    }
    sumsByFace.set(faceIndex, sums);
  }

  const labels = [];
  for (const [faceIndex, [sumX, sumY, sumZ, count]] of sumsByFace) {
    const sprite = new THREE.Sprite(labelMaterial(faceIndex));
    sprite.position.set(sumX / count, sumY / count, sumZ / count);
    sprite.scale.setScalar(LABEL_WORLD_SIZE);
    labels.push(sprite);
  }
  return labels;
}
