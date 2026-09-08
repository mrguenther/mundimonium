import * as THREE from 'three';

const MARKER_TEXTURE_SIZE = 64;
const MARKER_WORLD_SIZE = 0.06;

let cachedMarkerMaterial = null;

/**
 * A small circular marker texture, drawn once from an offscreen canvas
 * rather than loading an external image asset, and reused by every
 * sprite -- there's no per-kind visual styling yet.
 *
 * @returns {THREE.SpriteMaterial}
 */
function markerMaterial() {
  if (cachedMarkerMaterial) {
    return cachedMarkerMaterial;
  }
  const canvas = document.createElement('canvas');
  canvas.width = MARKER_TEXTURE_SIZE;
  canvas.height = MARKER_TEXTURE_SIZE;
  const context = canvas.getContext('2d');
  const radius = MARKER_TEXTURE_SIZE / 2;
  context.beginPath();
  context.arc(radius, radius, radius - 2, 0, Math.PI * 2);
  context.fillStyle = '#ffcc44';
  context.fill();
  context.lineWidth = 2;
  context.strokeStyle = '#332200';
  context.stroke();

  cachedMarkerMaterial = new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas),
  });
  return cachedMarkerMaterial;
}

/**
 * Builds one billboarded marker per item (a `THREE.Sprite` always faces
 * the camera). Each sprite's `userData` carries the item's own `kind`/
 * `label`, for future use -- not rendered as visible text this phase.
 *
 * @param {{ kind: string, label?: string, x: number, y: number,
 *   z: number }[]} items
 * @returns {THREE.Sprite[]}
 */
export function buildItemSprites(items) {
  const material = markerMaterial();
  return items.map((item) => {
    const sprite = new THREE.Sprite(material);
    sprite.position.set(item.x, item.y, item.z);
    sprite.scale.setScalar(MARKER_WORLD_SIZE);
    sprite.userData = { kind: item.kind, label: item.label };
    return sprite;
  });
}
