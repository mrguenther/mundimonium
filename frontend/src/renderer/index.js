import * as THREE from 'three';

import { SceneManager } from './scene/SceneManager.js';
import { OrbitCameraController } from './scene/CameraController.js';
import { buildGeometry, buildShadedMesh } from './scene/MeshLoader.js';
import { buildItemSprites } from './scene/ItemLoader.js';

const statusElement = document.getElementById('status');

const TESSELLATION = { tessellation: 'spherical', radius: 1, frequency: 3 };

// Debug-only stand-in for a real click-to-subdivide UI (not built this
// phase -- growing LOD detail defaults to off, since it's expected to
// eventually trigger procedural generation). Press 'g' to subdivide the
// first sector from the most recently rendered frontier.
const DEBUG_SUBDIVIDE_KEY = 'g';

function setStatus(text) {
  statusElement.textContent = text;
}

/**
 * @param {THREE.Camera} camera
 * @returns {[number, number, number]}
 */
function cameraPositionArray(camera) {
  return [camera.position.x, camera.position.y, camera.position.z];
}

async function main() {
  const sceneManager = new SceneManager(document.body);
  const cameraController = new OrbitCameraController(
      sceneManager.renderer.domElement);
  sceneManager.setActiveController(cameraController);
  sceneManager.start();

  window.mundimonium.onPythonStatus((status) => {
    if (status.state === 'crashed' || status.state === 'exited') {
      setStatus(`Python ${status.state}${status.detail ? ': ' + status.detail : ''}`);
    }
  });

  let mesh;
  let lastSectors = [];
  const itemGroup = new THREE.Group();

  async function refreshLodMesh() {
    const meshData = await window.mundimonium.getLodMesh({
      ...TESSELLATION,
      cameraPosition: cameraPositionArray(cameraController.camera),
    });
    lastSectors = meshData.sectors;
    if (mesh) {
      mesh.geometry.dispose();
      mesh.geometry = buildGeometry(meshData);
    }
    return meshData;
  }

  async function refreshItems() {
    const { items } = await window.mundimonium.getItems({
      ...TESSELLATION,
      cameraPosition: cameraPositionArray(cameraController.camera),
    });
    itemGroup.clear();
    for (const sprite of buildItemSprites(items)) {
      itemGroup.add(sprite);
    }
  }

  setStatus('Requesting mesh...');
  try {
    const meshData = await refreshLodMesh();
    const built = buildShadedMesh(meshData);
    mesh = built.mesh;
    sceneManager.addToScene(mesh);
    for (const light of built.lights) {
      sceneManager.addToScene(light);
    }
    sceneManager.addToScene(itemGroup);
    await refreshItems();
    setStatus('');
  } catch (error) {
    setStatus(`Failed to load: ${error.message}`);
    return;
  }

  cameraController.onChange(() => {
    refreshLodMesh().catch((error) => {
      setStatus(`Failed to update mesh: ${error.message}`);
    });
    refreshItems().catch((error) => {
      setStatus(`Failed to update items: ${error.message}`);
    });
  });

  window.addEventListener('keydown', (event) => {
    if (event.key !== DEBUG_SUBDIVIDE_KEY || lastSectors.length === 0) {
      return;
    }
    window.mundimonium
        .subdivideSector({ ...TESSELLATION, sector: lastSectors[0] })
        .then(refreshLodMesh)
        .catch((error) => setStatus(`Failed to subdivide: ${error.message}`));
  });
}

main();
