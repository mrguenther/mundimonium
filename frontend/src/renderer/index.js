import { SceneManager } from './scene/SceneManager.js';
import { OrbitCameraController } from './scene/CameraController.js';
import { buildShadedMesh } from './scene/MeshLoader.js';

const statusElement = document.getElementById('status');

function setStatus(text) {
  statusElement.textContent = text;
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

  setStatus('Requesting mesh...');
  try {
    const meshData = await window.mundimonium.getMesh({
      tessellation: 'spherical', radius: 1, frequency: 3,
    });
    const { mesh, lights } = buildShadedMesh(meshData);
    sceneManager.addToScene(mesh);
    for (const light of lights) {
      sceneManager.addToScene(light);
    }
    setStatus('');
  } catch (error) {
    setStatus(`Failed to load mesh: ${error.message}`);
  }
}

main();
