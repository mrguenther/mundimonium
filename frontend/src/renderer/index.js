import * as THREE from 'three';

import { SceneManager } from './scene/SceneManager.js';
import {
  FlatMapCameraController, OrbitCameraController,
} from './scene/CameraController.js';
import { buildGeometry, buildShadedMesh, buildWireframe } from './scene/MeshLoader.js';
import { buildItemSprites } from './scene/ItemLoader.js';
import { buildFaceIndexLabels } from './scene/DebugFaceLabels.js'; // DEBUG ONLY -- see its own docstring

const statusElement = document.getElementById('status');

const TESSELLATION = { tessellation: 'spherical', radius: 1, frequency: 3 };

// Mirrors `SphericalTessellation.__init__`'s own `ideal_face_side_length`
// formula (mundimonium/coordinates/spherical_tessellation.py) -- computed
// client-side to avoid a round-trip just for a constant that only depends
// on `TESSELLATION`'s fixed radius/frequency.
const SIDE_LENGTH = (() => {
  const idealSurfaceArea = 4.0 * Math.PI * TESSELLATION.radius ** 2;
  const faceCount = 20 * TESSELLATION.frequency ** 2;
  const idealFaceArea = idealSurfaceArea / faceCount;
  return Math.sqrt(idealFaceArea * 4 / Math.sqrt(3));
})();

// Camera altitude above the surface, in units of SIDE_LENGTH, at which the
// view hard-cuts between the 3D orbit mode and the flat 2D map mode. Same
// threshold both directions (no hysteresis) -- a placeholder value,
// expected to need empirical tuning once there's a real scene to test it
// against.
const FLAT_MODE_THRESHOLD = 5.0;

// Debug-only stand-in for a real click-to-subdivide UI (not built this
// phase -- growing LOD detail defaults to off, since it's expected to
// eventually trigger procedural generation). Press 'g' to subdivide the
// first sector from the most recently rendered frontier.
const DEBUG_SUBDIVIDE_KEY = 'g';

// Debug-only toggle between the spherical world and the fixed
// `GenericTessellation` demo shape (a stellated icosahedron -- see
// `mundimonium/rendering/generic_demo.py`). `'generic'` only supports the
// plain static `get_mesh`/`get_items` endpoints so far (no LOD streaming,
// no flat-map mode -- see the plan's Phase 9), so this bypasses the
// camera-driven LOD refresh loop entirely while showing it, rather than
// making that loop generic-aware for a feature it doesn't support yet.
const DEBUG_TOGGLE_GENERIC_KEY = 't';
const GENERIC_TESSELLATION = { tessellation: 'generic' };

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

/**
 * @param {[number, number, number]} cameraPosition
 * @returns {number}
 */
function altitudeAboveSurface(cameraPosition) {
  const [x, y, z] = cameraPosition;
  return Math.hypot(x, y, z) - TESSELLATION.radius;
}

async function main() {
  const sceneManager = new SceneManager(document.body);
  const orbitController = new OrbitCameraController(
      sceneManager.renderer.domElement);
  sceneManager.setActiveController(orbitController);
  sceneManager.start();

  window.mundimonium.onPythonStatus((status) => {
    if (status.state === 'crashed' || status.state === 'exited') {
      setStatus(`Python ${status.state}${status.detail ? ': ' + status.detail : ''}`);
    }
  });

  let mesh;
  let wireframe; // debug overlay -- see `MeshLoader.buildWireframe`'s docstring
  let lastSectors = [];
  const itemGroup = new THREE.Group();
  const debugLabelGroup = new THREE.Group(); // DEBUG ONLY -- see DebugFaceLabels.js

  /**
   * Swaps `mesh`'s geometry (disposing the old one) and keeps `wireframe`
   * in sync with it -- the one place geometry actually changes, so every
   * refresh path (LOD, flat, generic-demo-toggle) shares this instead of
   * repeating the same three lines and risking the wireframe drifting out
   * of sync with what it's supposed to be outlining.
   *
   * @param {THREE.BufferGeometry} newGeometry
   * @param {{ face: number, path: number[][] }[] | undefined} sectors -
   *   DEBUG ONLY -- forwarded to `buildFaceIndexLabels`, see its docstring.
   */
  function updateMeshGeometry(newGeometry, sectors) {
    mesh.geometry.dispose();
    mesh.geometry = newGeometry;
    wireframe.geometry.dispose();
    wireframe.geometry = new THREE.WireframeGeometry(newGeometry);

    debugLabelGroup.clear(); // DEBUG ONLY
    for (const label of buildFaceIndexLabels(newGeometry, sectors)) {
      debugLabelGroup.add(label);
    }
  }

  // True exactly while `DEBUG_TOGGLE_GENERIC_KEY`'s demo mesh is showing --
  // suppresses the camera-driven LOD/flat-mode refresh loop below, which
  // only knows how to talk to the spherical world's endpoints.
  let viewingGenericDemo = false;

  // Non-null exactly while the flat map is the active mode. Its camera is
  // rebuilt fresh each time flat mode is entered. `flatModeCameraPosition`
  // is the 3D camera position flat mode was entered with -- used only for
  // that first request, since a real 3D position stops being meaningful
  // once navigation continues in 2D (see `flatCenter`).
  let flatController = null;
  let flatModeCameraPosition = null;

  // The server's own resolved center/tangent-basis for the flattened
  // view currently showing -- both `null` until the first `getFlatMesh`/
  // `getFlatItems` response arrives after entering flat mode. Once set,
  // every further refresh re-centers on `flatCenter` (offset by however
  // far `flatController`'s camera has panned since), rather than reusing
  // the increasingly stale `flatModeCameraPosition`, so the projection
  // itself updates in real time as you pan instead of just sliding a
  // frozen snapshot around. `flatBasis` is carried forward the same way
  // so the server can parallel-transport it (`SphericalTessellation
  // .unflatten_point_and_transport_basis`) rather than independently
  // recomputing an orientation at each new center, which would otherwise
  // slowly rotate the view relative to the path actually panned.
  let flatCenter = null;
  let flatBasis = null;

  // `switchToFlatMode` awaits a round-trip before `orbitController` is
  // disabled, so a second drag/zoom in that gap could otherwise trigger a
  // re-entrant call (constructing a second `FlatMapCameraController` no
  // one tracks or disposes).
  let switchingModes = false;

  async function refreshLodMesh() {
    const meshData = await window.mundimonium.getLodMesh({
      ...TESSELLATION,
      cameraPosition: cameraPositionArray(orbitController.camera),
    });
    lastSectors = meshData.sectors;
    updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
    return meshData;
  }

  async function refreshItems() {
    const { items } = await window.mundimonium.getItems({
      ...TESSELLATION,
      cameraPosition: cameraPositionArray(orbitController.camera),
    });
    itemGroup.clear();
    for (const sprite of buildItemSprites(items)) {
      itemGroup.add(sprite);
    }
  }

  /**
   * The request fields that tell the server to re-center exactly on
   * wherever the flat view currently is, once a center/basis are already
   * known (every call after flat mode's first) -- `{}` before that, so
   * the request falls back to deriving both from `flatModeCameraPosition`
   * instead (see `server.py`'s own `_resolve_flat_center_and_basis`).
   */
  function flatRecenterRequestFields() {
    if (!flatCenter) {
      return {};
    }
    return {
      center: flatCenter,
      basis: flatBasis,
      panOffset: [
        flatController.camera.position.x, flatController.camera.position.y,
      ],
    };
  }

  async function refreshFlatMesh() {
    const meshData = await window.mundimonium.getFlatMesh({
      ...TESSELLATION, cameraPosition: flatModeCameraPosition,
      ...flatRecenterRequestFields(),
    });
    lastSectors = meshData.sectors;
    flatCenter = meshData.center;
    flatBasis = meshData.basis;
    updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
    flatController.recenter();
    return meshData;
  }

  async function refreshFlatItems() {
    const { items, center, basis } = await window.mundimonium.getFlatItems({
      ...TESSELLATION, cameraPosition: flatModeCameraPosition,
      ...flatRecenterRequestFields(),
    });
    flatCenter = center;
    flatBasis = basis;
    itemGroup.clear();
    for (const sprite of buildItemSprites(items)) {
      itemGroup.add(sprite);
    }
    flatController.recenter();
  }

  async function switchToFlatMode() {
    if (switchingModes) {
      return;
    }
    switchingModes = true;
    try {
      flatModeCameraPosition = cameraPositionArray(orbitController.camera);
      // Fresh entry -- derive center/basis from 3D, not a stale prior visit's.
      flatCenter = null;
      flatBasis = null;
      const halfHeight = Math.max(
          altitudeAboveSurface(flatModeCameraPosition), 0.01);

      orbitController.setEnabled(false);
      flatController = new FlatMapCameraController(
          sceneManager.renderer.domElement, { halfHeight });
      await Promise.all([refreshFlatMesh(), refreshFlatItems()]);
      sceneManager.setActiveController(flatController);

      flatController.onChange(() => {
        const visibleHalfHeight =
            flatController.camera.top / flatController.camera.zoom;
        if (visibleHalfHeight > FLAT_MODE_THRESHOLD * SIDE_LENGTH) {
          switchToOrbitMode();
          return;
        }
        // Re-centers the projection on wherever the view now is, exactly
        // (see `flatRecenterRequestFields`) -- without this, panning
        // would just slide a frozen snapshot around instead of updating
        // it in real time.
        Promise.all([refreshFlatMesh(), refreshFlatItems()]).catch((error) => {
          setStatus(`Failed to update flat view: ${error.message}`);
        });
      });
    } finally {
      switchingModes = false;
    }
  }

  function switchToOrbitMode() {
    // `orbitController.camera` hasn't moved since flat mode was entered
    // (it was disabled, not driven, the whole time) -- it's still sitting
    // right at the threshold that triggered the switch in the first
    // place. Without repositioning it here, the very next `onChange` tick
    // would see that same too-close position and immediately switch back
    // into flat mode. Push it out along its existing look direction to
    // match how far the user actually zoomed out while in flat mode --
    // the flat camera's own current zoom already encodes that distance
    // exactly, via the same `visibleHalfHeight`-as-altitude convention
    // `switchToFlatMode` used to set it up to begin with.
    const visibleHalfHeight =
        flatController.camera.top / flatController.camera.zoom;
    const direction = orbitController.camera.position.clone().normalize();
    orbitController.camera.position.copy(
        direction.multiplyScalar(TESSELLATION.radius + visibleHalfHeight));
    orbitController.camera.lookAt(0, 0, 0);

    flatController.dispose();
    flatController = null;
    flatModeCameraPosition = null;
    flatCenter = null;
    flatBasis = null;
    orbitController.setEnabled(true);
    sceneManager.setActiveController(orbitController);
    refreshLodMesh().catch((error) => {
      setStatus(`Failed to update mesh: ${error.message}`);
    });
    refreshItems().catch((error) => {
      setStatus(`Failed to update items: ${error.message}`);
    });
  }

  setStatus('Requesting mesh...');
  try {
    const meshData = await window.mundimonium.getLodMesh({
      ...TESSELLATION,
      cameraPosition: cameraPositionArray(orbitController.camera),
    });
    lastSectors = meshData.sectors;
    const built = buildShadedMesh(meshData);
    mesh = built.mesh;
    sceneManager.addToScene(mesh);
    wireframe = buildWireframe(mesh.geometry);
    sceneManager.addToScene(wireframe);
    for (const light of built.lights) {
      sceneManager.addToScene(light);
    }
    sceneManager.addToScene(itemGroup);
    sceneManager.addToScene(debugLabelGroup); // DEBUG ONLY
    await refreshItems();
    setStatus('');
  } catch (error) {
    setStatus(`Failed to load: ${error.message}`);
    return;
  }

  orbitController.onChange(() => {
    if (switchingModes || viewingGenericDemo) {
      return;
    }
    const cameraPosition = cameraPositionArray(orbitController.camera);
    if (altitudeAboveSurface(cameraPosition) < FLAT_MODE_THRESHOLD * SIDE_LENGTH) {
      switchToFlatMode().catch((error) => {
        setStatus(`Failed to switch to flat mode: ${error.message}`);
      });
      return;
    }
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
    const refresh = flatController ? refreshFlatMesh : refreshLodMesh;
    window.mundimonium
        .subdivideSector({ ...TESSELLATION, sector: lastSectors[0] })
        .then(refresh)
        .catch((error) => setStatus(`Failed to subdivide: ${error.message}`));
  });

  window.addEventListener('keydown', (event) => {
    if (event.key !== DEBUG_TOGGLE_GENERIC_KEY || flatController) {
      return; // no generic flat-map support yet -- stay in whichever mode
    }
    toggleGenericDemo().catch((error) => {
      setStatus(`Failed to toggle generic demo: ${error.message}`);
    });
  });

  async function toggleGenericDemo() {
    if (switchingModes) {
      return;
    }
    switchingModes = true;
    try {
      const cameraPosition = cameraPositionArray(orbitController.camera);
      if (viewingGenericDemo) {
        // Back to the spherical world, exactly as on initial load.
        const meshData = await window.mundimonium.getLodMesh(
            { ...TESSELLATION, cameraPosition });
        lastSectors = meshData.sectors;
        updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
        viewingGenericDemo = false;
        await refreshItems();
      } else {
        const meshData = await window.mundimonium.getMesh(GENERIC_TESSELLATION);
        lastSectors = []; // subdivide/LOD refresh isn't supported here
        // No `sectors` in a plain `getMesh` response -- no per-face
        // addresses to label, so `updateMeshGeometry` just clears any
        // debug labels left over from the spherical world.
        updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
        viewingGenericDemo = true;
        const { items } = await window.mundimonium.getItems(
            { ...GENERIC_TESSELLATION, cameraPosition });
        itemGroup.clear();
        for (const sprite of buildItemSprites(items)) {
          itemGroup.add(sprite);
        }
      }
    } finally {
      switchingModes = false;
    }
  }
}

main();
