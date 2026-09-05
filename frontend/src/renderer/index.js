import * as THREE from 'three';

import { SceneManager } from './scene/SceneManager.js';
import {
  FlatMapCameraController, OrbitCameraController,
} from './scene/CameraController.js';
import { buildGeometry, buildShadedMesh, buildWireframe } from './scene/MeshLoader.js';
import { buildItemSprites } from './scene/ItemLoader.js';
import { buildFaceIndexLabels } from './scene/DebugFaceLabels.js'; // DEBUG ONLY -- see its own docstring
import { SurfaceCameraController } from './scene/SurfaceCameraController.js';

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
// against. Deliberately small: flat mode should only ever show a handful
// of faces around the view center, not most of the mesh (kept in sync,
// by convention rather than shared code, with `flat_mesh_export.py`'s own
// `_FLAT_MODE_RENDER_RADIUS_HOPS`, which governs how large a patch the
// server actually renders once in flat mode).
const FLAT_MODE_THRESHOLD = 2.0;

// The surface-following camera reuses this same threshold, against its
// own hover height rather than altitude above a sphere's center: both
// cameras share the same vertical FOV and the same face-side-length
// normalization, so the same multiple-of-side-length distance produces
// the same visible-face-width framing at the transition regardless of
// which camera it's measured on -- there's no reason for the two to
// differ. `SurfaceCameraController`'s own `DEFAULT_HOVER_HEIGHT_FACE_
// WIDTHS` is set comfortably above this threshold (mirroring how the
// orbit camera's own default distance sits comfortably above it too),
// so ordinary 3D exploration at the default height isn't immediately
// swallowed into flat mode.

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

/**
 * Whether `flatController`'s camera has panned since its projection was
 * last centered -- `FlatMapCameraController.recenter()` always resets
 * the camera back to exactly `(0, 0)`, so a nonzero position here can
 * only come from real panning since then, never from zooming alone (an
 * orthographic `OrbitControls`' zoom only changes `camera.zoom`, never
 * `camera.position`). The flattened geometry depends only on where the
 * view is centered, not how far zoomed in/out it is, so this is exactly
 * the condition under which a fresh reflattening request is worth it.
 *
 * @param {FlatMapCameraController} flatController
 * @returns {boolean}
 */
function hasPanned(flatController) {
  const { x, y } = flatController.camera.position;
  return Math.hypot(x, y) > 1e-9;
}

/**
 * Wraps `refresh` (a flat view's paired mesh+items reflattening round-
 * trip) so that calling the returned function while a previous call is
 * still in flight never starts a second, overlapping request -- it just
 * remembers that another refresh is needed and runs exactly one more
 * once the current one settles, reflecting whatever the view's latest
 * state is by then, rather than the (possibly already stale) state that
 * requested it. At most one such follow-up is ever remembered (`queued`
 * is a single flag, not a list) -- a burst of several changes while a
 * request is in flight still only ever produces one more request after
 * it, using whatever is latest by the time that request actually goes
 * out.
 *
 * Without this, panning fast enough to fire the debounced `onChange`
 * again before a round-trip completes launches a second, redundant
 * request recomputing the same geometry -- wasted work, and slower to
 * catch up overall, since two round-trips end up serialized end-to-end
 * (the Python subprocess itself only ever handles one request at a time)
 * instead of the second one being skipped in favor of a single fresh
 * one afterward.
 *
 * @param {() => Promise<void>} refresh
 * @param {(error: Error) => void} onError
 * @returns {() => void} Requests a refresh.
 */
function serializeRefresh(refresh, onError) {
  let inFlight = false;
  let queued = false;

  async function run() {
    if (inFlight) {
      queued = true;
      return;
    }
    inFlight = true;
    try {
      await refresh();
    } catch (error) {
      onError(error);
    } finally {
      inFlight = false;
      if (queued) {
        queued = false;
        run();
      }
    }
  }

  return run;
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

  // True exactly while `DEBUG_TOGGLE_GENERIC_KEY`'s demo mesh is showing
  // (in either its own 3D or flat mode) -- suppresses the camera-driven
  // LOD/flat-mode refresh loop below, which only knows how to talk to the
  // spherical world's endpoints.
  let viewingGenericDemo = false;

  // The generic demo's own 3D camera -- a `SurfaceCameraController`
  // (hugs the mesh surface, not an orbit) -- non-null exactly while
  // viewing the generic demo in 3D. Unlike `orbitController`, this is
  // rebuilt fresh each time the demo is (re-)entered rather than kept
  // alive for the app's whole lifetime, since it's cheap to construct
  // and there's exactly one demo shape to build it from.
  let surfaceController = null;

  // The generic demo's own resolved flat-mode center (`{face, b, s}`) --
  // `null` until the first `getFlatMesh`/`getFlatItems` response arrives
  // after entering flat mode. Serves the same two purposes `flatCenter`
  // does below: every further refresh re-centers on it (offset by
  // however far `flatController`'s camera has panned since, resolved
  // server-side via `GenericTessellation
  // .unflatten_point_and_transport_orientation`), and `switchToGeneric
  // OrbitMode` resumes the surface camera exactly here on exit --
  // wherever the view has actually panned to, not the pre-entry
  // position. `genericFlatOrientation` is carried forward the same way
  // so the server can keep the projection's screen-space orientation
  // continuous across re-centers (see that method's own docstring) --
  // `null` until resolved via a pan, same as `genericFlatCenter`.
  let genericFlatCenter = null;
  let genericFlatOrientation = null;

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
  // `flatCenterPosition` is `flatCenter`'s own true 3D position (from
  // `getFlatMesh`'s own `center_position`), used by `switchToOrbitMode`
  // to resume the 3D camera wherever flat mode has actually panned to,
  // rather than its stale pre-entry position.
  let flatCenter = null;
  let flatBasis = null;
  let flatCenterPosition = null;

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
    flatCenterPosition = meshData.centerPosition;
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

  const scheduleFlatRefresh = serializeRefresh(
      () => flatController
          ? Promise.all([refreshFlatMesh(), refreshFlatItems()])
          : Promise.resolve(),
      (error) => setStatus(`Failed to update flat view: ${error.message}`));

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
      flatCenterPosition = null;
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
        if (!hasPanned(flatController)) {
          return; // zoom only -- the flattened geometry hasn't changed
        }
        // Re-centers the projection on wherever the view now is, exactly
        // (see `flatRecenterRequestFields`) -- without this, panning
        // would just slide a frozen snapshot around instead of updating
        // it in real time. Serialized via `scheduleFlatRefresh` so a
        // fast pan can never have two of these requests in flight at once.
        scheduleFlatRefresh();
      });
    } finally {
      switchingModes = false;
    }
  }

  function switchToOrbitMode() {
    // `orbitController.camera` hasn't moved since flat mode was entered
    // (it was disabled, not driven, the whole time), so it can't just be
    // left where it is -- resume along the direction to `flatCenter`'s
    // own true 3D position (`flatCenterPosition`, from the most recent
    // `getFlatMesh` response) instead of the camera's own stale pre-
    // entry position, so exiting respects wherever flat mode has
    // actually panned to since, not just where 3D mode was left off.
    // Pushed out to match how far the user actually zoomed out while in
    // flat mode -- the flat camera's own current zoom already encodes
    // that distance exactly, via the same `visibleHalfHeight`-as-
    // altitude convention `switchToFlatMode` used to set it up to begin
    // with -- which also keeps this comfortably past the threshold that
    // triggered the switch, so the very next `onChange` tick doesn't
    // immediately trigger it again.
    const visibleHalfHeight =
        flatController.camera.top / flatController.camera.zoom;
    const direction = new THREE.Vector3(...flatCenterPosition).normalize();
    orbitController.camera.position.copy(
        direction.multiplyScalar(TESSELLATION.radius + visibleHalfHeight));
    orbitController.camera.lookAt(0, 0, 0);

    flatController.dispose();
    flatController = null;
    flatModeCameraPosition = null;
    flatCenter = null;
    flatBasis = null;
    flatCenterPosition = null;
    orbitController.setEnabled(true);
    sceneManager.setActiveController(orbitController);
    refreshLodMesh().catch((error) => {
      setStatus(`Failed to update mesh: ${error.message}`);
    });
    refreshItems().catch((error) => {
      setStatus(`Failed to update items: ${error.message}`);
    });
  }

  async function refreshSurfaceItems() {
    const { items } = await window.mundimonium.getItems({
      ...GENERIC_TESSELLATION,
      cameraPosition: cameraPositionArray(surfaceController.camera),
    });
    itemGroup.clear();
    for (const sprite of buildItemSprites(items)) {
      itemGroup.add(sprite);
    }
  }

  /**
   * The request fields that tell the server to re-center (and re-
   * orient) exactly on wherever the generic flat view currently is --
   * `genericFlatCenter`/`genericFlatOrientation` (the previous
   * response's own resolved values) plus how far the camera has panned
   * from it since, for the server to resolve via `GenericTessellation
   * .unflatten_point_and_transport_orientation` (see `server.py`'s own
   * `_resolve_generic_flat_center_and_orientation`). Mirrors the
   * spherical flat mode's own `flatRecenterRequestFields`, simplified
   * since generic flat mode has no "first call" case needing a
   * different fallback: `switchToGenericFlatMode` always sets
   * `genericFlatCenter` from the surface camera's own exact position
   * before the first request.
   */
  function genericFlatRecenterRequestFields() {
    return {
      center: genericFlatCenter,
      orientation: genericFlatOrientation,
      panOffset: [
        flatController.camera.position.x, flatController.camera.position.y,
      ],
    };
  }

  async function refreshGenericFlatMesh() {
    const meshData = await window.mundimonium.getFlatMesh({
      ...GENERIC_TESSELLATION, ...genericFlatRecenterRequestFields(),
    });
    genericFlatCenter = meshData.center;
    genericFlatOrientation = meshData.orientation;
    updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
    flatController.recenter();
    return meshData;
  }

  async function refreshGenericFlatItems() {
    const { items, center, orientation } =
        await window.mundimonium.getFlatItems({
          ...GENERIC_TESSELLATION, ...genericFlatRecenterRequestFields(),
        });
    genericFlatCenter = center;
    genericFlatOrientation = orientation;
    itemGroup.clear();
    for (const sprite of buildItemSprites(items)) {
      itemGroup.add(sprite);
    }
  }

  const scheduleGenericFlatRefresh = serializeRefresh(
      () => flatController
          ? Promise.all([refreshGenericFlatMesh(), refreshGenericFlatItems()])
          : Promise.resolve(),
      (error) => setStatus(`Failed to update flat view: ${error.message}`));

  async function switchToGenericFlatMode() {
    if (switchingModes) {
      return;
    }
    switchingModes = true;
    try {
      genericFlatCenter = surfaceController.getFlatModeCenter();
      genericFlatOrientation = null; // fresh entry -- nothing to continue yet
      const halfHeight = Math.max(surfaceController.hoverHeight, 0.01);

      surfaceController.setEnabled(false);
      flatController = new FlatMapCameraController(
          sceneManager.renderer.domElement, { halfHeight });
      await Promise.all([refreshGenericFlatMesh(), refreshGenericFlatItems()]);
      sceneManager.setActiveController(flatController);

      flatController.onChange(() => {
        // Same shape as the spherical flat mode's own exit check
        // (`visibleHalfHeight > threshold`): entry and exit cross the
        // threshold in opposite directions (entry by zooming in, `h`
        // shrinking below it; exit by zooming back out, growing past it
        // again), so the check that just triggered entry is never still
        // satisfied the instant flat mode starts -- no immediate bounce.
        const visibleHalfHeight =
            flatController.camera.top / flatController.camera.zoom;
        if (visibleHalfHeight
            > FLAT_MODE_THRESHOLD * surfaceController.currentFaceSideLength) {
          switchToGenericOrbitMode().catch((error) => {
            setStatus(`Failed to switch to 3D: ${error.message}`);
          });
          return;
        }
        if (!hasPanned(flatController)) {
          return; // zoom only -- the flattened geometry hasn't changed
        }
        // Re-centers the projection on wherever the view now is, exactly
        // (see `genericFlatRecenterRequestFields`) -- without this,
        // panning would just slide a frozen snapshot around instead of
        // updating it in real time. Serialized via `scheduleGenericFlat
        // Refresh` so a fast pan can never have two of these requests in
        // flight at once.
        scheduleGenericFlatRefresh();
      });
    } finally {
      switchingModes = false;
    }
  }

  async function switchToGenericOrbitMode() {
    // Unlike the spherical `switchToOrbitMode`, `surfaceController`
    // already knows exactly which face/local-position to resume at --
    // no repositioning heuristic needed, just the flat view's own last
    // known center and a hover height matching how far it had zoomed out.
    const visibleHalfHeight =
        flatController.camera.top / flatController.camera.zoom;
    const { face, b, s } = genericFlatCenter;
    surfaceController.recenterAt(face, b, s, Math.max(visibleHalfHeight, 0.01));

    flatController.dispose();
    flatController = null;
    genericFlatCenter = null;
    genericFlatOrientation = null;
    surfaceController.setEnabled(true);
    sceneManager.setActiveController(surfaceController);

    // The mesh geometry is still whatever flat mode last set it to (a
    // flattened, `z = 0` snapshot) -- restore the true 3D one. Re-fetched
    // rather than cached from generic-3D's own initial load, matching the
    // spherical `switchToOrbitMode`'s own "always re-fetch on a mode
    // switch" precedent, even though this particular mesh never changes.
    const meshData = await window.mundimonium.getMesh(GENERIC_TESSELLATION);
    updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
    await refreshSurfaceItems();
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
      return; // ambiguous which flat mode would apply -- stay put
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
      if (viewingGenericDemo) {
        // Back to the spherical world, exactly as on initial load.
        surfaceController.dispose();
        surfaceController = null;
        const cameraPosition = cameraPositionArray(orbitController.camera);
        const meshData = await window.mundimonium.getLodMesh(
            { ...TESSELLATION, cameraPosition });
        lastSectors = meshData.sectors;
        updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
        viewingGenericDemo = false;
        orbitController.setEnabled(true);
        sceneManager.setActiveController(orbitController);
        await refreshItems();
      } else {
        const meshData = await window.mundimonium.getMesh(GENERIC_TESSELLATION);
        lastSectors = []; // subdivide/LOD refresh isn't supported here
        // No `sectors` in a plain `getMesh` response -- no per-face
        // addresses to label, so `updateMeshGeometry` just clears any
        // debug labels left over from the spherical world.
        updateMeshGeometry(buildGeometry(meshData), meshData.sectors);
        viewingGenericDemo = true;

        orbitController.setEnabled(false);
        surfaceController = new SurfaceCameraController(
            sceneManager.renderer.domElement, meshData);
        sceneManager.setActiveController(surfaceController);
        surfaceController.onChange(() => {
          const h = surfaceController.hoverHeight;
          const sideLength = surfaceController.currentFaceSideLength;
          if (h < FLAT_MODE_THRESHOLD * sideLength) {
            switchToGenericFlatMode().catch((error) => {
              setStatus(`Failed to switch to flat mode: ${error.message}`);
            });
            return;
          }
          refreshSurfaceItems().catch((error) => {
            setStatus(`Failed to update items: ${error.message}`);
          });
        });
        await refreshSurfaceItems();
      }
    } finally {
      switchingModes = false;
    }
  }
}

main();
