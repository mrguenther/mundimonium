import * as THREE from 'three';

import { ChangeDebouncer } from './ChangeDebouncer.js';

// Default/min hover height, in the mesh's own real-distance units --
// tunable placeholders, expected to need empirical adjustment once
// there's a real scene to look at (matching every other such threshold
// in this project). Expressed as multiples of a face's own side length,
// since the mesh's absolute scale is arbitrary. The default sits well
// above `index.js`'s own `FLAT_MODE_THRESHOLD` (2 face-widths) so
// ordinary 3D exploration doesn't immediately fall into flat mode --
// mirroring how the orbit camera's own default distance from a sphere
// sits comfortably above its analogous threshold too.
const DEFAULT_HOVER_HEIGHT_FACE_WIDTHS = 4.0;
const MIN_HOVER_HEIGHT_FACE_WIDTHS = 0.1;

// How many pixels of vertical drag correspond to one full vertical field
// of view's worth of world-space movement, mirrored horizontally by
// aspect ratio -- see `_pan`.
const VERTICAL_FOV_DEGREES = 50;

// `focalPointMarker`'s radius, as a fraction of the current hover height
// `h` -- scaling with `h` (rather than a fixed world-space size) keeps its
// on-screen size roughly constant as the camera zooms in/out, matching how
// `_pan`'s own pixel-to-world conversion already scales with `h`.
const FOCAL_POINT_MARKER_RADIUS_RATIO = 0.03;

// Below this squared length, an angle-weighted (or, failing that, plain)
// sum of a vertex's adjacent face normals is treated as degenerate -- see
// `computeVertexNormals`'s own fallback. A placeholder threshold, not
// expected to trigger on this project's current demo mesh.
const VERTEX_NORMAL_DEGENERATE_LENGTH_SQ = 1e-6;

// Safety cap on how many faces `_locateFace` will cross while walking a
// single step -- see its own docstring for why this loops rather than
// recurses. Generous relative to how many faces an ordinary drag step
// crosses (usually 0-2) without being unbounded.
const MAX_FACE_LOCATE_STEPS = 64;

// The largest single sub-step `_applyLocalStep` will decompose onto a
// face's own basis before possibly crossing into a neighbor and
// switching to *its* basis instead, as a fraction of the current face's
// own side length -- see `_applyLocalStep`'s own docstring for why a
// large step needs to be subdivided at all.
const MAX_STEP_FRACTION_OF_SIDE_LENGTH = 0.1;

// Hard cap on how many sub-steps a single `_applyLocalStep` call will
// take, regardless of how large `worldDelta` is -- keeps an unusually
// large single pointer-move delta (a very fast drag) cheap, at the cost
// of possibly under-subdividing it (rare in practice: an ordinary drag's
// per-event pixel delta is already small).
const MAX_STEPS_PER_PAN = 32;

/**
 * One outward-facing "smoothed normal" per vertex -- an angle-weighted
 * average of the flat normals of every face touching that vertex,
 * normalized. Weighting by the angle each face subtends at the vertex
 * (rather than a plain unweighted average) is the standard refinement for
 * an irregular mesh: an unweighted average skews toward whichever faces
 * happen to be more numerous or more acute at that vertex, less
 * representative of the true local surface direction. This is the same
 * idea `MeshLoader.js` already uses (via Three's own, area-weighted
 * `computeVertexNormals()`) to smooth-shade the *rendered* mesh -- applied
 * here to drive the camera's own viewing direction instead of lighting,
 * independently (see the class docstring).
 *
 * Only needs `vertexFaces[v]`'s membership, not cyclic order -- a plain
 * weighted sum doesn't care what order its terms arrive in. Confirmed via
 * `mesh_export.py`'s own `faces_around_vertex` docstring: even its
 * boundary-vertex fallback (when the fan doesn't close into a cycle) is
 * still a *complete* list, just unordered -- so unlike this file's own
 * previous dihedral-fan-chaining design, there's no boundary-vertex
 * caveat here at all.
 *
 * @param {THREE.Vector3[]} vertexPositions
 * @param {{ vertexIndices: number[], normal: THREE.Vector3 }[]} triangles
 * @param {number[][]} vertexFaces
 * @returns {THREE.Vector3[]}
 */
function computeVertexNormals(vertexPositions, triangles, vertexFaces) {
  return vertexFaces.map((fan, vertexIndex) => {
    const vertexPosition = vertexPositions[vertexIndex];
    const weightedSum = new THREE.Vector3();
    for (const faceIndex of fan) {
      const triangle = triangles[faceIndex];
      const slot = triangle.vertexIndices.indexOf(vertexIndex);
      const otherA = vertexPositions[triangle.vertexIndices[(slot + 1) % 3]];
      const otherB = vertexPositions[triangle.vertexIndices[(slot + 2) % 3]];
      const edgeA = otherA.clone().sub(vertexPosition).normalize();
      const edgeB = otherB.clone().sub(vertexPosition).normalize();
      const angle = Math.acos(
          THREE.MathUtils.clamp(edgeA.dot(edgeB), -1, 1));
      weightedSum.addScaledVector(triangle.normal, angle);
    }
    if (weightedSum.lengthSq() > VERTEX_NORMAL_DEGENERATE_LENGTH_SQ) {
      return weightedSum.normalize();
    }
    // Degenerate: the angle-weighted sum nearly cancels (e.g. a sharply
    // folded valley where opposing faces point almost oppositely) -- fall
    // back to a plain unweighted average, and if that's *also*
    // degenerate, to the first adjacent face's own flat normal. Not
    // expected to trigger on this project's current demo mesh; a safety
    // net rather than a carefully-designed resolution.
    const plainSum = fan.reduce(
        (sum, f) => sum.add(triangles[f].normal), new THREE.Vector3());
    if (plainSum.lengthSq() > VERTEX_NORMAL_DEGENERATE_LENGTH_SQ) {
      return plainSum.normalize();
    }
    return triangles[fan[0]].normal.clone();
  });
}

/**
 * Builds the static, per-triangle geometry (and per-vertex smoothed
 * normals -- see `computeVertexNormals`) a `SurfaceCameraController`
 * needs from a `getMesh` response's raw buffers -- computed once when the
 * mesh loads, not per frame, so panning never re-derives it.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array,
 *   adjacency: number[][], vertexFaces: number[][] }} meshData
 */
function buildSurfaceGraph(meshData) {
  const { positions, indices, adjacency, vertexFaces } = meshData;

  const vertexPositions = [];
  for (let i = 0; i < positions.length / 3; i++) {
    vertexPositions.push(new THREE.Vector3(
        positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]));
  }

  const triangles = [];
  for (let i = 0; i < indices.length / 3; i++) {
    const vertexIndices = [
      indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2],
    ];
    const [origin, sPos, dPos] = vertexIndices.map(
        (v) => vertexPositions[v]);

    const eX = sPos.clone().sub(origin);
    const sideLength = eX.length();
    eX.normalize();

    const rawEY = dPos.clone().sub(origin);
    const eY = rawEY.clone().addScaledVector(eX, -rawEY.dot(eX));
    const altitude = eY.length();
    eY.normalize();

    // `eX`/`eY` are wound the same way `IsometricPoint`'s own canonical
    // frame is (vertex B at the origin, S along +u, D along +v) -- see
    // `mundimonium/coordinates/tessellation.py`'s `vertex_b/s/d`. Their
    // cross product points outward, matching this project's established
    // CCW-when-viewed-from-outside winding convention (verified directly
    // against the actual demo mesh before writing this).
    const normal = new THREE.Vector3().crossVectors(eX, eY).normalize();

    triangles.push({
      vertexIndices, origin, eX, eY, normal, sideLength, altitude,
      neighbors: adjacency[i],
    });
  }

  const vertexNormals = computeVertexNormals(
      vertexPositions, triangles, vertexFaces);

  return { triangles, vertexNormals, vertexFaces };
}

/** The surface point (before the hover-height offset) for local `(u, v)`
 * -- see `buildSurfaceGraph`'s `eX`/`eY` convention. */
function facePoint(triangle, u, v) {
  return triangle.origin.clone()
      .addScaledVector(triangle.eX, u)
      .addScaledVector(triangle.eY, v);
}

/** The 3D position of `triangle`'s own vertex at `slot` (0 = B, 1 = S,
 * 2 = D -- see `buildSurfaceGraph`'s `eX`/`eY` convention), derived from
 * the triangle's own cached geometry rather than a separate vertex
 * lookup. */
function triangleVertexPosition(triangle, slot) {
  if (slot === 0) {
    return triangle.origin.clone();
  }
  if (slot === 1) {
    return facePoint(triangle, triangle.sideLength, 0);
  }
  return facePoint(triangle, triangle.sideLength / 2, triangle.altitude);
}

/**
 * `fan`'s own face indices, reordered by hop-distance from `fromIndex`
 * (a position *within* `fan`, not a face index itself) -- its two
 * immediate neighbors first, then their neighbors, and so on outward,
 * alternating direction at each distance; `fromIndex` itself is not
 * included. Falls back to `fan`'s own raw order if `fromIndex` is `-1`
 * (not found) -- see `_resolveVertexExit`'s own docstring for why this
 * order (nearest first) matters, not just correctness.
 */
function vertexFanByHopDistance(fan, fromIndex) {
  if (fromIndex === -1) {
    return fan;
  }
  const n = fan.length;
  const order = [];
  for (let distance = 1; distance <= Math.floor(n / 2); distance++) {
    const forward = fan[(fromIndex + distance) % n];
    const backward = fan[(fromIndex - distance + n) % n];
    order.push(forward);
    if (backward !== forward) {
      order.push(backward);
    }
  }
  return order;
}

/** `vector` rotated by `angle` radians around unit vector `axis`
 * (Rodrigues' rotation formula). */
function rotateAroundAxis(vector, axis, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return vector.clone().multiplyScalar(cos)
      .add(new THREE.Vector3().crossVectors(axis, vector).multiplyScalar(sin))
      .addScaledVector(axis, axis.dot(vector) * (1 - cos));
}

/**
 * A unit vector along the edge shared by `triangleIndexA` and
 * `triangleIndexB` (which must actually be edge-adjacent) -- the
 * physical rotation axis for unfolding one face onto the other (see
 * `_locateFace`'s own docstring). Sign is arbitrary (whichever endpoint
 * order the slot lookup happens to give) and doesn't matter to any
 * caller: `rotateAroundAxis(v, axis, angle)` and `rotateAroundAxis(v,
 * -axis, -angle)` are the same rotation, and `dihedralRotationAngle`
 * derives its own signed angle relative to whichever axis it's given.
 */
function sharedEdgeAxis(graph, triangleIndexA, triangleIndexB) {
  const triangleA = graph.triangles[triangleIndexA];
  const slot = triangleA.neighbors.indexOf(triangleIndexB);
  const otherSlots = [0, 1, 2].filter((s) => s !== slot);
  const posA = triangleVertexPosition(triangleA, otherSlots[0]);
  const posB = triangleVertexPosition(triangleA, otherSlots[1]);
  return posB.sub(posA).normalize();
}

/**
 * The signed rotation angle about `edgeAxis` that takes `normalA` to
 * `normalB` -- i.e. `rotateAroundAxis(normalA, edgeAxis, angle)` equals
 * `normalB` exactly.
 *
 * Recovered via `atan2` against an in-plane companion axis rather than
 * `acos(dot(normalA, normalB))` (which can't recover the *sign*, only
 * the magnitude) combined with a cross-product-derived axis (which
 * becomes numerically unreliable as the angle between the normals
 * approaches 180 degrees, since the cross product's own magnitude
 * vanishes there even though the angle itself is perfectly well-defined)
 * -- this mesh has folds sharp enough for that to matter.
 */
function dihedralRotationAngle(normalA, normalB, edgeAxis) {
  const inPlaneAxis = new THREE.Vector3()
      .crossVectors(edgeAxis, normalA).normalize();
  const cosAngle = normalB.dot(normalA);
  const sinAngle = normalB.dot(inPlaneAxis);
  return Math.atan2(sinAngle, cosAngle);
}

/**
 * A camera that hugs a triangle mesh's surface at a fixed hover height,
 * panning across faces with a continuously-varying viewing direction --
 * see `computeVertexNormals`/`_currentNormal`'s own docstrings for how.
 * Implements the same `{ camera, update(deltaSeconds), onChange(callback),
 * dispose() }` shape as `OrbitCameraController`/`FlatMapCameraController`,
 * so `SceneManager`/`index.js` don't need to know it works differently
 * under the hood.
 *
 * Can't reuse `OrbitControls` -- its math assumes orbiting a fixed
 * target, not walking a surface -- so pointer events are handled here
 * directly instead.
 *
 * Also owns `focalPointMarker`, a small red sphere at the camera's own
 * focal point (`_currentSurfacePoint()`, the point on the mesh the
 * camera is hovering above/looking at) -- a debugging aid for spotting
 * exactly where the camera's motion is misbehaving during a pan. The
 * caller (`index.js`) is responsible for adding it to the scene and
 * removing it on disposal, matching how every other piece of scene
 * composition is owned outside this class.
 */
export class SurfaceCameraController {
  /**
   * @param {HTMLElement} domElement
   * @param {{ positions: Float32Array, indices: Uint32Array,
   *   adjacency: number[][], vertexFaces: number[][] }} meshData
   * @param {{ faceIndex?: number, u?: number, v?: number, h?: number }}
   *   [initialState] - Where to start; defaults to face 0's own centroid
   *   at a default hover height.
   */
  constructor(domElement, meshData, initialState = {}) {
    this._domElement = domElement;
    this._graph = buildSurfaceGraph(meshData);
    this._dispatcher = new THREE.EventDispatcher();
    this._changeDebouncer = new ChangeDebouncer(this._dispatcher);
    this._enabled = true;

    this.camera = new THREE.PerspectiveCamera(
        VERTICAL_FOV_DEGREES,
        window.innerWidth / window.innerHeight, 0.01, 1000);

    this.focalPointMarker = new THREE.Mesh(
        new THREE.SphereGeometry(1, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xff0000 }));

    const startTriangle = this._graph.triangles[initialState.faceIndex ?? 0];
    const defaultH = startTriangle.sideLength * DEFAULT_HOVER_HEIGHT_FACE_WIDTHS;
    this._state = {
      faceIndex: initialState.faceIndex ?? 0,
      u: initialState.u ?? startTriangle.sideLength / 2,
      v: initialState.v ?? startTriangle.altitude / 3,
      h: initialState.h ?? defaultH,
    };

    this._updateCameraFromState();

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onWheel = this._onWheel.bind(this);
    domElement.addEventListener('pointerdown', this._onPointerDown);
    domElement.addEventListener('wheel', this._onWheel, { passive: false });

    this._dragPointerId = null;
    this._dragLastX = 0;
    this._dragLastY = 0;
  }

  /**
   * @param {() => void} callback
   * @param {number} [debounceMs]
   * @returns {() => void} Unsubscribe function.
   */
  onChange(callback, debounceMs) {
    return this._changeDebouncer.onChange(callback, debounceMs);
  }

  /** @param {number} _deltaSeconds */
  update(_deltaSeconds) {
    // No per-frame damping/animation -- the camera only moves in direct
    // response to pointer/wheel events, applied immediately.
  }

  /** @param {boolean} enabled */
  setEnabled(enabled) {
    this._enabled = enabled;
    // Hidden rather than removed from the scene while disabled (e.g. flat
    // mode) -- its position stops updating too, so leaving it visible
    // would show a stale, misleading point.
    this.focalPointMarker.visible = enabled;
  }

  dispose() {
    this._domElement.removeEventListener('pointerdown', this._onPointerDown);
    this._domElement.removeEventListener('wheel', this._onWheel);
    window.removeEventListener('pointermove', this._onPointerMove);
    window.removeEventListener('pointerup', this._onPointerUp);
    this._changeDebouncer.dispose();
  }

  /**
   * The current position as a flat-mode entry point, for the caller to
   * pass into `getFlatMesh`/`getFlatItems`'s own `center` field.
   *
   * `IsometricPoint(face, b, s)`'s own `(b, s)` is a *different*
   * convention from this file's own local `(u, v)` (see `buildSurfaceGraph`'s
   * `eX`/`eY`): vertex D sits at `(b, s) = (0, 0)`, vertex B at
   * `(altitude, 0)`, vertex S at `(0, altitude)` -- confirmed directly
   * against `IsometricPoint`/`point_to_3d_position` before writing this,
   * since getting this conversion wrong silently produces a valid-looking
   * (b, s) pair that resolves to a nonsensical position server-side.
   * `_faceBarycentric`'s own weights are convention-independent (an
   * intrinsic property of the point's location, not of `(u, v)`'s own
   * basis), so converting through them is exact.
   *
   * @returns {{ face: number, b: number, s: number }}
   */
  getFlatModeCenter() {
    const { faceIndex, u, v } = this._state;
    const triangle = this._graph.triangles[faceIndex];
    const { wb, ws } = this._faceBarycentric(triangle, u, v);
    return { face: faceIndex, b: wb * triangle.altitude, s: ws * triangle.altitude };
  }

  /**
   * Resets this controller to the given face-local position -- used when
   * *exiting* flat mode, to resume exactly where the flat view's own last
   * center was, rather than some arbitrary default position.
   *
   * @param {number} faceIndex
   * @param {number} isometricB - In `IsometricPoint(face, b, s)`'s own
   *   convention (see `getFlatModeCenter`'s docstring), not this file's
   *   own local `(u, v)`.
   * @param {number} isometricS
   * @param {number} h
   */
  recenterAt(faceIndex, isometricB, isometricS, h) {
    const triangle = this._graph.triangles[faceIndex];
    const wb = isometricB / triangle.altitude;
    const ws = isometricS / triangle.altitude;
    const wd = 1 - wb - ws;
    const u = ws * triangle.sideLength + (wd * triangle.sideLength) / 2;
    const v = wd * triangle.altitude;
    this._state = { faceIndex, u, v, h };
    this._updateCameraFromState();
  }

  /** The current hover height, in the mesh's own real-distance units. */
  get hoverHeight() {
    return this._state.h;
  }

  /** The `side_length` of the face the camera is currently over -- used
   * by `index.js` for the flat-mode threshold check. */
  get currentFaceSideLength() {
    return this._graph.triangles[this._state.faceIndex].sideLength;
  }

  // -------------------------------------------------------------------
  // Position/orientation from state
  // -------------------------------------------------------------------

  /** The current surface point (before the hover-height offset). */
  _currentSurfacePoint() {
    const { faceIndex, u, v } = this._state;
    return facePoint(this._graph.triangles[faceIndex], u, v);
  }

  /**
   * The current viewing direction: the current face's three vertex
   * normals (see `computeVertexNormals`), barycentrically interpolated at
   * `(u, v)` and renormalized -- Phong normal interpolation. This is what
   * makes the surface, as far as the camera's orientation is concerned,
   * appear continuously curved rather than faceted: adjacent faces share
   * the same two vertex normals along their common edge, so this matches
   * exactly from both sides, and every face touching a vertex converges
   * to that vertex's own single normal there -- no special-casing needed
   * at edges or vertices at all (unlike this file's previous
   * edge/vertex-patch design).
   */
  _currentNormal() {
    const { faceIndex, u, v } = this._state;
    const triangle = this._graph.triangles[faceIndex];
    const { wb, ws, wd } = this._faceBarycentric(triangle, u, v);
    const [nb, ns, nd] = triangle.vertexIndices.map(
        (vertexIndex) => this._graph.vertexNormals[vertexIndex]);
    return nb.clone().multiplyScalar(wb)
        .addScaledVector(ns, ws)
        .addScaledVector(nd, wd)
        .normalize();
  }

  /**
   * Positions/orients the camera from the current state, then carries
   * the *rendered* `up` forward continuously via a rotation-minimizing
   * update: rotate the previous frame's own `up` by whatever rotation
   * took the previous `_currentNormal()` to the current one. Works
   * because `_currentNormal()` is itself continuous (see its own
   * docstring) -- incrementally rotating `up` by the same rotation the
   * normal just underwent keeps the camera's own roll continuous too,
   * regardless of how far a single step moves.
   */
  _updateCameraFromState() {
    const surfacePoint = this._currentSurfacePoint();
    const normal = this._currentNormal();
    const position = surfacePoint.clone().addScaledVector(
        normal, this._state.h);
    this.camera.position.copy(position);

    this.focalPointMarker.position.copy(surfacePoint);
    this.focalPointMarker.scale.setScalar(
        this._state.h * FOCAL_POINT_MARKER_RADIUS_RATIO);

    if (this._visualUp === undefined) {
      // First-ever call: bootstrap from the starting face's own `eY` --
      // only approximately orthogonal to the true interpolated normal,
      // corrected below the same as every later frame.
      this._visualUp = this._graph.triangles[this._state.faceIndex]
          .eY.clone();
    } else {
      const rotationAxis = new THREE.Vector3()
          .crossVectors(this._previousNormal, normal);
      if (rotationAxis.lengthSq() > 1e-12) {
        const rotationAngle = Math.acos(THREE.MathUtils.clamp(
            this._previousNormal.dot(normal), -1, 1));
        this._visualUp = rotateAroundAxis(
            this._visualUp, rotationAxis.normalize(), rotationAngle);
      }
    }
    // Re-orthogonalize against the current normal every time (not just
    // the rotate-forward branch) -- corrects both the bootstrap frame's
    // only-approximate `eY` and ordinary floating-point drift.
    this._visualUp
        .addScaledVector(normal, -this._visualUp.dot(normal))
        .normalize();
    this._previousNormal = normal.clone();

    this.camera.up.copy(this._visualUp);
    this.camera.lookAt(surfacePoint);
  }

  // -------------------------------------------------------------------
  // Pointer/wheel handling
  // -------------------------------------------------------------------

  _onPointerDown(event) {
    if (!this._enabled || event.button !== 2) {
      return; // right-drag pans, matching FlatMapCameraController's own convention
    }
    this._dragPointerId = event.pointerId;
    this._dragLastX = event.clientX;
    this._dragLastY = event.clientY;
    window.addEventListener('pointermove', this._onPointerMove);
    window.addEventListener('pointerup', this._onPointerUp);
    event.preventDefault();
  }

  _onPointerMove(event) {
    if (event.pointerId !== this._dragPointerId) {
      return;
    }
    const dx = event.clientX - this._dragLastX;
    const dy = event.clientY - this._dragLastY;
    this._dragLastX = event.clientX;
    this._dragLastY = event.clientY;
    this._pan(dx, dy);
  }

  _onPointerUp(event) {
    if (event.pointerId !== this._dragPointerId) {
      return;
    }
    this._dragPointerId = null;
    window.removeEventListener('pointermove', this._onPointerMove);
    window.removeEventListener('pointerup', this._onPointerUp);
  }

  _onWheel(event) {
    if (!this._enabled) {
      return;
    }
    event.preventDefault();
    const zoomFactor = Math.exp(event.deltaY * 0.001);
    const minH = this._graph.triangles[this._state.faceIndex].sideLength
        * MIN_HOVER_HEIGHT_FACE_WIDTHS;
    this._state.h = Math.max(minH, this._state.h * zoomFactor);
    this._updateCameraFromState();
    this._dispatcher.dispatchEvent({ type: 'change' });
  }

  /**
   * `{ right, up }` -- an orthonormal basis spanning `triangle`'s own
   * flat plane, used to convert a pixel drag into a `(du, dv)` step
   * within that specific face. Built by *projecting* `_visualUp` onto
   * the plane (removing whatever component it has along the face's own
   * flat normal, then renormalizing), not by decomposing a delta that
   * was built in some other plane -- see `_pan`'s own docstring for why
   * that distinction is the actual fix for this method's history.
   *
   * `_visualUp` is deliberately the *exact same* vector
   * `_updateCameraFromState` renders `camera.up` from: "drag right"
   * needs to mean the same thing on screen no matter which face it's
   * currently being interpreted against or how many faces a drag has
   * crossed, and the only way to *guarantee* that is to interpret the
   * drag using the identical vector that determines what's actually
   * rendered, rather than a second, separately-tracked one. This part of
   * the design went through two wrong attempts before landing here:
   * - Each face's own *raw* `(eX, eY)` has no attenuation risk, but is
   *   independently authored per face (whatever `mesh_export.py`'s own
   *   vertex winding happens to produce), with no relationship between
   *   neighbors at all -- so "screen right", read off a different face's
   *   own raw axes after every crossing, could rotate by an arbitrary
   *   angle including 180 degrees, becoming "screen left" and sending
   *   the drag straight back the way it came.
   * - A *separate* continuously-transported vector (anchored to each
   *   face's own flat normal) fixed that, but not the underlying
   *   problem: transported independently of `_visualUp` (anchored to the
   *   smoothed normal instead), the two could drift out of alignment
   *   with each other over a sequence of crossings even though each was
   *   individually continuous -- so the drag's own internal bookkeeping
   *   could stay perfectly smooth while what actually got *rendered*
   *   still visibly changed direction relative to it.
   */
  _tangentBasisForTriangle(triangle) {
    const flatNormal = triangle.normal;
    const up = this._visualUp.clone()
        .addScaledVector(flatNormal, -this._visualUp.dot(flatNormal))
        .normalize();
    const right = new THREE.Vector3().crossVectors(up, flatNormal).normalize();
    return { right, up };
  }

  /**
   * Converts a screen-space pixel drag `(dx, dy)` into a world-space
   * step and applies it via `_applyLocalStep`.
   *
   * The pixel-to-world conversion uses the same similar-triangles
   * relationship a hovering perspective camera implies: at hover height
   * `h` and vertical field of view `fov`, one vertical pixel of drag
   * corresponds to `2 * h * tan(fov / 2) / screenHeightPixels` world
   * units of tangential movement (horizontal analogously, scaled by
   * aspect ratio) -- this keeps pan speed visually consistent across
   * zoom levels.
   *
   * Deliberately passes the raw pixel deltas (and this conversion
   * factor) through to `_applyLocalStep` rather than building one 3D
   * `worldDelta` vector here: `_tangentBasisForTriangle` depends on
   * *which* face's plane it's projecting onto, so a single upfront
   * `worldDelta` (built against only the starting face) would still need
   * reprojecting for every face a multi-face drag crosses -- exactly the
   * lossy step an earlier version of this design tried to patch after
   * the fact with a rescale. Recomputing the tangent basis fresh for
   * whichever face is current, once per sub-step, means every sub-step's
   * `(du, dv)` is exact by construction, with nothing to correct for.
   */
  _pan(dxPixels, dyPixels) {
    const h = this._state.h;
    const halfFovRadians = THREE.MathUtils.degToRad(VERTICAL_FOV_DEGREES / 2);
    const worldPerPixelY = (2 * h * Math.tan(halfFovRadians))
        / this._domElement.clientHeight;
    const worldPerPixelX = worldPerPixelY
        * (this._domElement.clientWidth / this._domElement.clientHeight);

    this._applyLocalStep(dxPixels, dyPixels, worldPerPixelX, worldPerPixelY);
    this._updateCameraFromState();
    this._dispatcher.dispatchEvent({ type: 'change' });
  }

  /**
   * Applies a screen-space pixel drag to the current position, walking
   * across face boundaries as needed via `_locateFace`.
   *
   * Applied in several small sub-steps rather than one big one so that
   * `_tangentBasisForTriangle` gets re-evaluated against `_visualUp`
   * partway through a large, multi-face drag, not just once at the
   * start. `_locateFace` itself already crosses any number of faces
   * correctly regardless of step size (see its own docstring), but its
   * "unfold across the shared edge" technique only carries forward
   * whatever direction it's given -- a single huge step would commit to
   * the *starting* face's own tangent basis and propagate that direction
   * via pure flat-geometry rotation the rest of the way, without ever
   * re-consulting `_visualUp` again until the step finishes. Sub-stepping
   * keeps the drag's own direction tied to `_visualUp` throughout the
   * walk, not just at its start.
   *
   * Each sub-step recomputes `_tangentBasisForTriangle` fresh for
   * whichever face is *current* at that point, rather than reusing one
   * `worldDelta` built against a single face and decomposing it onto
   * every face a multi-face drag happens to cross: a fixed `worldDelta`,
   * projected onto a face its own basis wasn't built from, can lose most
   * of its magnitude near enough curvature.
   *
   * *Rescaling* that projection back up (an earlier version of this
   * method did) doesn't really fix it, since a projection that's mostly
   * cancellation is mostly rounding noise, and rescaling amplifies that
   * noise to full strength instead of the intended direction -- confirmed
   * directly via a diagnostic showing rescale factors as high as ~13x
   * (i.e. over 90% of the projected step was noise), the kind of bug a
   * plain "did it eventually cover the total distance" check doesn't
   * catch, since the net distance can look fine while the
   * moment-to-moment direction is erratic, reading as "can't reliably
   * steer into that face" even though nothing is technically stuck.
   * Computing the basis directly in whichever plane is current sidesteps
   * the problem instead of correcting for it after the fact: the
   * projection is exact by construction, so there's nothing left to lose
   * or to amplify.
   */
  _applyLocalStep(dxPixels, dyPixels, worldPerPixelX, worldPerPixelY) {
    const totalLength = Math.hypot(
        dxPixels * worldPerPixelX, dyPixels * worldPerPixelY);
    if (totalLength < 1e-12) {
      return;
    }
    const startSideLength = this._graph.triangles[this._state.faceIndex]
        .sideLength;
    const maxStepLength = startSideLength * MAX_STEP_FRACTION_OF_SIDE_LENGTH;
    const stepCount = Math.min(MAX_STEPS_PER_PAN,
        Math.max(1, Math.ceil(totalLength / maxStepLength)));
    const dxSub = dxPixels / stepCount;
    const dySub = dyPixels / stepCount;
    for (let i = 0; i < stepCount; i++) {
      const { faceIndex, u, v } = this._state;
      const triangle = this._graph.triangles[faceIndex];
      // Screen `+x` is `right`; screen `+y` (downward) is `-up`, matching
      // how a drag "down" should move the view "up" relative to the
      // world (the same convention `OrbitControls`' own panning uses).
      const { right, up } = this._tangentBasisForTriangle(triangle);
      const worldDelta = right.clone()
          .multiplyScalar(-dxSub * worldPerPixelX)
          .addScaledVector(up, dySub * worldPerPixelY);
      const du = worldDelta.dot(triangle.eX);
      const dv = worldDelta.dot(triangle.eY);
      this._locateFace(faceIndex, u, v, du, dv);
    }
  }

  /**
   * Walks local step `(du, dv)` from `(u, v)` on `startFaceIndex`,
   * crossing into neighboring faces as needed -- a single step can cross
   * several faces at once (a fast pan, or a large jump). Loops rather
   * than recurses: an unbounded recursive cascade (this file's own
   * previous design) has no cap on how many faces a single step can walk
   * through, exactly the shape of bug this file's redesign started from
   * in the first place (indefinite bouncing between patches). Capped at
   * `MAX_FACE_LOCATE_STEPS` -- if still outside after that many crossings
   * (shouldn't happen on this mesh for an ordinary drag), clamps into
   * whatever face it last tried rather than looping forever.
   *
   * Ordinary edge crossings (a single negative barycentric weight)
   * "unfold" the *remaining* portion of `(du, dv)` across the shared
   * edge rather than naively reprojecting the extended target point onto
   * the neighbor's own, differently-oriented plane. An earlier version
   * of this method did the latter, and it doesn't correctly represent
   * "keep going straight" once the dihedral fold is sharp enough (this
   * mesh's isn't gentle): the extended point lies in the *current* face's
   * own flat plane, extrapolated past its boundary, which is not the
   * same thing as a point actually further along the mesh's own surface,
   * and reprojecting it via plain dot products silently discards
   * whatever out-of-plane component it has.
   *
   * Confirmed directly as a real bug via a diagnostic that bypassed
   * pixel-drag simulation entirely: for 84 of this mesh's 180 directed
   * face-neighbor pairs, walking straight from a face's own centroid
   * through the opposite edge's midpoint and well past it never actually
   * crossed at all -- each attempt reprojected to a point that, on the
   * neighbor's own plane, *still* tested as being on the near side,
   * sending it back across the same edge, converging geometrically on
   * the edge itself rather than progressing (a stable, self-reinforcing
   * oscillation, not a one-off glitch: dragging further in the same
   * direction doesn't help, since every attempt re-triggers the
   * identical round trip).
   *
   * The correct technique -- unfolding, i.e. rotating the remaining
   * direction of travel by the exact dihedral angle between the two
   * faces' own flat normals about their shared edge axis before
   * continuing on the neighbor -- is the standard way to walk a straight
   * line across a folded triangle mesh, and is exact regardless of how
   * sharp the fold is (see `dihedralRotationAngle`'s own docstring for
   * the numerically robust way this angle is recovered).
   *
   * Vertex crossings (two negative weights, a corner cut) are handled
   * separately by `_resolveVertexExit`, which searches for a genuinely
   * containing face directly rather than trying to continue a straight
   * line (there's no single well-defined "unfolded direction" once more
   * than one edge is involved) -- any leftover `(du, dv)` is simply
   * dropped after a vertex resolution.
   */
  _locateFace(startFaceIndex, u, v, du, dv) {
    let faceIndex = startFaceIndex;
    for (let step = 0; step < MAX_FACE_LOCATE_STEPS; step++) {
      const triangle = this._graph.triangles[faceIndex];
      const targetU = u + du;
      const targetV = v + dv;
      const exit = this._exitedBoundary(this._faceBarycentric(triangle, targetU, targetV));
      if (exit === null) {
        this._state = { faceIndex, u: targetU, v: targetV, h: this._state.h };
        return;
      }
      if (exit.type === 'vertex') {
        const vertexIndex = triangle.vertexIndices[exit.slot];
        const resolved = this._resolveVertexExit(
            vertexIndex, faceIndex, targetU, targetV);
        faceIndex = resolved.faceIndex;
        u = resolved.u;
        v = resolved.v;
        du = 0;
        dv = 0;
        continue;
      }
      const neighborIndex = triangle.neighbors[exit.slot];
      if (neighborIndex === -1) {
        u = targetU;
        v = targetV;
        du = 0;
        dv = 0;
        break; // mesh boundary -- clamp below
      }

      // Find the fraction `t` (0..1) along (u,v) -> (targetU,targetV)
      // where the crossed edge's own weight hits exactly 0 -- barycentric
      // weights are affine in (u, v), so this is a plain linear
      // interpolation to a zero-crossing.
      const startWeights = this._faceBarycentric(triangle, u, v);
      const targetWeights = this._faceBarycentric(triangle, targetU, targetV);
      const weightAt = (weights) => [weights.wb, weights.ws, weights.wd][exit.slot];
      const startWeight = weightAt(startWeights);
      const targetWeight = weightAt(targetWeights);
      const t = startWeight / (startWeight - targetWeight);
      const crossingU = u + du * t;
      const crossingV = v + dv * t;
      const remainingDu = du * (1 - t);
      const remainingDv = dv * (1 - t);

      const neighborTriangle = this._graph.triangles[neighborIndex];
      const edgeAxis = sharedEdgeAxis(this._graph, faceIndex, neighborIndex);
      const angle = dihedralRotationAngle(
          triangle.normal, neighborTriangle.normal, edgeAxis);
      const remaining3D = triangle.eX.clone().multiplyScalar(remainingDu)
          .addScaledVector(triangle.eY, remainingDv);
      const unfoldedRemaining3D = rotateAroundAxis(remaining3D, edgeAxis, angle);

      // The crossing point itself lies exactly on the shared edge, so
      // reprojecting *it* (unlike the overshoot) onto the neighbor's own
      // plane is always exact, regardless of the fold.
      const crossingPoint = facePoint(triangle, crossingU, crossingV);
      const localCrossing = crossingPoint.clone().sub(neighborTriangle.origin);

      faceIndex = neighborIndex;
      u = localCrossing.dot(neighborTriangle.eX);
      v = localCrossing.dot(neighborTriangle.eY);
      du = unfoldedRemaining3D.dot(neighborTriangle.eX);
      dv = unfoldedRemaining3D.dot(neighborTriangle.eY);
    }
    const triangle = this._graph.triangles[faceIndex];
    const clamped = this._clampToFace(triangle, u + du, v + dv);
    this._state = {
      faceIndex, u: clamped.u, v: clamped.v, h: this._state.h,
    };
  }

  /**
   * The barycentric-style weights of local `(u, v)` within `triangle`'s
   * own canonical frame (vertex B at the origin, S at `(sideLength, 0)`,
   * D at `(sideLength / 2, altitude)`) -- mirrors `IsometricPoint`'s own
   * convention, computed here in Cartesian rather than isometric terms
   * since the camera already tracks plain `(u, v)`.
   */
  _faceBarycentric(triangle, u, v) {
    const wd = v / triangle.altitude;
    const ws = (u - (wd * triangle.sideLength) / 2) / triangle.sideLength;
    const wb = 1 - ws - wd;
    return { wb, ws, wd };
  }

  /**
   * Which boundary of the triangle's own domain local `(u, v)` has
   * crossed, given its barycentric weights -- `null` if still inside. A
   * single negative weight is an ordinary edge crossing (the edge
   * opposite that weight's own vertex). Two negative weights means a
   * step cut across a corner without first grazing a single edge --
   * exits through the vertex shared by both crossed edges (the slot
   * whose own weight is *not* negative).
   */
  _exitedBoundary({ wb, ws, wd }) {
    const epsilon = -1e-9;
    const weights = [wb, ws, wd];
    const negativeSlots = weights
        .map((w, slot) => (w < epsilon ? slot : null))
        .filter((slot) => slot !== null);
    if (negativeSlots.length === 0) {
      return null;
    }
    if (negativeSlots.length === 1) {
      return { type: 'edge', slot: negativeSlots[0] };
    }
    const remainingSlot = [0, 1, 2].find((slot) => !negativeSlots.includes(slot));
    return { type: 'vertex', slot: remainingSlot ?? negativeSlots[0] };
  }

  /**
   * Resolves a corner-cut exit at `vertexIndex` by checking every face
   * touching that vertex (`vertexFaces[vertexIndex]`, a small, bounded
   * set) for whichever one actually contains the target point -- i.e.
   * has a non-negative worst barycentric weight -- rather than guessing
   * a single neighboring face and hoping repeated single-edge crossings
   * eventually find it (an early version of this method did that; it
   * can walk the *entire* fan without ever landing inside, since a
   * single-edge guess can enter a neighbor whose own wedge doesn't
   * contain the target either).
   *
   * Searches the fan in order of *hop distance* from `fromFaceIndex`
   * (its immediate neighbors first, then their neighbors, and so on),
   * not the fan's own raw storage order: a corner cut almost always
   * lands in an immediately-adjacent face (`_applyLocalStep`'s own
   * sub-stepping keeps individual steps small), so this finds the
   * geometrically nearest valid match rather than whichever technically-
   * valid but fan-distant one happens to appear earlier in the fan's own
   * storage order.
   *
   * If *no* face's own wedge contains the target direction at all (a
   * sharply convex vertex -- e.g. a spike tip -- where the surrounding
   * faces' wedges don't cover the full range of directions, leaving a
   * "missing" wedge past the tip that no face can represent), slides
   * along whichever *edge* incident to the vertex comes closest (in true
   * 3D distance) to the target, rather than collapsing all the way to
   * the vertex itself -- collapsing to the vertex made it a
   * self-reinforcing trap in an earlier version of this method, since a
   * fixed screen-drag direction, re-expressed through whichever face
   * happens to be current at the vertex, kept landing in *that* face's
   * own missing wedge too.
   */
  _resolveVertexExit(vertexIndex, fromFaceIndex, u, v) {
    const fromTriangle = this._graph.triangles[fromFaceIndex];
    const surfacePoint = facePoint(fromTriangle, u, v);
    const fan = this._graph.vertexFaces[vertexIndex];
    const fromIndex = fan.indexOf(fromFaceIndex);

    for (const faceIndex of vertexFanByHopDistance(fan, fromIndex)) {
      const triangle = this._graph.triangles[faceIndex];
      const local = surfacePoint.clone().sub(triangle.origin);
      const candidateU = local.dot(triangle.eX);
      const candidateV = local.dot(triangle.eY);
      const bary = this._faceBarycentric(triangle, candidateU, candidateV);
      if (Math.min(bary.wb, bary.ws, bary.wd) >= -1e-9) {
        return { faceIndex, u: candidateU, v: candidateV };
      }
    }

    const vertexSlot = fromTriangle.vertexIndices.indexOf(vertexIndex);
    const vertexPosition = triangleVertexPosition(fromTriangle, vertexSlot);
    const toSurfacePoint = surfacePoint.clone().sub(vertexPosition);

    let bestFaceIndex = fromFaceIndex;
    let bestU = 0;
    let bestV = 0;
    let bestDistSq = Infinity;
    for (const faceIndex of fan) {
      const triangle = this._graph.triangles[faceIndex];
      const slot = triangle.vertexIndices.indexOf(vertexIndex);
      for (const otherSlot of [(slot + 1) % 3, (slot + 2) % 3]) {
        const otherPosition = triangleVertexPosition(triangle, otherSlot);
        const edgeVector = otherPosition.clone().sub(vertexPosition);
        const t = THREE.MathUtils.clamp(
            toSurfacePoint.dot(edgeVector) / edgeVector.lengthSq(), 0, 1);
        const candidatePoint = vertexPosition.clone()
            .addScaledVector(edgeVector, t);
        const distSq = candidatePoint.distanceToSquared(surfacePoint);
        if (distSq < bestDistSq) {
          const local = candidatePoint.clone().sub(triangle.origin);
          bestDistSq = distSq;
          bestFaceIndex = faceIndex;
          bestU = local.dot(triangle.eX);
          bestV = local.dot(triangle.eY);
        }
      }
    }
    return { faceIndex: bestFaceIndex, u: bestU, v: bestV };
  }

  /** Clamps `(u, v)` to lie within `triangle`'s own bounds -- used only
   * at a mesh boundary, where there's no neighbor to hand off to, or if
   * `_locateFace`'s own step cap is somehow exhausted. */
  _clampToFace(triangle, u, v) {
    const { wb, ws, wd } = this._faceBarycentric(triangle, u, v);
    const clampedWb = Math.max(0, wb);
    const clampedWs = Math.max(0, ws);
    const clampedWd = Math.max(0, wd);
    const total = clampedWb + clampedWs + clampedWd;
    const normalizedWs = clampedWs / total;
    const normalizedWd = clampedWd / total;
    return {
      u: normalizedWs * triangle.sideLength
          + (normalizedWd * triangle.sideLength) / 2,
      v: normalizedWd * triangle.altitude,
    };
  }
}
