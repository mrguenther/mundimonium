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
// aspect ratio -- see `_pixelDeltaToWorld`.
const VERTICAL_FOV_DEGREES = 50;

/**
 * Builds the static, per-triangle geometry a `SurfaceCameraController`
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

  return { vertexPositions, triangles, vertexFaces };
}

/** The surface point (before the hover-height offset) for a face patch's
 * local `(u, v)` -- see `buildSurfaceGraph`'s `eX`/`eY` convention. */
function facePoint(triangle, u, v) {
  return triangle.origin.clone()
      .addScaledVector(triangle.eX, u)
      .addScaledVector(triangle.eY, v);
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
 * The two exported-vertex indices `triangle` shares with its neighbor
 * across edge slot `edgeSlot` -- the edge opposite `triangle`'s own
 * vertex at that slot (see `mesh_export.py`'s own `adjacency` docstring).
 */
function sharedEdgeVertices(triangle, edgeSlot) {
  return triangle.vertexIndices.filter((_v, i) => i !== edgeSlot);
}

/**
 * Precomputes the geometry of the edge between `triangle` and its
 * neighbor across `edgeSlot`: the two shared vertex positions, a
 * canonical axis along the edge, and the signed rotation (`sign`,
 * `angle`) around that axis taking `triangle.normal` to the neighbor's
 * own normal.
 *
 * The two candidate sweep directions around an edge (the short arc
 * between the two normals, or its reflex complement) were both
 * considered for distinguishing convex from concave edges, matching an
 * intuitive "rolling ball" picture. Checked directly against this app's
 * own demo mesh's known convex and concave edges, though, the short arc
 * landed in a geometrically plausible position (offset toward/away from
 * the mesh's own center consistently with the edge's convexity) in both
 * cases, while a clean idealized model that would justify the reflex
 * rule for concave edges broke down entirely at the demo mesh's actual
 * notch angle. Given this is a visual-smoothness detail, not a
 * correctness-critical one, this implementation always takes the short
 * arc -- simpler, well-defined everywhere, and not visibly wrong on the
 * one real mesh this has been checked against. Revisit if panning across
 * a concave edge ever looks visibly wrong in practice.
 *
 * @returns {{ vertexIndices: [number, number], axis: THREE.Vector3,
 *   sign: number, angle: number } | null} `null` for a boundary edge
 *   (`edgeSlot`'s neighbor is `-1`).
 */
function computeEdgeGeometry(graph, triangleIndex, edgeSlot) {
  const triangle = graph.triangles[triangleIndex];
  const neighborIndex = triangle.neighbors[edgeSlot];
  if (neighborIndex === -1) {
    return null;
  }
  const neighbor = graph.triangles[neighborIndex];
  const vertexIndices = sharedEdgeVertices(triangle, edgeSlot);
  const [posA, posB] = vertexIndices.map((v) => graph.vertexPositions[v]);
  const axis = posB.clone().sub(posA).normalize();

  const cross = new THREE.Vector3().crossVectors(
      triangle.normal, neighbor.normal);
  const sign = Math.sign(cross.dot(axis)) || 1;
  const cosAngle = THREE.MathUtils.clamp(
      triangle.normal.dot(neighbor.normal), -1, 1);
  const angle = Math.acos(cosAngle);

  return { neighborIndex, vertexIndices, axis, sign, angle };
}

/**
 * A camera that hugs a triangle mesh's surface at a fixed hover height,
 * panning directly across faces and sweeping smoothly around edges and
 * vertices (convex or concave) rather than orbiting a fixed target --
 * see the plan's own "surface-following camera" design. Implements the
 * same `{ camera, update(deltaSeconds), onChange(callback), dispose() }`
 * shape as `OrbitCameraController`/`FlatMapCameraController`, so
 * `SceneManager`/`index.js` don't need to know it works differently
 * under the hood.
 *
 * Can't reuse `OrbitControls` -- its math assumes orbiting a fixed
 * target, not walking a surface -- so pointer events are handled here
 * directly instead.
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

    const startTriangle = this._graph.triangles[initialState.faceIndex ?? 0];
    const defaultH = startTriangle.sideLength * DEFAULT_HOVER_HEIGHT_FACE_WIDTHS;
    this._state = {
      patch: 'face',
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
   * pass into `getFlatMesh`/`getFlatItems`'s own `center` field. Exact
   * regardless of which patch type is currently active -- an edge or
   * vertex patch position is projected onto its own nearest face first.
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
    const { faceIndex, u, v } = this._faceLocalPosition();
    const triangle = this._graph.triangles[faceIndex];
    const { wb, ws } = this._faceBarycentric(triangle, u, v);
    return { face: faceIndex, b: wb * triangle.altitude, s: ws * triangle.altitude };
  }

  /**
   * Resets this controller to a fresh face-patch state at the given
   * face-local position -- used when *exiting* flat mode, to resume
   * exactly where the flat view's own last center was, rather than some
   * arbitrary default position.
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
    this._state = { patch: 'face', faceIndex, u, v, h };
    this._updateCameraFromState();
  }

  /** The current hover height, in the mesh's own real-distance units. */
  get hoverHeight() {
    return this._state.h;
  }

  /** The `side_length` of whichever face the camera is currently nearest
   * to -- used by `index.js` for the flat-mode threshold check. */
  get currentFaceSideLength() {
    return this._graph.triangles[this._faceLocalPosition().faceIndex]
        .sideLength;
  }

  // -------------------------------------------------------------------
  // Position/orientation from state
  // -------------------------------------------------------------------

  /**
   * The current `(faceIndex, u, v)` on the nearest face's own canonical
   * frame, regardless of which patch type is actually active -- computed
   * by projecting the current 3D surface point (without the hover-height
   * offset) into that face's frame.
   */
  _faceLocalPosition() {
    if (this._state.patch === 'face') {
      return {
        faceIndex: this._state.faceIndex, u: this._state.u, v: this._state.v,
      };
    }
    const surfacePoint = this._currentSurfacePoint();
    const faceIndex = this._nearestFaceIndex();
    const triangle = this._graph.triangles[faceIndex];
    const local = surfacePoint.clone().sub(triangle.origin);
    return {
      faceIndex, u: local.dot(triangle.eX), v: local.dot(triangle.eY),
    };
  }

  /** The face this controller's current edge/vertex patch is nearest to. */
  _nearestFaceIndex() {
    if (this._state.patch === 'face') {
      return this._state.faceIndex;
    }
    if (this._state.patch === 'edge') {
      return this._state.triangleIndex;
    }
    // Vertex patch: the anchor face in the vertex's own cyclic fan.
    return this._graph.vertexFaces[this._state.vertexIndex][
        this._state.faceSlot];
  }

  /** The current surface point (before the hover-height offset), for
   * whichever patch type is active. */
  _currentSurfacePoint() {
    const state = this._state;
    if (state.patch === 'face') {
      return facePoint(this._graph.triangles[state.faceIndex], state.u, state.v);
    }
    if (state.patch === 'edge') {
      const edge = computeEdgeGeometry(
          this._graph, state.triangleIndex, state.edgeSlot);
      const [posA, posB] = edge.vertexIndices.map(
          (v) => this._graph.vertexPositions[v]);
      return posA.clone().lerp(posB, state.u);
    }
    // Vertex patch.
    return this._graph.vertexPositions[state.vertexIndex];
  }

  /** The current outward normal direction, for whichever patch type is
   * active. */
  _currentNormal() {
    const state = this._state;
    if (state.patch === 'face') {
      return this._graph.triangles[state.faceIndex].normal;
    }
    if (state.patch === 'edge') {
      const edge = computeEdgeGeometry(
          this._graph, state.triangleIndex, state.edgeSlot);
      const startNormal = this._graph.triangles[state.triangleIndex].normal;
      return rotateAroundAxis(
          startNormal, edge.axis, edge.sign * edge.angle * state.t);
    }
    // Vertex patch: blend between consecutive faces in the vertex's own
    // cyclic fan -- the sketch's own explicitly-authorized approximation
    // (a normal blend, not an exact spherical-polygon boundary).
    const fan = this._graph.vertexFaces[state.vertexIndex];
    const normalA = this._graph.triangles[fan[state.faceSlot]].normal;
    const normalB = this._graph.triangles[
        fan[(state.faceSlot + 1) % fan.length]].normal;
    return normalA.clone().lerp(normalB, state.blendT).normalize();
  }

  _updateCameraFromState() {
    const surfacePoint = this._currentSurfacePoint();
    const normal = this._currentNormal();
    const position = surfacePoint.clone().addScaledVector(
        normal, this._state.h);
    this.camera.position.copy(position);

    // Look straight down at the surface point directly below, with `up`
    // taken from the current face's own `+v` axis (continuous across a
    // face's own interior by construction; see this file's own
    // docstring for the accepted small roll discontinuity at a patch
    // transition).
    const upHint = this._state.patch === 'face'
        ? this._graph.triangles[this._state.faceIndex].eY
        : new THREE.Vector3(0, 1, 0);
    this.camera.up.copy(upHint);
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
    const minH = this._graph.triangles[this._nearestFaceIndex()].sideLength
        * MIN_HOVER_HEIGHT_FACE_WIDTHS;
    this._state.h = Math.max(minH, this._state.h * zoomFactor);
    this._updateCameraFromState();
    this._dispatcher.dispatchEvent({ type: 'change' });
  }

  /**
   * Converts a screen-space pixel drag `(dx, dy)` into a local `(du, dv)`
   * step and applies it, transitioning between patch types as needed.
   *
   * The pixel-to-world conversion uses the same similar-triangles
   * relationship a hovering perspective camera implies: at hover height
   * `h` and vertical field of view `fov`, one vertical pixel of drag
   * corresponds to `2 * h * tan(fov / 2) / screenHeightPixels` world
   * units of tangential movement (horizontal analogously, scaled by
   * aspect ratio) -- this keeps pan speed visually consistent across
   * patch types and zoom levels.
   */
  _pan(dxPixels, dyPixels) {
    const h = this._state.h;
    const halfFovRadians = THREE.MathUtils.degToRad(VERTICAL_FOV_DEGREES / 2);
    const worldPerPixelY = (2 * h * Math.tan(halfFovRadians))
        / this._domElement.clientHeight;
    const worldPerPixelX = worldPerPixelY
        * (this._domElement.clientWidth / this._domElement.clientHeight);

    // Screen `+x` is the camera's own local `+x` (right); screen `+y`
    // (downward) is the camera's own local `-y` (up), matching how a
    // drag "down" should move the view "up" relative to the world (the
    // same convention `OrbitControls`' own panning uses).
    const cameraRight = new THREE.Vector3(1, 0, 0)
        .applyQuaternion(this.camera.quaternion);
    const cameraUp = new THREE.Vector3(0, 1, 0)
        .applyQuaternion(this.camera.quaternion);
    const worldDelta = cameraRight.multiplyScalar(-dxPixels * worldPerPixelX)
        .addScaledVector(cameraUp, dyPixels * worldPerPixelY);

    this._applyLocalStep(worldDelta);
    this._updateCameraFromState();
    this._dispatcher.dispatchEvent({ type: 'change' });
  }

  /**
   * Applies a world-space tangential step to the current patch, in local
   * `(du, dv)` terms, transitioning to a neighboring patch if the step
   * carries the position outside the current one's valid domain.
   */
  _applyLocalStep(worldDelta) {
    const state = this._state;
    if (state.patch === 'face') {
      const triangle = this._graph.triangles[state.faceIndex];
      const du = worldDelta.dot(triangle.eX);
      const dv = worldDelta.dot(triangle.eY);
      this._stepWithinFace(state.faceIndex, state.u + du, state.v + dv);
      return;
    }
    if (state.patch === 'edge') {
      const edge = computeEdgeGeometry(
          this._graph, state.triangleIndex, state.edgeSlot);
      const along = edge.vertexIndices.map(
          (v) => this._graph.vertexPositions[v]);
      const edgeLength = along[1].distanceTo(along[0]);
      const axisStep = worldDelta.dot(edge.axis) / edgeLength;
      // Movement perpendicular to the edge axis (i.e. "off the seam" back
      // onto one of the two adjacent faces) re-enters that face directly,
      // projected via the current surface point plus the perpendicular
      // component of `worldDelta`.
      const perpendicular = worldDelta.clone()
          .addScaledVector(edge.axis, -worldDelta.dot(edge.axis));
      if (perpendicular.length() > 1e-9) {
        const surfacePoint = this._currentSurfacePoint()
            .addScaledVector(edge.axis, axisStep * edgeLength)
            .add(perpendicular);
        this._enterFaceAtPoint(state.triangleIndex, surfacePoint);
        return;
      }
      this._stepAlongEdge(state, state.u + axisStep);
      return;
    }
    // Vertex patch: any pan moves off the vertex immediately, resolved by
    // re-entering whichever face the resulting surface point projects
    // onto most naturally.
    const surfacePoint = this._currentSurfacePoint().add(worldDelta);
    this._enterFaceAtPoint(this._nearestFaceIndex(), surfacePoint);
  }

  /** Moves within (or transitions out of) a face patch to local `(u, v)`. */
  _stepWithinFace(faceIndex, u, v) {
    const triangle = this._graph.triangles[faceIndex];
    const barycentric = this._faceBarycentric(triangle, u, v);
    const edgeSlot = this._exitedEdgeSlot(barycentric);
    if (edgeSlot === null) {
      this._state = { patch: 'face', faceIndex, u, v, h: this._state.h };
      return;
    }
    const edge = computeEdgeGeometry(this._graph, faceIndex, edgeSlot);
    if (edge === null) {
      // Mesh boundary: clamp to the edge rather than falling off the mesh.
      const clamped = this._clampToFace(triangle, u, v);
      this._state = {
        patch: 'face', faceIndex, u: clamped.u, v: clamped.v,
        h: this._state.h,
      };
      return;
    }
    const surfacePoint = facePoint(triangle, u, v);
    this._enterEdgeFromFace(faceIndex, edgeSlot, edge, surfacePoint);
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

  /** Which of the 3 edges `(u, v)` has crossed, or `null` if still
   * inside the triangle -- the edge opposite whichever barycentric
   * weight went negative. */
  _exitedEdgeSlot({ wb, ws, wd }) {
    const epsilon = -1e-9;
    if (wb < epsilon) return 0;
    if (ws < epsilon) return 1;
    if (wd < epsilon) return 2;
    return null;
  }

  /** Clamps `(u, v)` to lie within `triangle`'s own bounds -- used only
   * at a mesh boundary, where there's no neighbor to hand off to. */
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

  /** Transitions from a face patch onto the edge patch bordering it at
   * `edgeSlot`, entering at `theta = 0` (still exactly at the face's own
   * plane) for continuity. */
  _enterEdgeFromFace(faceIndex, edgeSlot, edge, surfacePoint) {
    const [posA, posB] = edge.vertexIndices.map(
        (v) => this._graph.vertexPositions[v]);
    const edgeVector = posB.clone().sub(posA);
    const u = surfacePoint.clone().sub(posA).dot(edgeVector)
        / edgeVector.lengthSq();
    if (u <= 0 || u >= 1) {
      this._enterVertexNear(edge.vertexIndices[u <= 0 ? 0 : 1]);
      return;
    }
    this._state = {
      patch: 'edge', triangleIndex: faceIndex, edgeSlot, u, t: 0,
      h: this._state.h,
    };
  }

  /** Moves along an edge patch to arc-length parameter `u`, transitioning
   * to the far face (if `u` exceeds `[0, 1]`) or a vertex patch. */
  _stepAlongEdge(state, u) {
    if (u < 0 || u > 1) {
      const edge = computeEdgeGeometry(
          this._graph, state.triangleIndex, state.edgeSlot);
      this._enterVertexNear(edge.vertexIndices[u < 0 ? 0 : 1]);
      return;
    }
    // `t` (how far around the dihedral sweep) tracks `u` isn't directly
    // meaningful here -- `t` is re-derived from the *actual* surface
    // point in `_currentSurfacePoint`/`_currentNormal` via `state.t`,
    // which this method leaves unchanged; only `u` (position along the
    // edge) is a free pan parameter. The dihedral angle itself is fixed
    // once entering the edge patch precisely on one face's own plane
    // (`t = 0`) -- panning along an edge doesn't sweep the dihedral, only
    // moving *across* it (handled in `_applyLocalStep`'s perpendicular
    // branch) does.
    this._state = { ...state, u };
  }

  /** Enters the face patch containing (or nearest to) `surfacePoint`,
   * starting from `hintFaceIndex`'s own neighborhood -- used for edge/
   * vertex-patch exits, where the destination face isn't already known
   * precisely. */
  _enterFaceAtPoint(hintFaceIndex, surfacePoint) {
    const triangle = this._graph.triangles[hintFaceIndex];
    const local = surfacePoint.clone().sub(triangle.origin);
    const u = local.dot(triangle.eX);
    const v = local.dot(triangle.eY);
    this._stepWithinFace(hintFaceIndex, u, v);
  }

  /** Enters a vertex patch at `vertexIndex`, anchored at whichever
   * adjacent face is first in its own cyclic fan (an arbitrary but
   * stable starting point -- see the vertex-patch blend's own docstring
   * for why exact positioning here isn't load-bearing). */
  _enterVertexNear(vertexIndex) {
    this._state = {
      patch: 'vertex', vertexIndex, faceSlot: 0, blendT: 0,
      h: this._state.h,
    };
  }
}
