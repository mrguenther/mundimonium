import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { ChangeDebouncer } from './ChangeDebouncer.js';

/**
 * Wraps a perspective camera + `OrbitControls` behind a small
 * `{ camera, update(deltaSeconds), onChange(callback), dispose() }` shape,
 * so `SceneManager` doesn't need any `OrbitControls`-specific knowledge --
 * `FlatMapCameraController` implements the same shape for the flat 2D view.
 */
export class OrbitCameraController {
  /**
   * @param {HTMLElement} domElement - The renderer's canvas, for pointer
   *   events.
   * @param {{ target?: THREE.Vector3 }} [options]
   */
  constructor(domElement, { target = new THREE.Vector3(0, 0, 0) } = {}) {
    this.camera = new THREE.PerspectiveCamera(
        50, window.innerWidth / window.innerHeight, 0.01, 1000);
    this.camera.position.set(0, 0, 5);

    this._controls = new OrbitControls(this.camera, domElement);
    this._controls.target.copy(target);
    this._controls.minDistance = 1.5;
    this._controls.maxDistance = 50;
    this._controls.enableDamping = true;
    // Right-drag orbits; left is deliberately left unbound, reserved for
    // future click/drag UI interaction (selection, etc.) rather than
    // camera movement -- this way, right-drag is consistently "move the
    // camera" across every view (`FlatMapCameraController`'s own pan is
    // already on the right button by default; `SurfaceCameraController`'s
    // hand-rolled panning already checks for it explicitly). Panning the
    // orbit camera itself is disabled entirely, not just left unbound to
    // a different button: it would let the camera drift away from
    // orbiting the origin, which nothing here is designed to handle.
    this._controls.enablePan = false;
    this._controls.mouseButtons = {
      LEFT: null,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    };
    this._controls.update();

    this._changeDebouncer = new ChangeDebouncer(this._controls);
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
    this._controls.update(); // required each frame when damping is on
  }

  /**
   * Enables or disables pointer interaction, without disposing anything
   * -- used while this controller is inactive (see `SceneManager.
   * setActiveController`), since `OrbitControls` keeps listening on its
   * `domElement` regardless of which controller is actually rendering.
   *
   * @param {boolean} enabled
   */
  setEnabled(enabled) {
    this._controls.enabled = enabled;
  }

  dispose() {
    this._controls.dispose();
    this._changeDebouncer.dispose();
  }
}

/**
 * Wraps an orthographic camera + `OrbitControls` (rotation disabled --
 * pan and zoom only) behind the same `{ camera, update(deltaSeconds),
 * onChange(callback), dispose() }` shape as `OrbitCameraController`, for
 * the flat 2D map view.
 */
export class FlatMapCameraController {
  /**
   * @param {HTMLElement} domElement - The renderer's canvas, for pointer
   *   events.
   * @param {{ halfHeight?: number, target?: THREE.Vector3 }} [options] -
   *   `halfHeight` is the world-space vertical extent visible at zoom 1,
   *   in the tessellation's own true-distance units (same units
   *   `Tessellation.flatten_region` outputs) -- typically set to roughly
   *   the 3D camera's altitude above the surface at the moment of
   *   switching into flat mode, for visual continuity across the cut.
   */
  constructor(
      domElement,
      { halfHeight = 1.0, target = new THREE.Vector3(0, 0, 0) } = {}) {
    const aspect = window.innerWidth / window.innerHeight;
    this.camera = new THREE.OrthographicCamera(
        -halfHeight * aspect, halfHeight * aspect, halfHeight, -halfHeight,
        0.01, 1000);
    this.camera.position.set(target.x, target.y, 10);
    this.camera.lookAt(target);

    this._controls = new OrbitControls(this.camera, domElement);
    this._controls.target.copy(target);
    this._controls.enableRotate = false; // pan + zoom only, no orbiting
    this._controls.minZoom = 0.01;
    this._controls.update();

    this._changeDebouncer = new ChangeDebouncer(this._controls);

    // Unset (no clamping) unless a caller needs to keep panning inside a
    // bounded region -- see `setMaxPanRadius`.
    this._maxPanRadius = Infinity;
  }

  /**
   * @param {() => void} callback
   * @param {number} [debounceMs]
   * @returns {() => void} Unsubscribe function.
   */
  onChange(callback, debounceMs) {
    return this._changeDebouncer.onChange(callback, debounceMs);
  }

  /**
   * Sets how far the camera may pan from wherever it was last `recenter()`-
   * ed, in world units -- for a view (e.g. `HyperbolicTessellation`'s)
   * whose projection is only numerically valid within a bounded radius of
   * its own center. `Infinity` (the default) disables clamping.
   *
   * @param {number} radius
   */
  setMaxPanRadius(radius) {
    this._maxPanRadius = radius;
  }

  /** @param {number} _deltaSeconds */
  update(_deltaSeconds) {
    this._controls.update();

    // Pin the camera at the boundary rather than letting a fast drag
    // overshoot it -- `recenter()` always resets to exactly `(0, 0)`, so
    // clamping distance-from-origin here is exactly "distance panned
    // since the view was last centered." Deliberately not run through
    // `_changeDebouncer.runSuppressed` (unlike `recenter()`): this only
    // ever *constrains* where a real pan settles, it never introduces a
    // spurious extra 'change' cycle on its own, so the debounced
    // `onChange` should still fire normally once the drag pauses or
    // releases.
    const { x, y } = this.camera.position;
    const distance = Math.hypot(x, y);
    if (distance > this._maxPanRadius) {
      const scale = this._maxPanRadius / distance;
      this.camera.position.x *= scale;
      this.camera.position.y *= scale;
      this._controls.target.x = this.camera.position.x;
      this._controls.target.y = this.camera.position.y;
    }
  }

  /**
   * Resets the camera (and `OrbitControls`' own target, which must stay
   * in sync with it -- see `_controls`) to `(0, 0)`, keeping the current
   * height/zoom -- for when a fresh `flatten_region` call has just been
   * re-centered on wherever the view currently is, so panning offsets
   * accumulated against the *old* center should reset to zero rather
   * than carrying over against the new one.
   *
   * Runs through `_changeDebouncer.runSuppressed` because `OrbitControls`
   * detects "change" by diffing against its own last-seen position/
   * target (not by anything panning-specific), so this reset would
   * otherwise fire a synthetic 'change' event whenever the camera had
   * actually panned -- re-triggering the very `onChange` callback this
   * recenter is a response to, one redundant extra time.
   */
  recenter() {
    this._changeDebouncer.runSuppressed(() => {
      this.camera.position.set(0, 0, this.camera.position.z);
      this._controls.target.set(0, 0, 0);
      this._controls.update();
    });
  }

  /** @param {boolean} enabled */
  setEnabled(enabled) {
    this._controls.enabled = enabled;
  }

  dispose() {
    this._controls.dispose();
    this._changeDebouncer.dispose();
  }
}
