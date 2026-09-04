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
    this._controls.update();
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
