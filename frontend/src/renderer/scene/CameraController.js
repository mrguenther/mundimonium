import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const DEFAULT_CHANGE_DEBOUNCE_MS = 150;

/**
 * Wraps a perspective camera + `OrbitControls` behind a small
 * `{ camera, update(deltaSeconds), dispose() }` shape, so `SceneManager`
 * doesn't need any `OrbitControls`-specific knowledge -- a future
 * `FlatMapCameraController` (orthographic, no orbiting) will implement the
 * same shape.
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

    this._changeListeners = [];
    this._controls.addEventListener('change', () => this._onControlsChange());
  }

  /**
   * Subscribes to camera changes, debounced so `callback` fires once
   * motion has settled (`debounceMs` after the last change) rather than
   * on every intermediate frame of a drag or damped motion.
   *
   * @param {() => void} callback
   * @param {number} [debounceMs]
   * @returns {() => void} Unsubscribe function.
   */
  onChange(callback, debounceMs = DEFAULT_CHANGE_DEBOUNCE_MS) {
    const listener = { callback, debounceMs, timer: null };
    this._changeListeners.push(listener);
    return () => {
      clearTimeout(listener.timer);
      this._changeListeners = this._changeListeners.filter(
          (other) => other !== listener);
    };
  }

  /** @param {number} _deltaSeconds */
  update(_deltaSeconds) {
    this._controls.update(); // required each frame when damping is on
  }

  dispose() {
    this._controls.dispose();
    for (const listener of this._changeListeners) {
      clearTimeout(listener.timer);
    }
    this._changeListeners = [];
  }

  _onControlsChange() {
    for (const listener of this._changeListeners) {
      clearTimeout(listener.timer);
      listener.timer = setTimeout(listener.callback, listener.debounceMs);
    }
  }
}
