import * as THREE from 'three';

/**
 * Owns the Three.js scene, renderer, and render loop.
 *
 * Renders whatever camera is currently "active" via `setActiveController`
 * -- it doesn't know or care whether that's an orbit-controlled
 * perspective camera (this phase) or, later, a flat-map orthographic one.
 * That's the seam a future camera-distance-based mode switch hooks into:
 * something outside this class watches distance-to-target each frame and
 * calls `setActiveController` with a different controller: no change
 * needed here.
 */
export class SceneManager {
  /** @param {HTMLElement} container - Element the canvas is appended to. */
  constructor(container) {
    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(this.renderer.domElement);

    this._activeController = null;
    this._clock = new THREE.Clock();

    window.addEventListener('resize', () => this._onResize());
  }

  /**
   * @param {{ camera: THREE.Camera, update: (deltaSeconds: number) => void }} controller
   */
  setActiveController(controller) {
    this._activeController = controller;
    this._onResize(); // the new camera needs the current aspect ratio
  }

  /** @param {THREE.Object3D} object */
  addToScene(object) {
    this.scene.add(object);
  }

  /** Starts the render loop. */
  start() {
    this.renderer.setAnimationLoop(() => this._renderFrame());
  }

  _renderFrame() {
    if (!this._activeController) {
      return;
    }
    const deltaSeconds = this._clock.getDelta();
    this._activeController.update(deltaSeconds);
    this.renderer.render(this.scene, this._activeController.camera);
  }

  _onResize() {
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    const camera = this._activeController && this._activeController.camera;
    if (camera && camera.isPerspectiveCamera) {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }
  }
}
