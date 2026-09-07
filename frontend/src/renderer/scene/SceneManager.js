import * as THREE from 'three';

/**
 * Owns the Three.js scene, renderer, and render loop.
 *
 * Renders whatever camera is currently "active" via `setActiveController`
 * -- it doesn't know or care whether that's `OrbitCameraController`'s
 * perspective camera or `FlatMapCameraController`'s orthographic one.
 * `index.js` watches camera distance and calls `setActiveController` to
 * hard-cut between the two.
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

  /** @param {THREE.Object3D} object */
  removeFromScene(object) {
    this.scene.remove(object);
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
    if (!camera) {
      return;
    }
    const aspect = window.innerWidth / window.innerHeight;
    if (camera.isPerspectiveCamera) {
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    } else if (camera.isOrthographicCamera) {
      // Vertical extent (`top`) stays fixed on resize; only the
      // horizontal extent adjusts to the new aspect ratio.
      const halfHeight = camera.top;
      camera.left = -halfHeight * aspect;
      camera.right = halfHeight * aspect;
      camera.updateProjectionMatrix();
    }
  }
}
