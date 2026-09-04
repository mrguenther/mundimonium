const DEFAULT_DEBOUNCE_MS = 150;

/**
 * Wraps a `THREE.EventDispatcher`-style source's `'change'` event into a
 * debounced `onChange(callback)` subscription API -- shared by both
 * `OrbitCameraController` and `FlatMapCameraController`, which each wire
 * their own `OrbitControls` instance's `'change'` event through one of
 * these.
 */
export class ChangeDebouncer {
  /** @param {{ addEventListener: (type: string, listener: () => void) => void }} source */
  constructor(source) {
    this._listeners = [];
    source.addEventListener('change', () => this._onSourceChange());
  }

  /**
   * Subscribes to the source's changes, debounced so `callback` fires
   * once motion has settled (`debounceMs` after the last change) rather
   * than on every intermediate frame of a drag or damped motion.
   *
   * @param {() => void} callback
   * @param {number} [debounceMs]
   * @returns {() => void} Unsubscribe function.
   */
  onChange(callback, debounceMs = DEFAULT_DEBOUNCE_MS) {
    const listener = { callback, debounceMs, timer: null };
    this._listeners.push(listener);
    return () => {
      clearTimeout(listener.timer);
      this._listeners = this._listeners.filter((other) => other !== listener);
    };
  }

  /** Cancels every pending debounced callback and clears all listeners. */
  dispose() {
    for (const listener of this._listeners) {
      clearTimeout(listener.timer);
    }
    this._listeners = [];
  }

  _onSourceChange() {
    for (const listener of this._listeners) {
      clearTimeout(listener.timer);
      listener.timer = setTimeout(listener.callback, listener.debounceMs);
    }
  }
}
