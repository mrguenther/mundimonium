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
    this._suppressed = false;
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

  /**
   * Runs `fn`, ignoring any 'change' events the source fires as a direct
   * side effect -- for a caller that needs to programmatically move the
   * source (e.g. resetting a camera to a new origin) without that motion
   * re-triggering debounced callbacks meant for actual user input.
   *
   * @param {() => void} fn
   */
  runSuppressed(fn) {
    this._suppressed = true;
    try {
      fn();
    } finally {
      this._suppressed = false;
    }
  }

  _onSourceChange() {
    if (this._suppressed) {
      return;
    }
    for (const listener of this._listeners) {
      clearTimeout(listener.timer);
      listener.timer = setTimeout(listener.callback, listener.debounceMs);
    }
  }
}
